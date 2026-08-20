# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ASGI middleware：接管 `POST /v1/audio/transcriptions`（纯 ASGI，零框架侵入）。

职责（对应 spec「服务架构」「请求参数矩阵」「audio_url 安全」「长音频」）：

- 纯函数（可 stub 离线测试）：
  - `validate_request_params`：timestamp_granularities × response_format 参数矩阵校验；
  - `check_audio_url`：audio_url 的 SSRF 校验（仅 https + 全 IP 内网/环回/链路本地拒绝）；
  - `error_response`：OpenAI 风格错误体构造；
  - `scan_flag_values` / `extract_served_model_names` / `gpu_memory_utilization_specified`：
    argv 扫描纯函数（serve.py 组装复用）。
- `TranscriptionsMiddleware`：只拦 POST + /v1/audio/transcriptions，其余路径零干预透传；
  - 标准模式（无 segment 粒度）：1200s 分块 ASR（信号量限并发）→ OpenAI 标准响应；
  - segment 模式：调度器排队准入后，ASR+对齐 与 diarization 两阶段线程并行，
    pipeline 纯函数组装 segment + speakerSummary 响应；
  - 排队/执行期间客户端断连：monitor 任务读 receive() 的 http.disconnect 取消主任务
    （scheduler.slot 的取消语义保证排队位/许可清理）。
"""

import asyncio
import contextlib
import ipaddress
import json
import logging
import socket
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..inference.utils import (
    MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    SAMPLE_RATE,
    merge_align_results,
    merge_languages,
    normalize_audio_input,
    offset_align_result,
    parse_asr_output,
    split_audio_into_chunks,
)
from .extensions import ExtensionState, budget_devices
from .pipeline import build_segment_response, language_name_to_code, resolve_language_name
from .scheduler import estimate_task_need_mb

__all__ = [
    "TranscriptionsMiddleware",
    "validate_request_params",
    "check_audio_url",
    "error_response",
    "scan_flag_values",
    "extract_served_model_names",
    "gpu_memory_utilization_specified",
    "build_asr_messages",
    "build_asr_prompt",
    "download_audio",
]

logger = logging.getLogger(__name__)

# starlette 随 vLLM 依赖存在，用于 multipart 表单解析；导入失败时目标路径透传
# 给 vLLM 原生处理（记日志），其余路径不受影响
try:
    from starlette.requests import Request as _StarletteRequest

    _STARLETTE_AVAILABLE = True
except Exception:  # pragma: no cover - 仅缺依赖环境触发
    _StarletteRequest = None
    _STARLETTE_AVAILABLE = False

#: audio_url 下载超时（秒）与流式下载块大小
DOWNLOAD_TIMEOUT_SECONDS = 60.0
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

#: OpenAI 规范 transcriptions 支持的 response_format 集合（本服务三格式）
_RESPONSE_FORMATS = ("json", "text", "verbose_json")
#: OpenAI 规范 timestamp_granularities 合法值（word 为合法值但 v1 不支持）
_GRANULARITY_VALUES = ("segment", "word")


# ---------------------------------------------------------------------------
# 纯函数：参数矩阵 / SSRF / 错误体 / argv 扫描
# ---------------------------------------------------------------------------


def validate_request_params(
    timestamp_granularities: List[str],
    response_format: str,
) -> None:
    """按 spec「请求参数矩阵」校验参数组合，违规抛 ``ValueError``（中文消息）。

    矩阵：

    - 无 segment → OpenAI 标准三格式（json 缺省 / text / verbose_json）；
    - segment + 缺省或 json → 扩展响应（通过）；
    - segment + text / verbose_json → 400（不兼容）；
    - 含 word（任意 response_format）→ 400（v1 不支持 word 粒度）；
    - 其他非法粒度值 → 400；
    - 非法 response_format（三格式之外）→ 400。
    """
    fmt = str(response_format).strip().lower() if response_format else "json"
    granularities = [str(g).strip() for g in (timestamp_granularities or []) if str(g).strip()]

    for g in granularities:
        if g not in _GRANULARITY_VALUES:
            raise ValueError(
                f"非法的 timestamp_granularities 值: {g!r}，仅支持 segment"
                "（word 粒度 v1 暂不支持）"
            )
    if "word" in granularities:
        raise ValueError(
            "v1 不支持 word 时间戳粒度；本服务支持 segment 粒度，"
            "请移除 word 或改用 timestamp_granularities[]=segment"
        )
    if fmt not in _RESPONSE_FORMATS:
        raise ValueError(
            f"非法的 response_format: {response_format!r}，可选值: json / text / verbose_json"
        )
    if "segment" in granularities and fmt in ("text", "verbose_json"):
        raise ValueError(
            f"segment 时间戳粒度与 response_format={fmt} 不兼容，"
            "请移除 timestamp_granularities 中的 segment 或改用 response_format=json"
        )


def _ip_is_forbidden(ip: "ipaddress._BaseAddress") -> bool:
    """判断解析出的 IP 是否命中禁入集合（环回/私网/链路本地/未指定/保留/组播）。

    覆盖：RFC1918（10./172.16-31./192.168.）、RFC4193（fd00::/8）、
    链路本地（169.254.0.0/16 与 fe80::/10，含云元数据 169.254.169.254）、
    0.0.0.0 与 ::、环回（127.0.0.0/8 与 ::1）；IPv4-mapped IPv6 归一后判定。
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def check_audio_url(url: str) -> str:
    """audio_url SSRF 校验（spec「audio_url 安全」），违规抛 ``ValueError``（中文消息）。

    - 仅允许 https 协议（拒绝 http/file/ftp 等）；
    - ``socket.getaddrinfo`` 解析出的**全部 IP** 逐个校验（防 DNS rebinding 基础形态），
      任一命中禁入集合（见 `_ip_is_forbidden`）即拒绝；
    - DNS 解析失败同样拒绝。

    Returns:
        str: 校验通过的原样 URL。
    """
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme.lower() != "https":
        raise ValueError(
            f"audio_url 仅支持 https 协议，收到: {parsed.scheme or '（空）'}"
        )
    host = parsed.hostname
    if not host:
        raise ValueError("audio_url 缺少有效主机名")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ValueError(f"audio_url 端口非法: {text}") from exc

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception as exc:
        raise ValueError(f"audio_url 主机名解析失败: {host}（{exc}）") from exc

    for info in infos:
        ip_text = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError as exc:
            raise ValueError(f"audio_url 解析出非法地址: {ip_text}") from exc
        if _ip_is_forbidden(ip):
            raise ValueError(
                f"audio_url 解析到内网/环回/链路本地等受限地址: {ip_text}，已拒绝"
            )
    return text


def error_response(status: int, message: str, err_type: str) -> Dict[str, Any]:
    """构造 OpenAI 风格错误响应（JSONResponse 风格 dict，纯函数）。

    形如 ``{"status": 400, "body": {"error": {"message": ..., "type": ..., "code": null}}}``，
    由 `_send_error` 落地为真实 HTTP 响应。
    """
    return {
        "status": int(status),
        "body": {"error": {"message": str(message), "type": str(err_type), "code": None}},
    }


def scan_flag_values(argv: List[str], flag: str) -> List[str]:
    """扫描 argv 中某 flag 的全部取值（兼容 ``--flag value`` 与 ``--flag=value``）。"""
    values: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag and i + 1 < len(argv):
            values.append(argv[i + 1])
            i += 2
            continue
        if token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
            i += 1
            continue
        i += 1
    return values


def extract_served_model_names(argv: List[str]) -> List[str]:
    """从剩余 argv 提取 served model names（纯函数，serve.py 复用）。

    优先 ``--served-model-name``（可多次）；缺省回退 ``--model`` / ``--model-tag``
    的值；再缺省返回空列表（表示服务端不做 model 名校验）。
    """
    names = scan_flag_values(argv, "--served-model-name")
    if names:
        return names
    for flag in ("--model", "--model-tag"):
        values = scan_flag_values(argv, flag)
        if values:
            return [values[-1]]
    return []


def gpu_memory_utilization_specified(argv: List[str]) -> bool:
    """判断用户 argv 是否显式指定了 ``--gpu-memory-utilization``（含 = 形式）。"""
    return any(
        token == "--gpu-memory-utilization"
        or token.startswith("--gpu-memory-utilization=")
        for token in argv
    )


# ---------------------------------------------------------------------------
# prompt 构造（复刻 Qwen3ASRModel._build_messages / _build_text_prompt）
# ---------------------------------------------------------------------------


def build_asr_messages(context: str, audio_payload: Any = "") -> List[Dict[str, Any]]:
    """构造 chat template 消息（system 上下文 + user 音频槽位，audio 占位为空串）。"""
    return [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
    ]


def build_asr_prompt(processor: Any, context: str, force_language: Optional[str]) -> str:
    """复刻 ``Qwen3ASRModel._build_text_prompt``：chat template + 可选强制语言后缀。

    processor 为 CPU 常驻的 Qwen3ASRProcessor；force_language 提供时追加
    ``language X<asr_text>`` 请求模型输出纯转写文本。
    """
    base = processor.apply_chat_template(
        build_asr_messages(context, ""),
        add_generation_prompt=True,
        tokenize=False,
    )
    if force_language:
        base = base + f"language {force_language}{'<asr_text>'}"
    return base


# ---------------------------------------------------------------------------
# 音频获取 / 解码 / 生成
# ---------------------------------------------------------------------------


def download_audio(url: str, max_bytes: int, timeout: float = DOWNLOAD_TIMEOUT_SECONDS) -> bytes:
    """流式下载 audio_url（Request 带 User-Agent；超时 timeout 秒；累计超限即中止）。

    Raises:
        ValueError: 下载内容累计超过 max_bytes。
        urllib.error.URLError / 其他网络异常: 由调用方统一映射 400。
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "qwen-asr-serve/segment-api"}
    )
    parts: List[bytes] = []
    total = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            piece = response.read(_DOWNLOAD_CHUNK_BYTES)
            if not piece:
                break
            total += len(piece)
            if total > int(max_bytes):
                raise ValueError(
                    f"audio_url 下载内容累计 {total} 字节，超过上限 {int(max_bytes)} 字节，已中止"
                )
            parts.append(piece)
    return b"".join(parts)


def decode_audio_bytes(data: bytes):
    """解码音频字节为 16k 单声道 float32 波形（复用 ``normalize_audio_input`` 归一）。

    延迟 import soundfile/numpy（随主依赖存在）。解码失败向上抛异常，
    由调用方映射 415。
    """
    import io

    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return normalize_audio_input((np.asarray(audio), int(sr)))


async def engine_generate(ext: ExtensionState, prompt: str, wav: Any) -> str:
    """经 vLLM engine client 生成一个分块的 ASR 输出（兼容层）。

    - engine_client 由 serve 钩子经 vLLM ``init_app_state`` 捕获后注入 ext，
      缺省时由 middleware 请求期懒解析（``scope["app"].state``）兜底；
    - ``SamplingParams`` 延迟 import（vllm 顶层 import 在无 vLLM 环境不可用）；
    - 防御式迭代：优先按异步生成器逐块消费取最后 output；若当前版本返回
      awaitable（非流式）则直接 await；取 ``outputs[0].text`` 前做空值防御。
    """
    from vllm import SamplingParams  # 延迟 import：保持本模块可离线 stub 测试

    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    request_id = f"qwen-asr-transcription-{uuid.uuid4().hex}"
    generator = ext.engine_client.generate(
        {"prompt": prompt, "multi_modal_data": {"audio": [wav]}},
        sampling_params,
        request_id=request_id,
    )
    final_output = None
    if hasattr(generator, "__aiter__"):
        async for output in generator:
            final_output = output
    else:
        final_output = await generator
    outputs = getattr(final_output, "outputs", None)
    if not outputs:
        return ""
    return getattr(outputs[0], "text", "") or ""


# ---------------------------------------------------------------------------
# segment 模式线程体（to_thread 执行；GPU 前向经进程级锁串行化）
# ---------------------------------------------------------------------------


def _align_batch(ext: ExtensionState, batch: List[Tuple[Any, str, str, float]]) -> List[Any]:
    """按 align_batch 批量调用 aligner.align 并做 chunk 偏移修正（锁内前向）。"""
    with ext.aligner_lock:
        results = ext.aligner.align(
            audio=[(cwav, SAMPLE_RATE) for cwav, _, _, _ in batch],
            text=[txt for _, txt, _, _ in batch],
            language=[lang for _, _, lang, _ in batch],
        )
    return [offset_align_result(r, off) for r, (_, _, _, off) in zip(results, batch)]


def _run_asr_align(
    ext: ExtensionState,
    wav: Any,
    context: str,
    force_language: Optional[str],
    loop: asyncio.AbstractEventLoop,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[str, str, Any]:
    """segment 模式线程体：180s 分块 ASR（批量提交）→ 批量对齐 → 偏移合并。

    ASR 生成经 ``run_coroutine_threadsafe`` 批量提交回主事件循环后按序收集
    （与 SDK ``_infer_asr_vllm`` 的批量提交方式一致，长音频多块不必逐块串行
    等待）；对齐在进程级锁内按 align_batch_size 批量执行。
    ``cancel_event`` 置位（客户端断连）时在块间/对齐批次间尽快中止后续处理，
    并对已提交的剩余分块生成调用 ``future.cancel()``（取消会传播到事件循环中
    的生成协程，在 await 点中断；GPU 上正在执行的引擎侧前向由 vLLM 引擎自行
    abort，属 spec 声明的尽力取消）。
    返回 ``(完整文本, 合并语言, 合并对齐结果或 None)``。
    """

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    chunks = split_audio_into_chunks(wav, SAMPLE_RATE, MAX_FORCE_ALIGN_INPUT_SECONDS)
    # 批量提交全部分块生成请求（engine 内部排队调度），再按序收集结果
    futures = []
    for cwav, _offset in chunks:
        prompt = build_asr_prompt(ext.processor, context, force_language)
        futures.append(
            asyncio.run_coroutine_threadsafe(engine_generate(ext, prompt, cwav), loop)
        )
    per_chunk: List[Tuple[Any, str, str, float]] = []
    try:
        for (cwav, offset), future in zip(chunks, futures):
            if _cancelled():
                raise RuntimeError("客户端已断开，中止后续 ASR 分块处理")
            raw = future.result()
            lang, txt = parse_asr_output(raw, user_language=force_language)
            per_chunk.append((cwav, txt, lang, offset))
    except BaseException:
        # 取消/异常路径：尽力取消尚未完成的全部分块生成（对已完成/不可取消
        # 的 future 无副作用），避免剩余生成跑完却无人消费
        for future in futures:
            future.cancel()
        raise

    full_text = "".join(txt for _, txt, _, _ in per_chunk)
    merged_lang = merge_languages([lang for _, _, lang, _ in per_chunk])

    aligned: List[Any] = []
    batch: List[Tuple[Any, str, str, float]] = []
    for item in per_chunk:
        if _cancelled():
            raise RuntimeError("客户端已断开，中止后续对齐处理")
        if not item[1].strip():
            continue  # 空文本块跳过对齐（与 SDK transcribe 行为一致）
        batch.append(item)
        if len(batch) >= int(ext.align_batch_size):
            aligned.extend(_align_batch(ext, batch))
            batch = []
    if batch:
        aligned.extend(_align_batch(ext, batch))

    merged = merge_align_results([r for r in aligned if r is not None])
    return full_text, merged_lang, merged


def _run_diarize(ext: ExtensionState, wav: Any, min_speakers: Optional[int], max_speakers: Optional[int]) -> List[Any]:
    """segment 模式线程体：整段音频一次 diarization（SpeakerDiarizer 内部自带锁）。"""
    return ext.diarizer.diarize(
        (wav, SAMPLE_RATE),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def _release_gpu_cache() -> None:
    """归还 PyTorch 缓存分配器保留的空闲显存块（仅本进程的扩展侧分配）。

    对齐/说话人前向结束后，缓存分配器会把空闲块留在自己手里不归还驱动，
    ``mem_get_info`` 空闲显存随即永久偏低——调度器准入据此误判"显存不足"，
    后续任务全部滞留等待队列（首任务成功、后续全卡死的假死锁；缓存的
    空闲块实际可复用，并非真被占满）。任务收尾（含异常路径）显式
    ``empty_cache`` 让队首重准入看到真实空闲显存；模型权重等在用块不受
    影响，vLLM 引擎进程的显存池也不受影响（独立进程独立分配器）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # 防御性：释放失败不影响主流程（仅恢复到旧行为）
        logger.debug("torch.cuda.empty_cache 失败", exc_info=True)


def _parse_optional_int(value: Any) -> Optional[int]:
    """解析可选整数字段；空/缺省为 None，非整数抛 ValueError。"""
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        raise ValueError(f"min_speakers/max_speakers 须为整数，收到: {value!r}")


def _merged_language_code(language: str) -> str:
    """合并语言串（如 "Chinese,English"）逐段转 BCP-47 码输出。"""
    parts = [p.strip() for p in str(language or "").split(",") if p.strip()]
    if not parts:
        return ""
    return ",".join(language_name_to_code(p) for p in parts)


# ---------------------------------------------------------------------------
# ASGI 响应原语
# ---------------------------------------------------------------------------


async def _send_body(send, status: int, body: bytes, content_type: bytes) -> None:
    """发送完整 ASGI 响应（单 body 分片）。"""
    await send(
        {
            "type": "http.response.start",
            "status": int(status),
            "headers": [
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _send_json(send, status: int, payload: Dict[str, Any]) -> None:
    await _send_body(send, status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), b"application/json; charset=utf-8")


async def _send_text(send, status: int, text: str) -> None:
    await _send_body(send, status, str(text).encode("utf-8"), b"text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# middleware
# ---------------------------------------------------------------------------


class TranscriptionsMiddleware:
    """接管 ``POST /v1/audio/transcriptions`` 的纯 ASGI middleware。

    其余方法/路径（/v1/chat/completions、/health 等）原样透传，零干预。
    """

    def __init__(self, app, ext: ExtensionState):
        self.app = app
        self.ext = ext

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("method") != "POST" or scope.get("path") != "/v1/audio/transcriptions":
            await self.app(scope, receive, send)
            return
        if not _STARLETTE_AVAILABLE:
            # multipart 解析依赖 starlette（随 vLLM 依赖存在）：缺失时透传原生
            logger.error(
                "starlette.requests 不可用，无法解析 multipart 表单，"
                "/v1/audio/transcriptions 透传给 vLLM 原生处理"
            )
            await self.app(scope, receive, send)
            return

        # processTime 起点 = 进入 handler（含排队等待）
        start = time.perf_counter()
        try:
            await self._handle_transcription(scope, receive, send, start)
        except asyncio.CancelledError:
            # 客户端断连（monitor 触发取消）或服务关闭：不写响应
            logger.warning("客户端已断开或任务被取消，转写请求处理中止")
            raise
        except Exception as exc:
            logger.exception("转写处理异常")
            with contextlib.suppress(Exception):
                preset = error_response(500, f"模型推理或服务内部异常: {exc}", "server_error")
                await _send_json(send, preset["status"], preset["body"])

    # -- 请求处理 -----------------------------------------------------------

    async def _handle_transcription(self, scope, receive, send, start: float) -> None:
        request = _StarletteRequest(scope, receive)
        try:
            form = await request.form()
        except Exception as exc:
            preset = error_response(400, f"multipart/form-data 解析失败: {exc}", "invalid_request_error")
            await _send_json(send, preset["status"], preset["body"])
            return

        # 断连监听：在 form 解析完成后启动，避免与 body 读取并发争用 receive；
        # cancel_event 供 to_thread 线程体（ASR 分块循环）在块间检查尽快中止
        cancel_event = threading.Event()
        main_task = asyncio.current_task()
        monitor = asyncio.create_task(self._watch_disconnect(receive, main_task, cancel_event))
        try:
            await self._process_form(scope, send, form, start, cancel_event)
        finally:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await monitor

    @staticmethod
    async def _watch_disconnect(receive, main_task, cancel_event: threading.Event) -> None:
        """监听客户端断连（http.disconnect）：一旦断开立即取消主任务并置位事件。

        scheduler.slot 的取消语义保证排队位/调度许可清理；执行中断连则尽力
        （slot 的 finally 与 gather 的异常安全已保证许可释放；cancel_event
        让线程内的 ASR 分块循环在块间尽快中止，正在执行的单次前向不可中断）。
        """
        try:
            while True:
                message = await receive()
                if message.get("type") == "http.disconnect":
                    cancel_event.set()
                    if main_task is not None and not main_task.done():
                        main_task.cancel()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            # 个别服务器在 body 消费完后 receive() 行为不一：忽略，仅丧失断连检测
            return

    async def _send_error(self, send, status: int, message: str, err_type: str) -> None:
        preset = error_response(status, message, err_type)
        await _send_json(send, preset["status"], preset["body"])

    def _resolve_engine_client(self, scope) -> Optional[Any]:
        """解析 vLLM engine client（三层防御的后两层在此落地）。

        优先 serve 钩子 init_app_state 包装注入的 ``ext.engine_client``；
        缺省回退 ``scope["app"].state.engine_client``（vLLM init_app_state 写入，
        请求期必然已就绪）并缓存回 ext。两层均不可得时返回 None。
        """
        if self.ext.engine_client is not None:
            return self.ext.engine_client
        app = scope.get("app") if isinstance(scope, dict) else None
        engine_client = getattr(getattr(app, "state", None), "engine_client", None)
        if engine_client is not None:
            self.ext.engine_client = engine_client
        return engine_client

    async def _process_form(self, scope, send, form, start: float, cancel_event: threading.Event) -> None:
        ext = self.ext
        file = form.get("file")
        audio_url = form.get("audio_url")

        # ---- 参数校验（统一 400 / OpenAI 错误体）----------------------------
        try:
            if (file is None) == (audio_url is None):
                raise ValueError("file 与 audio_url 必须提供且只能提供其中之一")
            model_name = form.get("model")
            if not model_name:
                raise ValueError("缺少必填参数 model")
            if ext.served_model_names and model_name not in ext.served_model_names:
                raise ValueError(
                    f"model {model_name!r} 不在本服务已加载模型列表内: {ext.served_model_names}"
                )
            language_name = None
            language_raw = form.get("language")
            if language_raw not in (None, ""):
                language_name = resolve_language_name(str(language_raw))
            min_speakers = _parse_optional_int(form.get("min_speakers"))
            max_speakers = _parse_optional_int(form.get("max_speakers"))
            if None not in (min_speakers, max_speakers) and min_speakers > max_speakers:
                raise ValueError(
                    f"min_speakers({min_speakers}) 不能大于 max_speakers({max_speakers})"
                )
            granularities = list(form.getlist("timestamp_granularities[]")) + list(
                form.getlist("timestamp_granularities")
            )
            response_format = str(form.get("response_format") or "json").strip().lower()
            validate_request_params(granularities, response_format)
        except ValueError as exc:
            await self._send_error(send, 400, str(exc), "invalid_request_error")
            return

        # ---- 音频获取 --------------------------------------------------------
        if file is not None:
            data = await file.read()
        else:
            try:
                safe_url = check_audio_url(str(audio_url))
            except ValueError as exc:
                await self._send_error(send, 400, str(exc), "invalid_request_error")
                return
            try:
                data = await asyncio.to_thread(download_audio, safe_url, ext.max_audio_bytes)
            except ValueError as exc:
                await self._send_error(send, 400, str(exc), "invalid_request_error")
                return
            except Exception as exc:
                await self._send_error(send, 400, f"audio_url 下载失败: {exc}", "invalid_request_error")
                return
        if not data:
            await self._send_error(send, 400, "音频内容为空", "invalid_request_error")
            return

        # ---- 解码与时长校验 ---------------------------------------------------
        try:
            wav = await asyncio.to_thread(decode_audio_bytes, data)
        except Exception as exc:
            await self._send_error(send, 415, f"音频解码失败: {exc}", "invalid_request_error")
            return
        duration = len(wav) / float(SAMPLE_RATE)
        if duration > float(ext.max_audio_seconds):
            await self._send_error(
                send,
                400,
                f"音频时长 {duration:.1f}s 超过上限 {ext.max_audio_seconds}s（--max-audio-seconds 可调）",
                "invalid_request_error",
            )
            return

        context = form.get("prompt") or ""
        segment_mode = "segment" in [str(g).strip() for g in granularities]
        if segment_mode:
            await self._run_segment(
                send, wav, context, language_name,
                min_speakers, max_speakers, duration, start, cancel_event, scope,
            )
        else:
            await self._run_standard(send, wav, context, language_name, response_format, scope)

    # -- 标准模式（OpenAI 兼容）----------------------------------------------

    async def _run_standard(self, send, wav, context, force_language, response_format, scope) -> None:
        """无 segment 粒度：1200s 分块 ASR（信号量限并发）→ OpenAI 标准响应。"""
        ext = self.ext
        engine_client = self._resolve_engine_client(scope)
        if engine_client is None:
            await self._send_error(send, 500, "vLLM engine client 未初始化", "server_error")
            return
        chunks = await asyncio.to_thread(
            split_audio_into_chunks, wav, SAMPLE_RATE, MAX_ASR_INPUT_SECONDS
        )
        semaphore = asyncio.Semaphore(max(1, int(ext.align_batch_size)))

        async def _one(cwav) -> str:
            async with semaphore:
                prompt = build_asr_prompt(ext.processor, context, force_language)
                return await engine_generate(ext, prompt, cwav)

        raw_outputs = await asyncio.gather(*[_one(cwav) for cwav, _ in chunks])
        langs: List[str] = []
        texts: List[str] = []
        for raw in raw_outputs:
            lang, txt = parse_asr_output(raw, user_language=force_language)
            langs.append(lang)
            texts.append(txt)
        text = "".join(texts)
        language = merge_languages(langs)
        duration = len(wav) / float(SAMPLE_RATE)

        if response_format == "text":
            await _send_text(send, 200, text)
            return
        payload: Dict[str, Any] = {"text": text}
        if response_format == "verbose_json":
            payload["duration"] = round(duration, 3)
            payload["language"] = _merged_language_code(language)
        await _send_json(send, 200, payload)

    # -- segment 模式（扩展管线）---------------------------------------------

    async def _run_segment(
        self, send, wav, context, force_language,
        min_speakers, max_speakers, duration, start, cancel_event, scope,
    ) -> None:
        """segment 模式：调度准入 → ASR+对齐 与 diarization 阶段并行 → 组装响应。"""
        ext = self.ext
        if ext.aligner is None or ext.diarizer is None:
            await self._send_error(
                send,
                503,
                "segment 时间戳模式需要对齐与说话人识别扩展：请以 --forced-aligner 与 "
                "--diarizer 启动参数启用（显式传空串为禁用）",
                "service_unavailable",
            )
            return
        engine_client = self._resolve_engine_client(scope)
        if engine_client is None:
            await self._send_error(send, 500, "vLLM engine client 未初始化", "server_error")
            return

        need_mb = estimate_task_need_mb(int(ext.align_batch_size), duration, *budget_devices(ext))
        loop = asyncio.get_running_loop()
        # 调度许可：排队等待期间被取消时由 slot 的取消语义清理排队位；
        # 任务任意阶段异常由 slot 的 try/finally 保证许可必然释放
        async with ext.scheduler.slot(need_mb):
            # 任务内阶段并行：diarization 不依赖 ASR/对齐结果，两线程同时执行；
            # return_exceptions=True 等待双方落定后再抛首个异常，避免孤儿线程
            try:
                results = await asyncio.gather(
                    asyncio.to_thread(
                        _run_asr_align, ext, wav, context, force_language, loop, cancel_event
                    ),
                    asyncio.to_thread(_run_diarize, ext, wav, min_speakers, max_speakers),
                    return_exceptions=True,
                )
            finally:
                # 归还缓存分配器保留的空闲显存块：必须在 slot 释放前执行
                # （slot 退出会立即触发队首重准入，晚于此则下一个排队任务
                # 仍按偏低的空闲显存误判而继续滞留队列）
                await asyncio.to_thread(_release_gpu_cache)
        for result in results:
            if isinstance(result, BaseException):
                raise result

        (full_text, language, merged_align), diar_results = results
        align_items = list(merged_align.items) if merged_align is not None else []
        diar_segments = diar_results[0].segments if diar_results else []
        process_time = time.perf_counter() - start  # 含排队等待
        response = build_segment_response(
            align_items,
            diar_segments,
            full_text,
            language,
            duration,
            process_time,
            segment_gap_threshold=float(ext.segment_gap_threshold),
            max_segment_seconds=float(ext.max_segment_seconds),
        )
        await _send_json(send, 200, response)
