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
Qwen3-ASR segment 级时间戳 + 说话人识别 API 调用示例（OpenAI 兼容扩展端点）。

对应 spec「Segment 级时间戳与说话人识别 API 扩展」：请求遵循 OpenAI
``POST /v1/audio/transcriptions`` 规范（multipart form），当
``timestamp_granularities[]`` 含 ``"segment"`` 时启用扩展管线
（vLLM ASR + 强制对齐 + pyannote 说话人归属），返回 segment 级时间戳 +
说话人标签 + speakerSummary 的结构化 JSON。

用法::

    # 离线自测（不依赖网络 / GPU / 服务端，仅标准库）
    python examples/example_segment_api.py --self-test

    # 上传本地音频文件
    python examples/example_segment_api.py --file meeting.wav --language zh

    # 通过 HTTPS URL 提供音频（大文件推荐）
    python examples/example_segment_api.py --audio-url https://example.com/meeting.wav

部署要点
--------

1. 反向代理（nginx，长音频必需）::

       client_max_body_size 500m;   # 1h wav 约 230MB 上传
       proxy_read_timeout 900s;     # 长音频端到端耗时（含排队等待）
       proxy_send_timeout 900s;

2. HF 门控模型：pyannote 管线（pyannote/speaker-diarization-community-1）为
   门控模型，需设置 ``HF_TOKEN`` 或 ``PYANNOTE_API_TOKEN`` 环境变量（先在
   模型页接受 user conditions，再到 hf.co/settings/tokens 创建 token）；
   Docker 部署建议挂载 ``HF_HOME`` 卷缓存模型，避免每次启动重新下载。

3. 显存参考（单卡 gpu_memory_utilization）：

   - A10 24GB：用默认即可（服务自动注入 0.70，双并发 1h 音频可行）；
     仅单并发时可显式提到 0.75；
   - T4 16GB：双并发 0.55 / 单并发 0.60；
   - P4 8GB：不推荐承载长音频；仅短音频（≤10min）可用 0.35 +
     ``--gpu-reserve-mb 512 --max-concurrent-tasks 1``。

4. 多卡拓扑示例：

   - 两卡：``qwen-asr-serve <model> --aligner-device cuda:1 --diarizer-device cuda:1``
     （vLLM 独占 GPU0，可用满 0.9）；
   - 三卡：``--aligner-device cuda:1 --diarizer-device cuda:2``（各自独占，显存互不竞争）。

5. 依赖：``pip install qwen-asr[vllm,diarization]``；系统需安装 ffmpeg
   （torchcodec 依赖）。

6. 与 OpenAI 标准的差异：OpenAI 规范中 ``segment`` 粒度配 ``verbose_json``
   返回无说话人的标准 segments；本服务将 ``segment`` 粒度定义为
   "segment + 说话人扩展"语义（即本示例展示的响应结构），属用户确认的产品决策。
"""

import argparse
import importlib.util
import json
import mimetypes
import os
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "qwen3-asr"
DEFAULT_TIMEOUT = 900


# ---------------------------------------------------------------------------
# multipart/form-data 请求体构造（纯标准库手工编码，不引入第三方依赖）
# ---------------------------------------------------------------------------


def build_request_body(args, boundary: Optional[str] = None) -> Tuple[bytes, str]:
    """构造 ``POST /v1/audio/transcriptions`` 的 multipart/form-data 请求体。

    两种模式（``args.standard`` 切换）：
    - segment 模式（默认）：``timestamp_granularities[]=segment`` +
      ``response_format=json`` 固定携带（segment 模式核心开关，与
      text/verbose_json 不兼容）；
    - 标准 OpenAI 模式（``--standard``）：不带粒度参数，``response_format``
      取 ``args.response_format``（json / text / verbose_json），服务端返回
      OpenAI 标准响应（``{"text"}`` 等，无 speaker 字段）。

    :param args: 含 ``model / file / audio_url / language / prompt /
        min_speakers / max_speakers / standard / response_format`` 属性的对象
        （argparse.Namespace 或 SimpleNamespace 均可；standard/response_format
        缺省时分别按 False / json 处理）
    :param boundary: multipart 分隔符（缺省自造随机值）
    :return: ``(请求体字节, Content-Type 头值)``
    """
    if not boundary:
        boundary = "----qwen3asr" + uuid.uuid4().hex
    standard_mode = bool(getattr(args, "standard", False))
    response_format = str(getattr(args, "response_format", None) or "json")
    parts: List[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    def add_file(name: str, filename: str, content: bytes) -> None:
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        parts.append(content)
        parts.append(b"\r\n")

    add_field("model", args.model)
    if args.audio_url:
        add_field("audio_url", args.audio_url)
    if args.language:
        add_field("language", args.language)
    if args.prompt:
        add_field("prompt", args.prompt)
    if not standard_mode:
        # 说话人数约束仅 segment 模式有意义（透传 pyannote）
        if getattr(args, "min_speakers", None) is not None:
            add_field("min_speakers", str(args.min_speakers))
        if getattr(args, "max_speakers", None) is not None:
            add_field("max_speakers", str(args.max_speakers))
        add_field("timestamp_granularities[]", "segment")
    add_field("response_format", response_format)
    if args.file:
        file_path = Path(args.file)
        add_file("file", file_path.name, file_path.read_bytes())

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# ---------------------------------------------------------------------------
# 响应结构断言与摘要输出
# ---------------------------------------------------------------------------


def validate_segment_response(resp: dict) -> List[str]:
    """对 segment 模式响应做基本结构断言，返回错误列表（空列表 = 全部通过）。

    检查项：segments 按 start 升序且每段字段齐全、
    speakerSummary.speakerCount == len(speakers)、processTime 存在。
    """
    errors: List[str] = []
    segments = resp.get("segments")
    if not isinstance(segments, list):
        errors.append("segments 缺失或非列表")
    else:
        starts = [seg.get("start") for seg in segments if isinstance(seg, dict)]
        if len(starts) != len(segments) or starts != sorted(starts):
            errors.append("segments 未按 start 升序（或存在非对象项）")
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            missing = [k for k in ("start", "end", "text", "speaker", "speakers") if k not in seg]
            if missing:
                errors.append(f"segments[{i}] 缺少字段: {missing}")
    summary = resp.get("speakerSummary")
    if not isinstance(summary, dict):
        errors.append("speakerSummary 缺失或非对象")
    else:
        speakers = summary.get("speakers")
        if not isinstance(speakers, list):
            errors.append("speakerSummary.speakers 缺失或非列表")
        elif summary.get("speakerCount") != len(speakers):
            errors.append(
                f"speakerCount({summary.get('speakerCount')}) != len(speakers)({len(speakers)})"
            )
    if "processTime" not in resp:
        errors.append("processTime 缺失")
    return errors


def print_speaker_summary(resp: dict) -> None:
    """输出 speakerSummary 摘要行（speakers 已按服务端 totalDuration 降序返回）。"""
    summary = resp.get("speakerSummary") or {}
    print(
        f"说话人摘要: 共 {summary.get('speakerCount')} 人 | "
        f"音频时长 {resp.get('duration')}s | 处理耗时 {resp.get('processTime')}s"
    )
    for sp in summary.get("speakers") or []:
        print(f"  - {sp.get('id')}: 发言 {sp.get('totalDuration')}s / {sp.get('segmentCount')} 段")


# ---------------------------------------------------------------------------
# 服务调用（urllib.request）
# ---------------------------------------------------------------------------


def transcribe(args: argparse.Namespace) -> Tuple[object, str]:
    """调用转写端点并返回 ``(响应, 内容类型标记)``（HTTPError 由调用方处理）。

    - ``response_format=json``（两种模式默认）：返回解析后的 dict；
    - ``response_format=text``（仅标准模式）：返回纯文本 str；
    - ``response_format=verbose_json``（仅标准模式）：返回 dict。
    """
    body, content_type = build_request_body(args)
    url = args.base_url.rstrip("/") + "/v1/audio/transcriptions"
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as resp:
        raw = resp.read().decode("utf-8")
    if str(getattr(args, "response_format", "json") or "json") == "text":
        return raw, "text"
    return json.loads(raw), "json"


# ---------------------------------------------------------------------------
# 离线自测（--self-test，不依赖网络 / GPU / 服务端）
# ---------------------------------------------------------------------------


def _load_pipeline_by_path():
    """按文件路径加载 pipeline 模块。

    pipeline.py 仅依赖标准库，按路径加载可规避完整包环境缺失
    numpy / torch / vllm 时 ``import qwen_asr`` 失败的问题。
    """
    pipeline_path = Path(__file__).resolve().parent.parent / "qwen_asr" / "service" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("qwen_asr_service_pipeline_standalone", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_self_test() -> None:
    """离线自测：pipeline.self_test 真实执行 + 语言映射完整性 + 本脚本构造/断言逻辑。"""
    # 优先正常 import（包环境完整时）；失败回退按文件路径加载（仅标准库依赖）
    try:
        from qwen_asr.service.pipeline import LANGUAGE_NAME_TO_CODE
        from qwen_asr.service.pipeline import self_test as pipeline_self_test
    except Exception:  # noqa: BLE001 —— numpy/torch/vllm 任一缺失即回退文件加载
        module = _load_pipeline_by_path()
        pipeline_self_test = module.self_test
        LANGUAGE_NAME_TO_CODE = module.LANGUAGE_NAME_TO_CODE

    # 1) 管道纯逻辑自测（真实执行，通过时打印 "pipeline self_test ok"）
    pipeline_self_test()

    # 2) 语言映射完整性：30 项，抽查首尾与易错项
    assert len(LANGUAGE_NAME_TO_CODE) == 30, (
        f"语言映射应为 30 项，实际 {len(LANGUAGE_NAME_TO_CODE)}"
    )
    assert LANGUAGE_NAME_TO_CODE["Chinese"] == "zh"
    assert LANGUAGE_NAME_TO_CODE["Cantonese"] == "yue"  # 粤语码是 yue
    assert LANGUAGE_NAME_TO_CODE["Filipino"] == "fil"  # 菲律宾语码是 fil
    assert LANGUAGE_NAME_TO_CODE["Macedonian"] == "mk"

    # 3) 请求体构造：临时文件 + 固定 boundary，验证字段名 / 文件内容 / 收尾分隔符
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.write(fd, b"RIFF-stub-audio-bytes")
        os.close(fd)
        fake_args = SimpleNamespace(
            model="qwen3-asr", file=tmp_path, audio_url=None,
            language="zh", prompt="会议记录", min_speakers=2, max_speakers=4,
        )
        boundary = "testboundary123"
        body, content_type = build_request_body(fake_args, boundary=boundary)
        assert content_type == f"multipart/form-data; boundary={boundary}"
        # 固定开关与全部可选文本字段均在请求体中
        for name, value in [
            ("model", "qwen3-asr"),
            ("language", "zh"),
            ("prompt", "会议记录"),
            ("min_speakers", "2"),
            ("max_speakers", "4"),
            ("timestamp_granularities[]", "segment"),
            ("response_format", "json"),
        ]:
            assert f'name="{name}"'.encode() in body, f"缺少字段名 {name}"
            assert value.encode("utf-8") in body, f"缺少字段值 {name}={value}"
        # file 部分带 filename 与二进制内容；audio_url 未混入；收尾分隔符完整
        assert b'name="file"; filename=' in body
        assert Path(tmp_path).name.encode() in body
        assert b"RIFF-stub-audio-bytes" in body
        assert b'name="audio_url"' not in body
        assert body.endswith(f"--{boundary}--\r\n".encode())

        # audio_url 分支：无文件部分、无可选字段
        url_args = SimpleNamespace(
            model="qwen3-asr", file=None, audio_url="https://example.com/meeting.wav",
            language=None, prompt=None, min_speakers=None, max_speakers=None,
        )
        body2, content_type2 = build_request_body(url_args, boundary=boundary)
        assert content_type2 == f"multipart/form-data; boundary={boundary}"
        assert b'name="audio_url"' in body2
        assert b"https://example.com/meeting.wav" in body2
        assert b'name="file"' not in body2
        assert b'name="language"' not in body2 and b'name="prompt"' not in body2

        # 标准 OpenAI 模式（--standard）：不带粒度参数与说话人约束，
        # response_format 透传（verbose_json 演示）
        std_args = SimpleNamespace(
            model="qwen3-asr", file=tmp_path, audio_url=None,
            language=None, prompt=None, min_speakers=2, max_speakers=None,
            standard=True, response_format="verbose_json",
        )
        body3, _ = build_request_body(std_args, boundary=boundary)
        assert b'name="timestamp_granularities[]"' not in body3, "标准模式不应携带粒度参数"
        assert b'name="min_speakers"' not in body3, "标准模式不应携带说话人约束"
        assert b'name="response_format"' in body3
        assert b"verbose_json" in body3
    finally:
        os.remove(tmp_path)

    # 4) 响应结构断言：合法响应零错误；非法响应（乱序 / 数目不符 / 缺 processTime）全部捕获
    valid = {
        "language": "zh", "duration": 5.0, "text": "你好欢迎光临", "processTime": 1.234,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "你好", "speaker": "SPEAKER_00", "speakers": ["SPEAKER_00"]},
            {"start": 3.0, "end": 5.0, "text": "欢迎光临", "speaker": "SPEAKER_01", "speakers": ["SPEAKER_01"]},
        ],
        "speakerSummary": {
            "speakerCount": 2,
            "speakers": [
                {"id": "SPEAKER_00", "totalDuration": 2.0, "segmentCount": 1},
                {"id": "SPEAKER_01", "totalDuration": 2.0, "segmentCount": 1},
            ],
        },
    }
    assert validate_segment_response(valid) == []

    invalid = {  # segments 降序 + speakerCount 不符 + 缺 processTime
        "language": "zh", "duration": 5.0, "text": "x",
        "segments": [
            {"start": 3.0, "end": 5.0, "text": "b", "speaker": "SPEAKER_01", "speakers": ["SPEAKER_01"]},
            {"start": 0.0, "end": 2.0, "text": "a", "speaker": "SPEAKER_00", "speakers": ["SPEAKER_00"]},
        ],
        "speakerSummary": {
            "speakerCount": 5,
            "speakers": [{"id": "SPEAKER_00", "totalDuration": 2.0, "segmentCount": 1}],
        },
    }
    errors = validate_segment_response(invalid)
    assert any("升序" in e for e in errors), "乱序 segments 未被捕获"
    assert any("speakerCount" in e for e in errors), "speakerCount 不符未被捕获"
    assert any("processTime" in e for e in errors), "processTime 缺失未被捕获"
    assert len(errors) == 3, f"应恰好检出 3 处错误，实际 {len(errors)}: {errors}"

    print("example self_test ok")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR segment 级时间戳 + 说话人识别 API 示例"
                    "（默认 segment 扩展模式；--standard 演示 OpenAI 标准模式）",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="离线自测（不依赖网络 / GPU / 服务端）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"服务地址（默认 {DEFAULT_BASE_URL}）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"模型名，须匹配服务端 --served-model-name（默认 {DEFAULT_MODEL}）")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", help="本地音频文件路径（与 --audio-url 二选一）")
    source.add_argument("--audio-url", help="音频 HTTPS URL（与 --file 二选一，大文件推荐）")
    parser.add_argument("--language", help="语言码（zh/en）或语言名（Chinese），可选")
    parser.add_argument("--prompt", help="上下文提示（映射到 ASR context），可选")
    parser.add_argument("--min-speakers", type=int, help="说话人数下限（透传 pyannote），可选")
    parser.add_argument("--max-speakers", type=int, help="说话人数上限（透传 pyannote），可选")
    parser.add_argument("--standard", action="store_true",
                        help="演示 OpenAI 标准模式：不带 timestamp_granularities，"
                             "返回 {\"text\"} 等标准响应（无 speaker 字段）")
    parser.add_argument("--response-format", choices=("json", "text", "verbose_json"),
                        default="json",
                        help="响应格式（仅 --standard 模式生效；segment 模式固定 json，"
                             "与 text/verbose_json 不兼容）")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"请求超时秒数（默认 {DEFAULT_TIMEOUT}，长音频建议不小于此值）")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0

    if not args.file and not args.audio_url:
        parser.error("--file 与 --audio-url 必须提供其一（用法见文件头 docstring）")
    if not args.standard and args.response_format != "json":
        print("提示: segment 模式固定 response_format=json，--response-format 已忽略",
              file=sys.stderr)

    try:
        resp, resp_kind = transcribe(args)
    except urllib.error.HTTPError as exc:
        # 服务端错误：读取 body 打印错误 JSON（OpenAI 风格 {"error": {...}}）
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} {exc.reason}:")
        try:
            print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print(body)
        return 1
    except urllib.error.URLError as exc:
        print(f"请求失败（服务未启动或网络不通）: {exc}", file=sys.stderr)
        return 1

    if args.standard:
        # OpenAI 标准模式：text 格式为纯文本，其余为 JSON（断言 text 字段存在）
        if resp_kind == "text":
            print(resp)
            return 0
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        if not isinstance(resp, dict) or "text" not in resp:
            print("结构断言失败: 标准模式响应缺少 text 字段", file=sys.stderr)
            return 1
        return 0

    print(json.dumps(resp, ensure_ascii=False, indent=2))
    errors = validate_segment_response(resp)
    if errors:
        for err in errors:
            print(f"结构断言失败: {err}", file=sys.stderr)
        return 1
    print_speaker_summary(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
