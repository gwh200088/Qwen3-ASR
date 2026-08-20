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
qwen-asr-serve：vLLM OpenAI 兼容服务的组装式入口。

流程（spec「服务架构（vLLM 进程内扩展）」）：

1. 注册 Qwen3-ASR 模型（transformers Auto 注册 + vLLM ModelRegistry，保留原有逻辑）；
2. 从 argv 剥离扩展参数（剩余参数原样转发 vLLM CLI，语义不变）；
3. 单卡默认拓扑下按需自动注入 ``--gpu-memory-utilization 0.70``；
4. 安装 vLLM ``build_app`` 钩子（在 ``vllm_main()`` 调用前）：
   vLLM 引擎初始化、app 构建完成后，加载扩展模型（processor CPU 常驻 +
   aligner/diarizer 按设备）、执行显存启动校验（失败即终止启动）、挂载
   ``TranscriptionsMiddleware``、注册 ``GET /health/detail``；
5. 以 ``<prog> serve <剩余参数>`` 转发 vLLM CLI 启动。

``--diarizer ""`` / ``--forced-aligner ""`` 显式禁用时：不加载扩展、不注入
gpu_memory_utilization、不安装钩子，行为与纯 vLLM 包装完全一致。
"""

import argparse
import functools
import logging
import sys
from typing import Any, List, Optional, Tuple

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_asr.service import load_extensions
from qwen_asr.service.extensions import should_inject_gmu
from qwen_asr.service.middleware import (
    TranscriptionsMiddleware,
    extract_served_model_names,
    gpu_memory_utilization_specified,
    scan_flag_values,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except Exception as e:
    raise ImportError(
        "vLLM is not available, to use qwen-asr-serve, please install with: pip install qwen-asr[vllm]"
    ) from e

from vllm.entrypoints.cli.main import main as vllm_main

logger = logging.getLogger(__name__)

#: 单卡默认拓扑自动注入的 vLLM 显存预分配比例（spec 显存预算方案，A10 双并发自洽）
AUTO_GPU_MEMORY_UTILIZATION = "0.70"

#: 钩子幂等标记（同一进程只安装一次）
_build_app_hook_installed = False


# ---------------------------------------------------------------------------
# 扩展参数定义与剥离（纯函数，可离线单测）
# ---------------------------------------------------------------------------


def build_extension_parser() -> argparse.ArgumentParser:
    """扩展参数定义（默认值逐项对应 spec MODIFIED Requirement）。

    注意 ``allow_abbrev=False``：防止 vLLM 的 ``--max-num-seqs`` 等未知参数被
    argparse 前缀缩写误匹配到本解析器的 ``--max-*`` 扩展参数导致解析歧义报错。
    """
    parser = argparse.ArgumentParser(
        prog="qwen-asr-serve",
        add_help=False,
        allow_abbrev=False,
        description="qwen-asr-serve 扩展参数（剥离后剩余参数原样转发 vLLM CLI）",
    )
    parser.add_argument(
        "--forced-aligner",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="强制对齐模型名/本地路径；显式传空串（--forced-aligner \"\"）禁用",
    )
    parser.add_argument(
        "--diarizer",
        default="pyannote/speaker-diarization-community-1",
        help="说话人识别管线名/本地路径（可配 legacy pyannote/speaker-diarization-3.1）；显式空串禁用",
    )
    parser.add_argument(
        "--pyannote-token",
        default=None,
        help="HuggingFace 访问令牌（pyannote 门控模型必填）；缺省依次取环境变量 "
        "PYANNOTE_API_TOKEN / HF_TOKEN",
    )
    parser.add_argument("--aligner-device", default="cuda:0", help="对齐模型设备（默认跟随 vLLM 主设备）")
    parser.add_argument("--diarizer-device", default="cuda:0", help="说话人识别设备（默认跟随 vLLM 主设备）")
    parser.add_argument("--max-concurrent-tasks", type=int, default=2, help="segment 任务最大并发数")
    parser.add_argument("--gpu-reserve-mb", type=int, default=1024, help="每设备显存安全余量（MB）")
    parser.add_argument("--max-audio-seconds", type=float, default=3600.0, help="音频时长上限（秒）")
    parser.add_argument("--max-audio-bytes", type=int, default=500 * 1024 * 1024, help="音频体积上限（字节，默认 500MB）")
    parser.add_argument("--segment-gap-threshold", type=float, default=0.8, help="segment 切分时间间隙阈值（秒）")
    parser.add_argument("--max-segment-seconds", type=float, default=30.0, help="segment 最大段长（秒）")
    parser.add_argument("--align-batch-size", type=int, default=4, help="对齐批大小（亦为标准模式 ASR 并发上限）")
    return parser


def split_extension_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """从 argv 剥离扩展参数；剩余参数（含未知参数）按原顺序原样保留。

    Returns:
        (ext_args, rest)：ext_args 为扩展参数 Namespace（未传项取默认值），
        rest 为剥离后剩余 argv（vLLM CLI 语义不变）。
    """
    parser = build_extension_parser()
    ext_args, rest = parser.parse_known_args(argv)
    return ext_args, rest


def extract_model_path(rest: List[str]) -> str:
    """从剩余 argv 提取 ASR 模型路径（供 processor 加载）。

    优先首个非 flag 位置参数（``qwen-asr-serve <model> --flags`` 常规形态），
    其次 ``--model`` / ``--model-tag`` 取值；均无则返回空串。
    限制：位于 flag 之后的位置参数（如 ``--port 8000 /model``）无法识别，
    请使用 ``--model`` 显式指定。
    """
    for token in rest:
        if token.startswith("-"):
            break  # 位置参数只可能在首个 flag 之前
        return token
    for flag in ("--model", "--model-tag"):
        values = scan_flag_values(rest, flag)
        if values:
            return values[-1]
    return ""


def maybe_inject_gpu_memory_utilization(rest: List[str], ext_args: argparse.Namespace) -> List[str]:
    """单卡默认拓扑下自动注入 ``--gpu-memory-utilization``（spec 自动调整规则）。

    仅当任一扩展启用、其设备与 vLLM 主设备（cuda:0）相同、且用户未显式指定
    该参数时注入 0.70 并打日志；扩展禁用或独立设备时不注入（vLLM 可用满默认值）。
    """
    aligner_enabled = bool(str(ext_args.forced_aligner or "").strip())
    diarizer_enabled = bool(str(ext_args.diarizer or "").strip())
    if not aligner_enabled and not diarizer_enabled:
        return rest  # 扩展全部禁用：行为与现状一致，不注入
    user_specified = gpu_memory_utilization_specified(rest)
    aligner_dev = str(ext_args.aligner_device or "cuda:0") if aligner_enabled else "cpu"
    diarizer_dev = str(ext_args.diarizer_device or "cuda:0") if diarizer_enabled else "cpu"
    if not should_inject_gmu(aligner_dev, diarizer_dev, user_specified):
        return rest
    logger.warning(
        "检测到对齐/说话人扩展与 vLLM 共用主设备 cuda:0，且未显式指定 "
        "--gpu-memory-utilization：自动注入 %s 为扩展模型预留显存（依据 spec 显存"
        "预算方案：A10 24GB 下 vLLM 占 ~16.8GB，余量覆盖扩展常驻 ~1.9GB + 安全余量 "
        "1GB + 默认双并发任务瞬态）；如需自定义请显式传 --gpu-memory-utilization。",
        AUTO_GPU_MEMORY_UTILIZATION,
    )
    return rest + ["--gpu-memory-utilization", AUTO_GPU_MEMORY_UTILIZATION]


# ---------------------------------------------------------------------------
# vLLM app 钩子（0.14.0 兼容层）
#
# vLLM 0.14.0 run_server_worker 的执行顺序（本兼容层依据的时序）：
#   async with build_async_engine_client(args, ...) as engine_client:
#       app = build_app(args)                                # ← 只接收 args
#       await init_app_state(engine_client, app.state, args) # ← engine_client 在
#                                                            #    此刻才注入 state
#       await serve_http(app, ...)                           # ← 此刻才开始 serve
# 因此 engine_client 的提取挂点是 init_app_state（其第一个参数即 engine_client），
# 而非 build_app；middleware 另有 scope["app"].state.engine_client 的请求期
# 懒解析兜底（三层防御，适配 vLLM 小版本签名差异）。
#
# 注意：vLLM 0.14.0 的 build_app 末尾调用 model_hosting_container_standards 的
# sagemaker.bootstrap(app)，其 load_middlewares 会立即执行
# ``app.middleware_stack = app.build_middleware_stack()``（重排中间件后主动建栈）。
# 此后 Starlette 的 add_middleware 见 middleware_stack 已非 None 即抛
# "Cannot add middleware after an application has started"——但此刻 serve 尚未
# 开始，属误判；挂载需走 _mount_transcriptions_middleware 的兼容路径。
# ---------------------------------------------------------------------------

#: build_app 包装期间构造的 ExtensionState 暂存（init_app_state 包装注入 engine_client 用）
_pending_exts: List[Any] = []


def _register_health_detail(app: Any, ext: Any) -> None:
    """注册 ``GET /health/detail``：调度状态 + extensionModelsLoaded。"""

    async def health_detail():
        stats = ext.scheduler.stats() if ext.scheduler is not None else {}
        return {"status": "ok", "extensionModelsLoaded": ext.extensions_enabled, **stats}

    if hasattr(app, "add_api_route"):  # FastAPI（vLLM build_app 返回值）
        app.add_api_route("/health/detail", health_detail, methods=["GET"], include_in_schema=False)
    elif hasattr(app, "add_route"):  # Starlette 兜底
        from starlette.responses import JSONResponse

        async def _starlette_endpoint(request):
            stats = ext.scheduler.stats() if ext.scheduler is not None else {}
            return JSONResponse({"status": "ok", "extensionModelsLoaded": ext.extensions_enabled, **stats})

        app.add_route("/health/detail", _starlette_endpoint, methods=["GET"])
    else:
        logger.warning("当前 app 不支持路由注册，/health/detail 未注册")


def _mount_transcriptions_middleware(app: Any, ext: Any) -> None:
    """挂载 TranscriptionsMiddleware（兼容 vLLM 0.14.0 的 SageMaker bootstrap）。

    vLLM 0.14.0 的 ``build_app`` 末尾调用
    ``model_hosting_container_standards.sagemaker.bootstrap(app)``，其
    ``load_middlewares`` 会主动执行
    ``app.middleware_stack = app.build_middleware_stack()``（重排为
    throttle → 引擎中间件 → pre_post_process 后立即建栈）。此后 Starlette 的
    ``add_middleware`` 见 middleware_stack 已非 None 即抛
    ``"Cannot add middleware after an application has started"``——但此刻
    serve 尚未开始（serve_http 在 build_app 返回之后才调用），属误判。

    兼容策略：优先走标准 ``add_middleware``；抛 RuntimeError 时改为手动等价
    操作——在 user_middleware 头部插入（与 add_middleware 的 ``insert(0, ...)``
    语义一致，位于最外层）并重建 middleware_stack（与 bootstrap 自身重建手法
    相同，此时未开始 serve，安全）。
    """
    try:
        app.add_middleware(TranscriptionsMiddleware, ext=ext)
        return
    except RuntimeError as exc:
        logger.info(f"add_middleware 被拒（{exc}），改用手动插入 + 重建中间件栈兼容路径")
    from starlette.middleware import Middleware

    app.user_middleware.insert(0, Middleware(TranscriptionsMiddleware, ext=ext))
    app.middleware_stack = app.build_middleware_stack()


def _install_init_app_state_hook(api_server: Any) -> None:
    """包装 ``init_app_state``：调用原函数后把 engine_client 注入暂存的 ext。

    vLLM 0.14.0 签名 ``init_app_state(engine_client, state, args)``：engine_client
    为第一个位置参数（或同名 kwarg）。找不到该符号时静默跳过——middleware 的
    请求期懒解析（scope["app"].state.engine_client）仍可兜底。
    """
    original_init_app_state = getattr(api_server, "init_app_state", None)
    if not callable(original_init_app_state) or getattr(
        original_init_app_state, "_qwen_asr_wrapped", False
    ):
        return

    @functools.wraps(original_init_app_state)
    async def wrapped_init_app_state(*call_args, **call_kwargs):
        result = await original_init_app_state(*call_args, **call_kwargs)
        engine_client = call_kwargs.get("engine_client")
        if engine_client is None and call_args:
            engine_client = call_args[0]
        if engine_client is not None and _pending_exts:
            for ext in _pending_exts:
                ext.engine_client = engine_client
            # 注入完成即清空：单进程单次启动场景下释放暂存引用
            _pending_exts.clear()
            logger.info("已从 vLLM init_app_state 捕获 engine_client 并注入扩展状态")
        return result

    wrapped_init_app_state._qwen_asr_wrapped = True  # type: ignore[attr-defined]
    api_server.init_app_state = wrapped_init_app_state


def install_build_app_hook(
    ext_args: argparse.Namespace,
    model_path: str,
    served_model_names: List[str],
) -> None:
    """把 ``vllm.entrypoints.openai.api_server.build_app`` 替换为包装函数。

    包装逻辑：原样调用原函数取回 app → ``load_extensions``（此时 vLLM 引擎已
    初始化、显存已占；扩展加载后执行启动校验，失败即抛 RuntimeError 终止启动）
    → ``_mount_transcriptions_middleware``（标准 add_middleware 优先，被
    SageMaker bootstrap 预建栈拒绝时手动插入并重建栈）→ 注册
    ``/health/detail`` → 返回 app。engine_client 不在此提取（时序上尚不可得），
    由 init_app_state 钩子或 middleware 请求期懒解析注入。

    Raises:
        RuntimeError: vLLM 未暴露 build_app（版本不兼容，中文排查指引）。
    """
    global _build_app_hook_installed
    import vllm.entrypoints.openai.api_server as api_server

    original_build_app = getattr(api_server, "build_app", None)
    if not callable(original_build_app):
        raise RuntimeError(
            "当前 vLLM 版本未暴露 vllm.entrypoints.openai.api_server.build_app，"
            "qwen-asr-serve 扩展无法挂载（兼容层按 vLLM 0.14.0 开发）。排查指引："
            "1) pip show vllm 确认版本，建议固定 vllm==0.14.0；"
            "2) 执行 python -c \"import vllm.entrypoints.openai.api_server as m; "
            "print(hasattr(m, 'build_app'))\" 验证符号存在；"
            "3) 若 vLLM 小版本调整了 app 构建入口，请尝试重装 vllm==0.14.0 或"
            "向 qwen-asr 仓库反馈 issue。"
        )
    if _build_app_hook_installed:
        return

    @functools.wraps(original_build_app)
    def wrapped_build_app(*call_args, **call_kwargs):
        app = original_build_app(*call_args, **call_kwargs)
        logger.info("vLLM app 构建完成，开始加载 qwen-asr-serve 服务扩展...")
        try:
            ext = load_extensions(ext_args, model_path, served_model_names)
        except Exception as exc:
            raise RuntimeError(
                f"qwen-asr-serve 扩展模型加载/显存启动校验失败，服务终止启动: {exc}"
            ) from exc
        _pending_exts.append(ext)
        _mount_transcriptions_middleware(app, ext)
        _register_health_detail(app, ext)
        logger.info(
            "qwen-asr-serve 扩展就绪：TranscriptionsMiddleware 已挂载，/health/detail 已注册"
        )
        return app

    api_server.build_app = wrapped_build_app
    _install_init_app_state_hook(api_server)
    _build_app_hook_installed = True
    logger.info("已安装 vLLM build_app / init_app_state 钩子（扩展将在 app 构建后加载）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _reject_multi_api_server(rest: List[str]) -> None:
    """扩展启用时拒绝 ``--api-server-count > 1``（快速失败，防静默退化）。

    多 API server 进程会各自重新 import vllm api_server 模块，本进程的
    build_app/init_app_state monkey-patch 在子进程中不生效——扩展会静默
    丢失且无任何告警，服务退化为纯 vLLM。与其带病运行，不如启动即报错。
    """
    values = scan_flag_values(rest, "--api-server-count")
    for value in values:
        try:
            count = int(str(value).strip())
        except ValueError:
            continue
        if count > 1:
            raise RuntimeError(
                f"qwen-asr-serve 扩展（segment 时间戳/说话人识别）不支持 "
                f"--api-server-count > 1（当前 {count}）：多 API server 进程无法"
                "继承本进程的 app 钩子，扩展将静默失效。请移除该参数（默认 1），"
                "或通过负载均衡部署多个单 API server 实例。"
            )


def main(argv: Optional[List[str]] = None) -> None:
    """组装式入口：剥离扩展参数 → 注入 gmu → 安装钩子 → 转发 ``vllm serve``。"""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    ext_args, rest = split_extension_args(raw_argv)
    served_model_names = extract_served_model_names(rest)
    model_path = extract_model_path(rest)
    rest = maybe_inject_gpu_memory_utilization(rest, ext_args)

    extensions_enabled = bool(str(ext_args.forced_aligner or "").strip()) or bool(
        str(ext_args.diarizer or "").strip()
    )
    if extensions_enabled:
        _reject_multi_api_server(rest)
        # 钩子必须在 vllm_main() 调用前安装
        install_build_app_hook(ext_args, model_path, served_model_names)
    else:
        logger.info(
            "扩展模型全部禁用（--forced-aligner/--diarizer 显式空串），"
            "qwen-asr-serve 以纯 vLLM 模式启动（行为与现状一致）"
        )

    sys.argv = [sys.argv[0] if sys.argv else "qwen-asr-serve", "serve", *rest]
    vllm_main()


if __name__ == "__main__":
    main()
