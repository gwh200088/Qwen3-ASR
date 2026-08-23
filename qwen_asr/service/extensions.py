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
服务扩展状态与加载器（qwen-asr-serve 进程内扩展）。

职责（对应 spec「服务架构（vLLM 进程内扩展）」与「显存预算方案」）：

- `ExtensionState`：扩展运行态——aligner / diarizer / processor（CPU 常驻）/
  GpuScheduler / 进程级对齐锁 / 全部服务配置项 / engine_client 注入位；
- `load_extensions()`：启动时加载 processor 与两个扩展模型（未显式禁用时），
  设备搬移后构建调度器并执行**按设备最小瞬态需求**的启动快速失败校验；
- `should_inject_gmu()`：单卡默认拓扑下自动注入 `--gpu-memory-utilization 0.70`
  的判定（纯函数，serve.py 复用）。

重依赖（torch / transformers / pyannote）全部延迟 import：本模块顶层仅依赖
标准库与 scheduler，无 GPU 环境下可正常导入与单测。
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .scheduler import (
    GpuScheduler,
    estimate_task_need_mb,
    normalize_device,
)

__all__ = [
    "VLLM_PRIMARY_DEVICE",
    "ExtensionState",
    "load_extensions",
    "should_inject_gmu",
    "budget_devices",
]

logger = logging.getLogger(__name__)

#: vLLM 主设备约定：同进程、同一 CUDA_VISIBLE_DEVICES 下的 cuda:0（spec「多 GPU 部署」）
VLLM_PRIMARY_DEVICE = "cuda:0"


@dataclass
class ExtensionState:
    """服务扩展运行态（由 `load_extensions` 构造，serve 钩子注入 engine_client）。

    Attributes:
        aligner: Qwen3ForcedAligner 实例；显式禁用（--forced-aligner ""）时为 None。
        diarizer: SpeakerDiarizer 实例；显式禁用（--diarizer ""）时为 None。
        processor: Qwen3ASRProcessor（仅 tokenizer/特征提取器，CPU 常驻不占显存），
            供 middleware 构造 chat template prompt。
        scheduler: GPU 显存感知调度器（segment 模式准入与排队）。
        aligner_lock: 进程级锁，串行化 aligner 前向（transformers 模型无并发保证）。
        aligner_device / diarizer_device: 规范化设备串（normalize_device 后）。
        segment_gap_threshold / max_segment_seconds: segment 切分参数。
        speaker_attribution / speaker_merge_gap: 说话人归属模式（word 词级归属 /
            segment 段级投票）与 word 模式同人相邻段合并阈值（秒，<=0 不合并）。
        max_audio_seconds / max_audio_bytes: 音频时长（秒）与体积（字节）上限。
        align_batch_size: 对齐批大小（亦是标准模式 ASR 并发信号量上限）。
        served_model_names: 已加载模型名列表；空列表表示不做 model 名校验。
        vllm_primary_device: vLLM 主设备约定（默认 cuda:0）。
        engine_client: vLLM 引擎客户端（serve 钩子从 build_app 提取后注入）。
    """

    aligner: Optional[Any] = None
    diarizer: Optional[Any] = None
    processor: Optional[Any] = None
    scheduler: Optional[GpuScheduler] = None
    aligner_lock: threading.Lock = field(default_factory=threading.Lock)

    aligner_device: str = "cpu"
    diarizer_device: str = "cpu"
    segment_gap_threshold: float = 0.8
    max_segment_seconds: float = 30.0
    speaker_attribution: str = "word"
    speaker_merge_gap: float = 2.0
    max_audio_seconds: float = 3600.0
    max_audio_bytes: int = 500 * 1024 * 1024
    align_batch_size: int = 4
    served_model_names: List[str] = field(default_factory=list)
    vllm_primary_device: str = VLLM_PRIMARY_DEVICE
    engine_client: Optional[Any] = None

    @property
    def extensions_enabled(self) -> bool:
        """是否启用了任一扩展模型（aligner / diarizer）。"""
        return self.aligner is not None or self.diarizer is not None


def budget_devices(state: ExtensionState) -> Tuple[str, str]:
    """返回预算用的 (aligner 设备, diarizer 设备)。

    未启用的扩展以 "cpu" 屏蔽——cpu 设备不参与显存准入（见 scheduler
    `estimate_task_need_mb` / `device_index`），从而禁用侧不产生任何预算需求。
    """
    aligner_dev = state.aligner_device if state.aligner is not None else "cpu"
    diarizer_dev = state.diarizer_device if state.diarizer is not None else "cpu"
    return aligner_dev, diarizer_dev


def should_inject_gmu(
    aligner_dev: Optional[str],
    diarizer_dev: Optional[str],
    user_specified_gmu: bool,
) -> bool:
    """是否应自动注入 `--gpu-memory-utilization 0.70`（纯函数）。

    条件（spec「gpu_memory_utilization 自动调整」）：任一扩展启用且其设备归一后
    等于 vLLM 主设备约定 cuda:0（同进程 CUDA_VISIBLE_DEVICES 下自洽），且用户
    未显式指定该参数。本函数约定只在至少一个扩展启用时被调用；禁用侧设备
    传入 "cpu"（见 `budget_devices`），从而不参与判定。

    混合拓扑（如 aligner 在 cuda:0、diarizer 在 cuda:1）同样命中 any() 注入：
    只要任一扩展与 vLLM 共享 cuda:0，vLLM 预分配就必须为该扩展预留显存，
    与另一扩展所在设备无关。
    """
    if user_specified_gmu:
        return False
    devices = (normalize_device(aligner_dev), normalize_device(diarizer_dev))
    return any(d == VLLM_PRIMARY_DEVICE for d in devices)


def _device_roles(
    vllm_primary: str,
    aligner: Optional[Any],
    aligner_device: str,
    diarizer: Optional[Any],
    diarizer_device: str,
) -> Dict[str, str]:
    """构建 设备 -> 角色描述 表（供 /health/detail devices[].role 展示）。

    形如 {"cuda:0": "vllm+aligner+diarizer", "cuda:1": "diarizer"}。
    """
    roles: Dict[str, List[str]] = {}

    def _add(dev: str, part: str) -> None:
        name = normalize_device(dev)
        parts = roles.setdefault(name, [])
        if part not in parts:
            parts.append(part)

    _add(vllm_primary, "vllm")
    if aligner is not None:
        _add(aligner_device, "aligner")
    if diarizer is not None:
        _add(diarizer_device, "diarizer")
    return {dev: "+".join(parts) for dev, parts in roles.items()}


def _move_to_device(model: Any, device: str) -> None:
    """将已加载模型搬移到指定设备；cpu 或无 to() 能力时不动作。"""
    if not device or device == "cpu":
        return
    to_fn = getattr(model, "to", None)
    if callable(to_fn):
        to_fn(device)


def _load_aligner(name: str, device: str) -> Any:
    """加载强制对齐模型：优先 device_map + bfloat16，失败回退默认加载后手动搬移。

    dtype=torch.bfloat16 为可选优化项（不强制）：部分 transformers/硬件组合
    不接受该 kwarg 或 bf16 加载失败时自动降级，保证启动成功率。
    """
    from ..inference.qwen3_forced_aligner import Qwen3ForcedAligner

    dtype_kwargs: Dict[str, Any] = {}
    try:
        import torch

        dtype_kwargs["dtype"] = torch.bfloat16
    except Exception:  # pragma: no cover - torch 缺失时交由上层报错
        dtype_kwargs = {}
    try:
        return Qwen3ForcedAligner.from_pretrained(name, device_map=device, **dtype_kwargs)
    except Exception as first_error:
        logger.warning(
            "对齐模型按 device_map=%s + bfloat16 加载失败（%s），回退默认加载后手动搬移到 %s",
            device,
            first_error,
            device,
        )
        aligner = Qwen3ForcedAligner.from_pretrained(name)
        _move_to_device(aligner.model, device)
        return aligner


def _load_diarizer(name: str, device: str, token: Optional[str]) -> Any:
    """加载 pyannote 说话人识别管线（token 优先级：入参 > PYANNOTE_API_TOKEN > HF_TOKEN）。

    SpeakerDiarizer.from_pretrained 内部以 token= 传参并设置 HF_TOKEN 环境变量
    双保险，且自带 pipeline.to(device) 搬移，此处直接透传设备即可。
    """
    from ..inference.qwen3_speaker_diarizer import SpeakerDiarizer

    if token is None or str(token).strip() == "":
        token = os.environ.get("PYANNOTE_API_TOKEN") or os.environ.get("HF_TOKEN")
    token = str(token).strip() if token else None
    return SpeakerDiarizer.from_pretrained(
        name,
        use_auth_token=token if token else None,
        device=device if device else None,
    )


def load_extensions(
    ext_args: Any,
    model_path: Optional[str],
    served_model_names: Optional[List[str]] = None,
) -> ExtensionState:
    """加载服务扩展并执行启动校验（任一环节失败即抛异常终止启动）。

    Args:
        ext_args: serve.py 剥离出的扩展参数 Namespace（forced_aligner / diarizer /
            pyannote_token / aligner_device / diarizer_device / max_concurrent_tasks /
            gpu_reserve_mb / max_audio_seconds / max_audio_bytes /
            segment_gap_threshold / max_segment_seconds / align_batch_size /
            speaker_attribution / speaker_merge_gap）。
        model_path: vLLM --model 值，用于加载 Qwen3ASRProcessor（CPU 常驻）。
        served_model_names: 已加载模型名列表（缺省空列表表示不校验）。

    Returns:
        ExtensionState: 扩展运行态。

    Raises:
        RuntimeError: 模型路径缺失、启动显存校验失败等（中文消息）。
        ImportError: 依赖未安装（如 pyannote.audio 缺失，消息含安装指引）。
    """
    model_path = str(model_path or "").strip()
    if not model_path:
        raise RuntimeError(
            "无法确定 ASR 模型路径（--model / --model-tag），扩展初始化中止；"
            "请以 qwen-asr-serve <model_path> 或 --model <model_path> 方式启动"
        )

    logger.info("加载 Qwen3ASRProcessor（CPU 常驻，仅 tokenizer/特征提取器）: %s", model_path)
    from ..core.transformers_backend import Qwen3ASRProcessor

    processor = Qwen3ASRProcessor.from_pretrained(model_path, fix_mistral_regex=True)

    aligner_name = str(getattr(ext_args, "forced_aligner", "") or "").strip()
    diarizer_name = str(getattr(ext_args, "diarizer", "") or "").strip()
    aligner_device = normalize_device(
        str(getattr(ext_args, "aligner_device", "") or VLLM_PRIMARY_DEVICE)
    )
    diarizer_device = normalize_device(
        str(getattr(ext_args, "diarizer_device", "") or VLLM_PRIMARY_DEVICE)
    )
    vllm_primary = normalize_device(
        str(getattr(ext_args, "vllm_primary_device", "") or VLLM_PRIMARY_DEVICE)
    )

    aligner = None
    if aligner_name:
        logger.info("加载强制对齐扩展: %s -> %s", aligner_name, aligner_device)
        aligner = _load_aligner(aligner_name, aligner_device)
    else:
        logger.info("对齐扩展已显式禁用（--forced-aligner 空串），不加载")

    diarizer = None
    if diarizer_name:
        logger.info("加载说话人识别扩展: %s -> %s", diarizer_name, diarizer_device)
        diarizer = _load_diarizer(
            diarizer_name, diarizer_device, getattr(ext_args, "pyannote_token", None)
        )
    else:
        logger.info("说话人识别扩展已显式禁用（--diarizer 空串），不加载")

    scheduler = GpuScheduler(
        max_concurrent_tasks=int(getattr(ext_args, "max_concurrent_tasks", 2) or 2),
        gpu_reserve_mb=int(getattr(ext_args, "gpu_reserve_mb", 1024) or 1024),
        devices=_device_roles(vllm_primary, aligner, aligner_device, diarizer, diarizer_device),
    )

    align_batch = int(getattr(ext_args, "align_batch_size", 4) or 4)
    # merge_gap 显式 0 合法（不合并），不能走 `or 默认值` 的 falsy 回退
    merge_gap_raw = getattr(ext_args, "speaker_merge_gap", None)
    speaker_merge_gap = float(merge_gap_raw) if merge_gap_raw is not None else 2.0

    # 启动快速失败校验（仅扩展启用时）：按设备最小瞬态需求——
    # max_concurrent_tasks=1、align_batch_size、30s 音频——空闲不足即抛 RuntimeError
    if aligner is not None or diarizer is not None:
        budget_aligner = aligner_device if aligner is not None else "cpu"
        budget_diarizer = diarizer_device if diarizer is not None else "cpu"
        min_need = estimate_task_need_mb(align_batch, 30, budget_aligner, budget_diarizer)
        logger.info("执行 GPU 显存启动校验（按设备最小瞬态需求 %s）", min_need)
        scheduler.start_up_validate(min_need)

    return ExtensionState(
        aligner=aligner,
        diarizer=diarizer,
        processor=processor,
        scheduler=scheduler,
        aligner_device=aligner_device,
        diarizer_device=diarizer_device,
        segment_gap_threshold=float(getattr(ext_args, "segment_gap_threshold", 0.8) or 0.8),
        max_segment_seconds=float(getattr(ext_args, "max_segment_seconds", 30.0) or 30.0),
        speaker_attribution=str(getattr(ext_args, "speaker_attribution", "word") or "word"),
        speaker_merge_gap=speaker_merge_gap,
        max_audio_seconds=float(getattr(ext_args, "max_audio_seconds", 3600.0) or 3600.0),
        max_audio_bytes=int(getattr(ext_args, "max_audio_bytes", 500 * 1024 * 1024) or 500 * 1024 * 1024),
        align_batch_size=align_batch,
        served_model_names=list(served_model_names or []),
        vllm_primary_device=vllm_primary,
    )
