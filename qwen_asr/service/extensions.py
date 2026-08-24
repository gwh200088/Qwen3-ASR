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
        punctuation_split: 句末标点硬切分开关（True 恒切分 + 标点附前段末尾；
            False 纯间隙切分，segment_split_mode 被忽略）。
        segment_split_mode: 切分维度模式——punctuation（默认，只按句末标点 +
            段长兜底，静音间隙与句中说话人变化均不切分）或 hybrid（标点 +
            间隙 + 说话人变化，上一代行为）；仅 punctuation_split=True 生效，
            segment_gap_threshold 与 speaker_merge_gap 仅 hybrid 生效。
        diarization_min_speakers / diarization_max_speakers: 说话人数约束的
            服务级默认（None 不约束）；请求级 min_speakers/max_speakers
            form 参数未传时回退到此默认，透传 pyannote 聚类约束。
        diarization_clustering_threshold: 说话人聚类阈值服务级覆写（None =
            管线默认，具体值以部署机模型 config.yaml 为准；调低更倾向拆分
            说话人，过度调低会过分割一人成多）。
        diarizer_embedding: diarization 声纹向量化模式——wespeaker（默认，
            community-1 管线现状）或 campplus（中文域 CAM++ 声纹 + 3.1 式
            AHC 余弦聚类，缓解中文男声相近被合并）。
        diarizer_embedding_model: CAM++ 声纹模型本地目录（campplus 模式
            必填；wespeaker 模式下传入被忽略并告警）。
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
    segment_gap_threshold: float = 2.0
    max_segment_seconds: float = 30.0
    speaker_attribution: str = "word"
    speaker_merge_gap: float = 2.0
    punctuation_split: bool = True
    segment_split_mode: str = "punctuation"
    diarization_min_speakers: Optional[int] = None
    diarization_max_speakers: Optional[int] = None
    diarization_clustering_threshold: Optional[float] = None
    diarizer_embedding: str = "wespeaker"
    diarizer_embedding_model: Optional[str] = None
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


def _load_diarizer(
    name: str,
    device: str,
    token: Optional[str],
    embedding: str = "wespeaker",
    embedding_model: Optional[str] = None,
) -> Any:
    """加载 pyannote 说话人识别管线（token 优先级：入参 > PYANNOTE_API_TOKEN > HF_TOKEN）。

    SpeakerDiarizer.from_pretrained 内部以 token= 传参并设置 HF_TOKEN 环境变量
    双保险，且自带 pipeline.to(device) 搬移（campplus 模式下 CAM++ 组件随其后
    显式搬移），此处直接透传设备与 embedding 模式即可。

    embedding="campplus" 时加载后组件替换注入 CAM++ 中文声纹 + 3.1 式 AHC
    聚类（生效机制与模型维度由 SpeakerDiarizer 内 INFO 日志输出；模型目录
    缺失/注入失败 → 中文 RuntimeError，不静默回退）。
    """
    from ..inference.qwen3_speaker_diarizer import SpeakerDiarizer

    if token is None or str(token).strip() == "":
        token = os.environ.get("PYANNOTE_API_TOKEN") or os.environ.get("HF_TOKEN")
    token = str(token).strip() if token else None
    return SpeakerDiarizer.from_pretrained(
        name,
        use_auth_token=token if token else None,
        device=device if device else None,
        embedding=embedding,
        embedding_model=embedding_model,
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
            speaker_attribution / speaker_merge_gap / punctuation_split /
            segment_split_mode / diarization_min_speakers /
            diarization_max_speakers / diarization_clustering_threshold /
            diarizer_embedding / diarizer_embedding_model）。
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

    # ---- 新增服务级参数解析 + 组合校验/告警（spec「segment 切分维度模式」
    # 「说话人数约束服务级默认」「聚类阈值服务级覆写」「CAM++ 集成」）----------
    # embedding 模式解析须在 diarizer 加载之前（campplus 参数透传加载路径）
    diarizer_embedding = str(getattr(ext_args, "diarizer_embedding", "wespeaker") or "wespeaker")
    diarizer_embedding_model = str(
        getattr(ext_args, "diarizer_embedding_model", "") or ""
    ).strip() or None
    if diarizer_embedding == "campplus":
        if not diarizer_name:
            # diarizer 整体禁用：embedding 参数无效果（diarization 关闭语义不变）
            logger.warning(
                "--diarizer-embedding campplus 与 --diarizer 显式禁用组合："
                "diarization 整体关闭，embedding 相关参数（含 --diarizer-embedding-model）"
                "无效果，按现状语义启动。"
            )
        elif not diarizer_embedding_model:
            raise RuntimeError(
                "--diarizer-embedding campplus 必须提供 --diarizer-embedding-model "
                "<目录>（CAM++ 声纹模型，如 speech_campplus_sv_zh-cn_16k-common 的本地"
                "路径）；回退上一代行为请改用 --diarizer-embedding wespeaker。"
            )
    elif diarizer_embedding_model:
        # wespeaker 模式传了模型目录：忽略 + WARNING
        logger.warning(
            "--diarizer-embedding wespeaker 模式下 --diarizer-embedding-model %s "
            "被忽略（该参数仅 campplus 模式生效），按 community-1 管线现状运行。",
            diarizer_embedding_model,
        )
        diarizer_embedding_model = None

    diarizer = None
    if diarizer_name:
        logger.info(
            "加载说话人识别扩展: %s -> %s（embedding=%s）",
            diarizer_name,
            diarizer_device,
            diarizer_embedding,
        )
        diarizer = _load_diarizer(
            diarizer_name,
            diarizer_device,
            getattr(ext_args, "pyannote_token", None),
            embedding=diarizer_embedding,
            embedding_model=diarizer_embedding_model,
        )
    else:
        logger.info("说话人识别扩展已显式禁用（--diarizer 空串），不加载")

    # segment_split_mode：None = 未显式传入 → 缺省 punctuation；
    # --punctuation-split off 时 mode 恒被忽略（纯间隙行为），无论是否显式
    # 传入 mode 均输出 WARNING（spec「punctuation-split off 组合告警」字面语义）
    split_mode_raw = getattr(ext_args, "segment_split_mode", None)
    segment_split_mode = str(split_mode_raw or "punctuation")
    punctuation_split = str(getattr(ext_args, "punctuation_split", "on") or "on") == "on"
    if not punctuation_split:
        logger.warning(
            "--punctuation-split off 时 segment 切分为纯间隙/段长（word 模式含说话人"
            "变化）行为，--segment-split-mode %s 被忽略；如需标点+间隙+说话人三维混合"
            "请改为 --punctuation-split on --segment-split-mode hybrid。",
            segment_split_mode,
        )

    # 说话人数约束服务级默认：非法组合启动即失败（fast fail，不区分 diarizer 状态）
    diar_min_raw = getattr(ext_args, "diarization_min_speakers", None)
    diar_max_raw = getattr(ext_args, "diarization_max_speakers", None)
    diarization_min_speakers = int(diar_min_raw) if diar_min_raw is not None else None
    diarization_max_speakers = int(diar_max_raw) if diar_max_raw is not None else None
    for label, value in (
        ("--diarization-min-speakers", diarization_min_speakers),
        ("--diarization-max-speakers", diarization_max_speakers),
    ):
        if value is not None and value < 1:
            raise RuntimeError(
                f"{label} 须为不小于 1 的整数（收到: {value}）；"
                "说话人数约束无意义，请修正后重启"
            )
    if (
        diarization_min_speakers is not None
        and diarization_max_speakers is not None
        and diarization_min_speakers > diarization_max_speakers
    ):
        raise RuntimeError(
            f"--diarization-min-speakers ({diarization_min_speakers}) 大于 "
            f"--diarization-max-speakers ({diarization_max_speakers})，"
            "约束自相矛盾，请修正后重启"
        )

    # 聚类阈值：argparse 已校验区间 (0, 2)，此处仅透传（应用在 diarizer 加载后
    # 防御式探测，见 qwen3_speaker_diarizer）
    diarization_clustering_threshold = getattr(ext_args, "diarization_clustering_threshold", None)
    if diarization_clustering_threshold is not None:
        diarization_clustering_threshold = float(diarization_clustering_threshold)

    # diarizer 禁用时说话人调优参数无效果（告警但不阻断，比照 embedding 先例）
    if not diarizer_name:
        ineffective = [
            flag
            for flag, value in (
                ("--diarization-min-speakers", diarization_min_speakers),
                ("--diarization-max-speakers", diarization_max_speakers),
                ("--diarization-clustering-threshold", diarization_clustering_threshold),
            )
            if value is not None
        ]
        if ineffective:
            logger.warning(
                "--diarizer 显式禁用时 %s 无效果（diarization 整体关闭），按现状语义启动。",
                " / ".join(ineffective),
            )

    # 聚类阈值防御式应用（diarizer 加载后 best-effort，全部机制不可用则 WARNING
    # 后正常启动——spec「聚类阈值服务级覆写」：AHC 路径 instantiate 为预期主机制）
    if diarization_clustering_threshold is not None and diarizer is not None:
        mechanism = diarizer.apply_clustering_threshold(diarization_clustering_threshold)
        if mechanism:
            logger.info(
                "说话人聚类阈值已覆写为 %s（生效机制: %s；调低更倾向拆分说话人，"
                "过度调低会过分割一人成多）",
                diarization_clustering_threshold,
                mechanism,
            )
        else:
            logger.warning(
                "聚类阈值 %s 应用失败：当前管线不支持任一探测机制"
                "（instantiate 超参 / clustering_threshold 属性 / 嵌套超参覆写），"
                "将按管线默认阈值运行（具体默认值以部署机模型 config.yaml 为准）。",
                diarization_clustering_threshold,
            )

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
        segment_gap_threshold=float(getattr(ext_args, "segment_gap_threshold", 2.0) or 2.0),
        max_segment_seconds=float(getattr(ext_args, "max_segment_seconds", 30.0) or 30.0),
        speaker_attribution=str(getattr(ext_args, "speaker_attribution", "word") or "word"),
        speaker_merge_gap=speaker_merge_gap,
        punctuation_split=punctuation_split,
        segment_split_mode=segment_split_mode,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        diarization_clustering_threshold=diarization_clustering_threshold,
        diarizer_embedding=diarizer_embedding,
        diarizer_embedding_model=diarizer_embedding_model,
        max_audio_seconds=float(getattr(ext_args, "max_audio_seconds", 3600.0) or 3600.0),
        max_audio_bytes=int(getattr(ext_args, "max_audio_bytes", 500 * 1024 * 1024) or 500 * 1024 * 1024),
        align_batch_size=align_batch,
        served_model_names=list(served_model_names or []),
        vllm_primary_device=vllm_primary,
    )
