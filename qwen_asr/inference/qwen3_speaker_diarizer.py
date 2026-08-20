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
SpeakerDiarizer：基于 pyannote.audio 的说话人识别（diarization）SDK 封装。

pyannote 集成防御性设计（兼容 3.x / 4.x）：
  - pyannote 为可选依赖（pip install qwen-asr[diarization]），模块顶部探测式导入，
    未安装时 Pipeline 为 None，由 from_pretrained 抛出带安装指引的 ImportError；
  - 认证优先以 token= 传参（新版 huggingface_hub 已移除 use_auth_token 透传），
    旧版 pyannote 3.1 不认 token= 时回退 use_auth_token=；同时设置 HF_TOKEN
    环境变量双保险；
  - 前向优先 pyannote 4.x 的 pipeline.diarize()，无该属性时回退 3.x 的
    pipeline()（__call__）；说话人数约束先经 inspect.signature 过滤，仅透传
    前向函数实际支持的参数（签名不可解析时原样透传，由 TypeError 降级重试兜底），
    避免 except TypeError 盲目回退掩盖管线内部真实错误；
  - 返回值防御性归一：4.x DiarizeOutput 带 speaker_diarization 属性则取之，
    否则将原对象视为 3.x Annotation 使用，统一经 itertracks(yield_label=True)
    收集片段；
  - min/max_speakers 约束 best-effort 透传：签名过滤后仍抛 TypeError（管线
    不支持该参数）时，以 warnings.warn 提示后去掉约束重试一次；
  - pyannote Pipeline 无并发调用安全保证，进程级 threading.Lock 串行化前向。
"""
import inspect
import os
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

# pyannote 为可选依赖：导入失败时保持 Pipeline = None，由 from_pretrained 统一报错
try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None

from .utils import (
    AudioLike,
    SAMPLE_RATE,
    normalize_audios,
)


@dataclass(frozen=True)
class DiarizationSegment:
    """
    单个说话人片段。

    Attributes:
        speaker (str): 说话人标签（如 "SPEAKER_00"）。
        start_time (float): 片段开始时间（秒）。
        end_time (float): 片段结束时间（秒）。
    """

    speaker: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class DiarizationResult:
    """
    单条音频的说话人识别结果。

    Attributes:
        segments (List[DiarizationSegment]):
            按开始时间升序排列的说话人片段；空结果为空列表。
    """

    segments: List[DiarizationSegment]

    @property
    def speakers(self) -> List[str]:
        """去重后的说话人标签列表（按标签名升序）。"""
        return sorted({segment.speaker for segment in self.segments})

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> DiarizationSegment:
        return self.segments[idx]


class SpeakerDiarizer:
    """
    pyannote.audio 说话人识别管线的 HuggingFace 风格封装。

    接口风格参照 Qwen3ForcedAligner：
      - from_pretrained() 工厂方法加载 pyannote Pipeline（显式 use_auth_token /
        device 常用参数）；
      - diarize() 对单条或一批音频做说话人识别，返回 List[DiarizationResult]。

    音频输入复用 utils.normalize_audios 归一化为 16k 单声道 float32（支持
    路径 / URL / base64 / (ndarray, sr) 及其列表）。
    """

    def __init__(self, pipeline: Any, device: Optional[str] = None):
        self.pipeline = pipeline
        self.device = device
        # pyannote Pipeline 无并发调用安全保证：进程级锁串行化前向，
        # 等待中的调用方仅持 CPU 张量，限制瞬态显存仅一个前向
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "pyannote/speaker-diarization-community-1",
        use_auth_token: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "SpeakerDiarizer":
        """
        加载 pyannote 说话人识别管线并初始化 SpeakerDiarizer。

        Args:
            pretrained_model_name_or_path (str):
                pyannote 管线名称（如 "pyannote/speaker-diarization-community-1"、
                legacy "pyannote/speaker-diarization-3.1"）或本地路径。
            use_auth_token (Optional[str]):
                HuggingFace 访问令牌（pyannote 门控模型必填）。提供时会先以
                setdefault 设置 HF_TOKEN 环境变量双保险，加载时优先 token= 传参，
                旧版 pyannote 抛 TypeError 时回退 use_auth_token=。
            device (Optional[str]):
                推理设备（如 "cuda:1"）。默认 None 不搬移；提供且管线支持 to()
                时调用 pipeline.to(device)。
            **kwargs:
                其余参数透传 Pipeline.from_pretrained(...)。

        Returns:
            SpeakerDiarizer: 初始化后的封装实例。

        Raises:
            ImportError: 未安装 pyannote.audio 时抛出，提示安装
                pip install qwen-asr[diarization]。
        """
        if Pipeline is None:
            raise ImportError(
                "pyannote.audio is required for SpeakerDiarizer but not installed. "
                "Install the diarization extra with: pip install qwen-asr[diarization]"
            )

        if use_auth_token is not None:
            # 双保险：部分依赖链（旧版 huggingface_hub 等）只读 HF_TOKEN 环境变量
            os.environ.setdefault("HF_TOKEN", use_auth_token)
            try:
                pipeline = Pipeline.from_pretrained(
                    pretrained_model_name_or_path, token=use_auth_token, **kwargs
                )
            except TypeError:
                # 旧版 pyannote（3.1）的 from_pretrained 不认 token=，回退 use_auth_token=
                pipeline = Pipeline.from_pretrained(
                    pretrained_model_name_or_path, use_auth_token=use_auth_token, **kwargs
                )
        else:
            pipeline = Pipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if device is not None and hasattr(pipeline, "to"):
            # pyannote 4.x 的 Pipeline.to() 严格要求 torch.device 实例（传 str 抛
            # TypeError）；3.x 则两者皆可。统一转 torch.device 兼容两版。
            pipeline.to(device if isinstance(device, torch.device) else torch.device(device))

        return cls(pipeline=pipeline, device=device)

    @staticmethod
    def _filter_constraints(fn: Any, constraints: Dict[str, int]) -> Optional[Dict[str, int]]:
        """按前向函数签名过滤说话人数约束（避免盲目 try/except 掩盖真实 TypeError）。

        - 签名可解析：仅保留 fn 形参中存在的约束项（或 fn 含 ``**kwargs`` 时全部保留）；
        - 签名不可解析（C 扩展等）：返回 None 表示无法判断，由调用方走 try/except 兜底。
        """
        try:
            parameters = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return None
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return dict(constraints)
        return {k: v for k, v in constraints.items() if k in parameters}

    def _invoke_pipeline(self, inp: Dict[str, Any], constraints: Dict[str, int]) -> Any:
        """调用 pyannote 前向：优先 4.x diarize()，无该属性时回退 3.x __call__()。

        约束参数先经签名过滤（签名不可解析时原样透传，由 _run_one 的
        TypeError 降级重试兜底）；不再用 except TypeError 盲目回退 __call__——
        那会掩盖管线内部真实的 TypeError 并多一次必然失败的调用。
        """
        fn = getattr(self.pipeline, "diarize", None)
        if not callable(fn):
            fn = self.pipeline
        filtered = self._filter_constraints(fn, constraints)
        kwargs = constraints if filtered is None else filtered
        return fn(inp, **kwargs)

    def _run_one(
        self,
        inp: Dict[str, Any],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> Any:
        """在锁内对单条音频执行前向，说话人数约束 best-effort 降级重试。"""
        # 约束参数仅在非 None 时传入，避免覆盖管线默认行为
        constraints: Dict[str, int] = {}
        if min_speakers is not None:
            constraints["min_speakers"] = min_speakers
        if max_speakers is not None:
            constraints["max_speakers"] = max_speakers

        with self._lock:
            try:
                return self._invoke_pipeline(inp, constraints)
            except TypeError:
                if not constraints:
                    raise
                # 管线不支持 min/max_speakers 约束：警告后去掉约束重试一次
                warnings.warn(
                    "pyannote pipeline rejected speaker-count constraints "
                    f"{constraints}; retrying without them (best-effort).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._invoke_pipeline(inp, {})

    @staticmethod
    def _to_result(out: Any) -> DiarizationResult:
        """对 pyannote 返回值做防御性归一，转换为 DiarizationResult。"""
        # 4.x DiarizeOutput 带 speaker_diarization 属性则取之，否则视为 3.x Annotation
        annotation = getattr(out, "speaker_diarization", None)
        if annotation is None:
            annotation = out

        segments: List[DiarizationSegment] = []
        if annotation is not None:
            for turn, _, label in annotation.itertracks(yield_label=True):
                segments.append(
                    DiarizationSegment(
                        speaker=str(label),
                        start_time=float(turn.start),
                        end_time=float(turn.end),
                    )
                )
        segments.sort(key=lambda s: s.start_time)
        return DiarizationResult(segments=segments)

    def diarize(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[DiarizationResult]:
        """
        对单条或一批音频做说话人识别。

        Args:
            audio:
                音频输入。每项支持本地路径 / https URL / base64 字符串 /
                (np.ndarray, sr)，均会被归一化为 16k 单声道 float32。
            min_speakers (Optional[int]):
                最少说话人数（best-effort 透传 pyannote，不支持时警告降级）。
            max_speakers (Optional[int]):
                最多说话人数（同上）。

        Returns:
            List[DiarizationResult]:
                每条音频一个结果；segments 为按开始时间升序的
                DiarizationSegment(speaker, start_time, end_time) 列表，
                时间单位为秒；无说话人片段时 segments 为空列表。
        """
        wavs = normalize_audios(audio)

        results: List[DiarizationResult] = []
        for wav in wavs:
            inp = {
                "waveform": torch.from_numpy(wav).float().unsqueeze(0),
                "sample_rate": SAMPLE_RATE,
            }
            out = self._run_one(inp, min_speakers, max_speakers)
            results.append(self._to_result(out))
        return results
