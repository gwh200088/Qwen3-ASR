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

CAM++ 中文声纹集成（``embedding="campplus"``，spec「CAM++ 中文声纹 embedding
集成」）：
  - 注入机制采用 **加载后组件替换**（spec 注入机制优先级中的 b）：正常加载
    基础管线（community-1，segmentation/PLDA 等依赖齐全）后，将
    ``pipeline._embedding`` 替换为 ``CampplusSpeakerEmbedding``（192 维中文
    声纹），``pipeline.clustering`` 替换为 pyannote ``AgglomerativeClustering``
    （3.1 式 AHC + 余弦相似度，不涉及 PLDA/VBx），并按 3.1 官方调优值实例化
    AHC 超参（method=centroid / threshold=0.515771 / min_cluster_size=12）；
  - 组件替换细节：``clustering`` 赋值经 pyannote Pipeline.__setattr__ 的
    Pipeline 分支自动清理旧 VBx 组件注册（``_pipelines``）；``_embedding``
    赋值因 wrapper 非 BaseInference 走 object.__setattr__，需手动
    ``_inferences.pop("_embedding")`` 清理旧 WeSpeaker 注册（防 ``to()``
    触碰已弃组件）；替换组件不在 ``to()`` 传播字典内，设备搬移由
    from_pretrained 显式执行（``pipeline.to`` 后 ``embedder.to``）；
  - ``apply_clustering_threshold`` 的 instantiate 机制与 AHC 替换天然兼容
    （管线级 instantiate 递归进 ``_pipelines["clustering"]``，``--diarization-
    clustering-threshold`` 对 campplus 路径同样生效）；
  - min/max_speakers 约束在 AHC 聚类下透传（``BaseClustering.__call__`` 的
    min_clusters/max_clusters 形参）；
  - fail fast：CAM++ 模型目录缺失/权重损坏/注入任一步失败 → 中文
    RuntimeError（含目录、期望文件与 ``--diarizer-embedding wespeaker``
    回退提示），不静默回退。
"""
import inspect
import logging
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

logger = logging.getLogger(__name__)


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
        embedding: str = "wespeaker",
        embedding_model: Optional[str] = None,
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
            embedding (str):
                声纹向量化模式：``"wespeaker"``（默认，community-1 管线现状）/
                ``"campplus"``（加载后组件替换为 CAM++ 中文声纹 + 3.1 式 AHC
                余弦聚类，缓解中文男声相近被合并）。
            embedding_model (Optional[str]):
                CAM++ 声纹模型本地目录（``embedding="campplus"`` 时必填，
                ModelScope ``iic/speech_campplus_sv_zh-cn_16k-common`` 产物，
                含 ``campplus_cn_common.bin`` / ``config.yaml``）。
            **kwargs:
                其余参数透传 Pipeline.from_pretrained(...)。

        Returns:
            SpeakerDiarizer: 初始化后的封装实例。

        Raises:
            ImportError: 未安装 pyannote.audio 时抛出，提示安装
                pip install qwen-asr[diarization]。
            RuntimeError: ``embedding="campplus"`` 时模型目录缺失、权重损坏
                或组件注入失败（中文消息含回退提示，不静默回退）。
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

        embedder = None
        if str(embedding) == "campplus":
            # 组件替换注入（spec 注入机制 b）：embedding → CAM++，clustering → AHC
            embedder = cls._inject_campplus(pipeline, embedding_model)

        if device is not None and hasattr(pipeline, "to"):
            # pyannote 4.x 的 Pipeline.to() 严格要求 torch.device 实例（传 str 抛
            # TypeError）；3.x 则两者皆可。统一转 torch.device 兼容两版。
            pipeline.to(device if isinstance(device, torch.device) else torch.device(device))
            if embedder is not None:
                # 替换组件不在 pipeline.to() 传播字典内（非 BaseInference），
                # CAM++ 模型搬移由此显式执行
                embedder.to(device)

        diarizer = cls(pipeline=pipeline, device=device)
        if embedder is not None:
            # 保留引用：设备归属查询与防 GC（pipeline._embedding 同持引用）
            diarizer._embedding_override = embedder
        return diarizer

    #: AHC 超参（pyannote/speaker-diarization-3.1 官方 config.yaml 调优值；
    #: cosine + centroid 链路，threshold 可被 --diarization-clustering-threshold
    #: 覆写——apply_clustering_threshold 的 instantiate 机制递归进 AHC 组件）
    _AHC_DEFAULTS = {"method": "centroid", "threshold": 0.515771, "min_cluster_size": 12}

    @classmethod
    def _inject_campplus(cls, pipeline: Any, embedding_model: Optional[str]) -> Any:
        """加载后组件替换注入 CAM++ 声纹与 3.1 式 AHC 聚类（spec 注入机制 b）。

        替换步骤（详见模块 docstring「CAM++ 中文声纹集成」）：

        1. ``CampplusSpeakerEmbedding.from_pretrained`` fail fast 加载
           （模型目录缺失/权重损坏 → 中文 RuntimeError）；
        2. 管线结构校验（``_embedding`` / ``clustering`` 属性存在性）——必须在
           任何清理动作**之前**执行：pyannote 4.x 的 ``_embedding`` 是
           BaseInference，仅注册于 ``_inferences`` 字典（不在实例 ``__dict__``），
           先 pop 再校验会自删自检、误报"缺少 _embedding 属性"；
        3. ``pipeline._inferences.pop("_embedding")`` 清理旧 WeSpeaker 的
           BaseInference 注册（防 ``pipeline.to()`` 触碰已弃组件）；
        4. ``pipeline._embedding = embedder``（wrapper 非 BaseInference，经
           object.__setattr__ 直存实例字典）；
        5. ``pipeline.clustering = AgglomerativeClustering(metric=embedder.metric)``
           （经 __setattr__ Pipeline 分支自动覆盖 ``_pipelines["clustering"]``，
           旧 VBx 组件引用随之释放）；
        6. 按 ``_AHC_DEFAULTS`` 实例化 AHC 超参（Parameter 未实例化时前向
           fcluster 会收到 Uniform 对象而崩溃，必须显式 instantiate）；
        7. ``pipeline._expects_num_speakers`` 按 AHC 重算（False，与 VBx 一致，
           防御式保持一致语义）。

        Args:
            pipeline: 已加载的 pyannote SpeakerDiarization 管线实例。
            embedding_model: CAM++ 模型本地目录（campplus 模式由调用方保证非空）。

        Returns:
            CampplusSpeakerEmbedding: 注入的声纹组件（调用方持有引用做设备搬移）。

        Raises:
            RuntimeError: embedding_model 缺失、CAM++ 加载失败、管线结构不符合
                组件替换预期（缺 ``_embedding`` / ``clustering`` 属性）或 AHC
                超参实例化失败——中文消息含回退提示，不静默回退。
        """
        if not embedding_model or not str(embedding_model).strip():
            raise RuntimeError(
                "--diarizer-embedding campplus 必须提供 --diarizer-embedding-model "
                "<目录>（ModelScope iic/speech_campplus_sv_zh-cn_16k-common 产物，"
                "含 campplus_cn_common.bin / config.yaml）；回退上一代行为请改用 "
                "--diarizer-embedding wespeaker。"
            )

        from .campplus_speaker_embedding import CampplusSpeakerEmbedding

        # 1. fail fast 加载 CAM++（目录/权重校验由 wrapper 内部完成）
        embedder = CampplusSpeakerEmbedding.from_pretrained(str(embedding_model).strip())

        try:
            # 2. 管线结构校验（先于清理：_embedding 是 BaseInference，仅存在于
            #    _inferences 注册，pop 之后再 hasattr 会自删自检误报缺失）
            if not hasattr(pipeline, "_embedding"):
                raise AttributeError("pipeline 缺少 _embedding 属性")
            if not hasattr(pipeline, "clustering"):
                raise AttributeError("pipeline 缺少 clustering 属性")

            # 3. 清理旧 WeSpeaker 的 BaseInference 注册（_embedding 键）：
            #    4.x core Pipeline.__setattr__ 对非 BaseInference 赋值不做
            #    remove_from，旧注册残留会导致 pipeline.to() 继续触碰弃用组件
            inferences = getattr(pipeline, "_inferences", None)
            if isinstance(inferences, dict):
                inferences.pop("_embedding", None)

            # 4. 替换声纹组件（object.__setattr__ 直存，前向经 self._embedding 调用）
            pipeline._embedding = embedder

            # 5. 替换聚类为 3.1 式 AHC（__setattr__ Pipeline 分支自动清理 VBx 注册）
            from pyannote.audio.pipelines.clustering import AgglomerativeClustering

            ahc = AgglomerativeClustering(metric=embedder.metric)
            pipeline.clustering = ahc

            # 6. 实例化 AHC 超参（Parameter 未实例化时前向会崩溃）
            ahc.instantiate(dict(cls._AHC_DEFAULTS))

            # 7. 防御式重算人数约束语义（AHC 与 VBx 均 False，保持一致）
            pipeline._expects_num_speakers = bool(
                getattr(pipeline.clustering, "expects_num_clusters", False)
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"CAM++ 组件注入失败（embedding/clustering 替换或 AHC 超参实例化）: "
                f"{exc}；请确认 pyannote.audio==4.0.7 且 --diarizer 指向完整的 "
                "community-1 模型目录（含 segmentation/plda 子目录）；回退上一代"
                "行为请改用 --diarizer-embedding wespeaker。"
            ) from exc

        logger.info(
            "CAM++ 注入完成（生效机制: 加载后组件替换）: embedding=CAM++(%d 维, %s) "
            "→ clustering=AgglomerativeClustering(method=%s, threshold=%s, "
            "min_cluster_size=%s, metric=%s)；说话人聚类切换为 3.1 式 AHC 余弦路径。",
            embedder.dimension,
            getattr(embedder, "sample_rate", 16000),
            cls._AHC_DEFAULTS["method"],
            cls._AHC_DEFAULTS["threshold"],
            cls._AHC_DEFAULTS["min_cluster_size"],
            embedder.metric,
        )
        return embedder

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

    def apply_clustering_hparams(self, **params: Any) -> Optional[str]:
        """启动期把 AHC 聚类超参写入 clustering 组件（多候选探测）。

        与 ``apply_clustering_threshold`` 同一套探测机制，但支持**一次性**写入
        多个超参。必须一次写入的原因：pyannote 的 ``instantiate`` 每次调用只
        覆盖 dict 中给出的键；若分多次调用不同键，探测阶段任一环节失败都会
        留下"部分生效"的中间态，难以从日志判断实际生效配置。

        Args:
            **params: 写入 ``clustering`` 组件的超参键值对，如
                ``threshold=0.5, min_cluster_size=3``。

        Returns:
            生效机制名；None 表示全部机制不可用（组件保持默认超参）。
        """
        if not params:
            return None

        # 候选 1：pyannote 4.x 管线级 instantiate（AHC 路径预期主机制）
        instantiate = getattr(self.pipeline, "instantiate", None)
        if callable(instantiate):
            try:
                instantiate({"clustering": dict(params)})
                return "instantiate"
            except Exception as exc:
                logger.debug("instantiate 机制应用聚类超参失败: %s", exc)

        # 候选 2：直接对 clustering 组件 instantiate（绕过管线级封装）
        clustering = getattr(self.pipeline, "clustering", None)
        cluster_instantiate = getattr(clustering, "instantiate", None)
        if callable(cluster_instantiate):
            try:
                cluster_instantiate(dict(params))
                return "clustering.instantiate"
            except Exception as exc:
                logger.debug("clustering.instantiate 应用聚类超参失败: %s", exc)

        # 候选 3：组件属性直写（无 instantiate 时的降级路径）
        if clustering is not None:
            applied = []
            try:
                for key, value in params.items():
                    if hasattr(clustering, key):
                        setattr(clustering, key, value)
                        applied.append(key)
            except Exception as exc:
                logger.debug("clustering 属性直写失败: %s", exc)
            if applied:
                return "attribute:" + ",".join(applied)

        return None

    def apply_clustering_threshold(self, threshold: float) -> Optional[str]:
        """启动期防御式应用聚类阈值（多候选探测），返回生效机制名或 None。

        本机/离线环境无法穷举 pyannote 各版本管线的超参结构（community-1 的
        VBx 与 3.1 式 AHC 路径结构不同），按优先级尝试三个候选机制，任一
        不抛异常即视为生效并返回机制名（供日志/部署手册回填确认）；全部
        失败返回 None，由调用方 WARNING 后按管线默认阈值正常启动。

        候选机制（spec「聚类阈值服务级覆写」）：

        1. ``"instantiate"``：pyannote 4.x ``pipeline.instantiate({"clustering":
           {"threshold": t}})`` 超参机制——AHC 路径预期主机制；
        2. ``"attribute"``：3.1 式 ``SpeakerDiarization`` 实例属性
           ``clustering_threshold`` 直改（3.x from_pretrained 以构造参数
           注入超参，实例属性即生效配置）；
        3. ``"hparams"``：管线持有嵌套超参 dict（``hparams`` / ``_hparams``）
           且含 ``clustering.threshold`` 键时原地覆写（4.x 备选结构）。

        Args:
            threshold (float): 聚类阈值（调用方已校验 ``0 < t < 2``；
                调低更倾向拆分说话人，过度调低会过分割一人成多）。

        Returns:
            Optional[str]: 生效机制名（上述三者之一）；None 表示全部机制
            不可用（管线默认阈值保持不变）。
        """
        threshold = float(threshold)

        # 候选 1：pyannote 4.x instantiate 超参机制（AHC 预期主机制）
        instantiate = getattr(self.pipeline, "instantiate", None)
        if callable(instantiate):
            try:
                instantiate({"clustering": {"threshold": threshold}})
                return "instantiate"
            except Exception as exc:
                logger.debug("instantiate 机制应用聚类阈值失败: %s", exc)

        # 候选 2：3.1 式实例属性直改（属性存在即可写，3.x 构造参数注入机制）
        if hasattr(self.pipeline, "clustering_threshold"):
            try:
                self.pipeline.clustering_threshold = threshold
                return "attribute"
            except Exception as exc:
                logger.debug("clustering_threshold 属性覆写失败: %s", exc)

        # 候选 3：嵌套超参 dict 原地覆写（4.x 备选结构）
        for attr in ("hparams", "_hparams"):
            hparams = getattr(self.pipeline, attr, None)
            if not isinstance(hparams, dict):
                continue
            clustering = hparams.get("clustering")
            if isinstance(clustering, dict) and "threshold" in clustering:
                try:
                    clustering["threshold"] = threshold
                    return "hparams"
                except Exception as exc:
                    logger.debug("%s.clustering.threshold 覆写失败: %s", attr, exc)

        return None

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
