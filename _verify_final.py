# -*- coding: utf-8 -*-
"""最终验证：CAM++ 注入全链路 + 聚类阈值覆写超参保留断言（容器内执行）。"""
import os
import sys

import torch

from qwen_asr.inference.qwen3_speaker_diarizer import SpeakerDiarizer

PYANNOTE_DIR = "/models/pyannote"
CAMPLUS_DIR = "/models/campplus"

# 1. 加载 + CAM++ 组件替换注入
print("== [1] SpeakerDiarizer.from_pretrained(embedding='campplus')")
diarizer = SpeakerDiarizer.from_pretrained(
    PYANNOTE_DIR,
    device=None,
    embedding="campplus",
    embedding_model=CAMPLUS_DIR,
)
assert diarizer._embedding_override is not None, "embedding_override 引用缺失"
assert diarizer._embedding_override.dimension == 192, "CAM++ 维度应为 192"
# 注入后管线侧组件确认为 CAM++ wrapper、聚类为 AHC
assert diarizer.pipeline._embedding is diarizer._embedding_override, \
    "pipeline._embedding 未替换为 CAM++"
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
assert isinstance(diarizer.pipeline.clustering, AgglomerativeClustering), \
    "pipeline.clustering 未替换为 AHC"

# 2. 记录注入时的 AHC 超参（method/threshold/min_cluster_size 实例化值）
ahc = diarizer.pipeline.clustering
before = dict(ahc._instantiated)
print(f"== [2] 注入后 AHC 实例化超参: {before}")
assert before.get("method") == "centroid", f"AHC method 应为 centroid: {before}"
assert "min_cluster_size" in before, f"AHC 缺 min_cluster_size: {before}"

# 3. 聚类阈值覆写（extensions.py 启动期路径）——关键断言：仅覆写 threshold，
#    method/min_cluster_size 不得被 instantiate 清空/重置（逐参数 setattr 语义）
mechanism = diarizer.apply_clustering_threshold(0.4)
print(f"== [3] apply_clustering_threshold(0.4) -> 生效机制: {mechanism}")
assert mechanism is not None, "聚类阈值覆写全部机制不可用"
after = dict(ahc._instantiated)
print(f"== [3] 覆写后 AHC 实例化超参: {after}")
assert after.get("threshold") == 0.4, f"threshold 未覆写: {after}"
assert after.get("method") == before["method"], f"method 被清空/重置: {after}"
assert after.get("min_cluster_size") == before["min_cluster_size"], \
    f"min_cluster_size 被清空/重置: {after}"

# 4. 前向冒烟：示例音频交替拼接模拟双人对话，min=max=2 约束下检出 2 人
import numpy as np
import soundfile as sf


def _wav(name):
    data, sr = sf.read(os.path.join(CAMPLUS_DIR, "examples", name), dtype="float32")
    assert sr == 16000
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data)


spk1 = _wav("speaker1_a_cn_16k.wav")
spk2 = _wav("speaker2_a_cn_16k.wav")
mix = torch.cat([spk1, spk1, spk2, spk2, spk1, spk2])
print(f"== [4] 前向冒烟（拼接 {mix.numel() / 16000:.1f}s 双人交替音频, min=max=2）")
results = diarizer.diarize((mix.numpy(), 16000), min_speakers=2, max_speakers=2)
res = results[0]
print(f"== [4] 结果：{len(res.segments)} 段，说话人 {res.speakers}")
assert len(res.speakers) == 2, f"双人约束下应检出 2 个说话人，实得 {res.speakers}"
assert res.segments, "前向结果为空"
# 段时间戳单调升序
starts = [s.start_time for s in res.segments]
assert starts == sorted(starts), "segments 未按 start 升序"

print("== 全部验证通过 ==")
