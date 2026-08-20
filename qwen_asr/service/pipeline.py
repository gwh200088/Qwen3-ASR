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
Segment 转写管道纯逻辑（零 GPU 依赖，仅标准库）。

职责（对应 spec「Segment 切分与说话人归属」）：

- 语言名 ↔ BCP-47 风格码双向映射（30 项，逐项照抄 spec 表格）；
- 对齐 token 序列 → 句级 segment 切分（时间间隙阈值 / 段长上限强制切分）；
- 段文本游标匹配：从完整 ASR 文本截取（保留标点与空格），失败回退拼接；
- 说话人归属：segment 与 diarization 片段的时间重叠计算（dominant + speakers 列表）；
- speakerSummary 汇总（覆盖全部识别说话人，含零值项）。

本模块供 middleware 调用，也可被 example ``--self-test`` 离线自测，
不 import torch / pyannote / vLLM 等任何重依赖。
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "LANGUAGE_NAME_TO_CODE",
    "LANGUAGE_CODE_TO_NAME",
    "resolve_language_name",
    "language_name_to_code",
    "DiarizationTurn",
    "build_segment_response",
    "self_test",
]


# ---------------------------------------------------------------------------
# 语言码映射（spec「语言码映射」完整表，30 项）
# ---------------------------------------------------------------------------

LANGUAGE_NAME_TO_CODE: Dict[str, str] = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "fil",
    "Persian": "fa",
    "Greek": "el",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Macedonian": "mk",
}

#: 反向映射：码（小写）→ 内部语言名；由正向表推导，避免两份手工维护
LANGUAGE_CODE_TO_NAME: Dict[str, str] = {code: name for name, code in LANGUAGE_NAME_TO_CODE.items()}


def resolve_language_name(user_input: str) -> str:
    """解析请求入参 ``language`` 为内部语言名。

    匹配顺序（spec「语言码映射」：入参同时接受 ISO 码与语言名）：

    1. 按内部名匹配，大小写归一（内部名均为单个首字母大写单词，
       复用简单 title 处理，如 ``"chinese"`` → ``"Chinese"``）；
    2. 按码匹配（码统一小写后查反向表，如 ``"YUE"`` → ``"Cantonese"``）；
    3. 均失败 → 抛 ``ValueError``（中文消息列出可选值概要）。
    """
    text = str(user_input).strip()
    titled = text.title()
    if titled in LANGUAGE_NAME_TO_CODE:
        return titled
    lowered = text.lower()
    if lowered in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[lowered]
    preview = "、".join(f"{name}({code})" for name, code in list(LANGUAGE_NAME_TO_CODE.items())[:6])
    raise ValueError(
        f"不支持的语言: {user_input!r}。支持 {len(LANGUAGE_NAME_TO_CODE)} 种语言，"
        f"可传语言名（如 Chinese）或语言码（如 zh），可选值: {preview} 等，"
        f"完整列表见 qwen_asr.service.pipeline.LANGUAGE_NAME_TO_CODE。"
    )


def language_name_to_code(name: str) -> str:
    """内部语言名 → 响应输出码；未匹配回退小写输入（spec：未匹配项回退小写全名）。

    支持跨块合并出的逗号分隔多语言名（如 "Chinese,English" → "zh,en"），与标准模式
    verbose_json 的多语言码输出保持一致。
    """
    text = str(name).strip()
    if "," in text:
        return ",".join(language_name_to_code(part) for part in text.split(",") if part.strip())
    if text in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[text]
    return text.lower()


# ---------------------------------------------------------------------------
# diarization 片段归一
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DiarizationTurn:
    """说话人片段（轻量）：说话人标签与起止时间（秒）。"""

    speaker: str
    start_time: float
    end_time: float


def _to_turns(diarization: List[Any]) -> List[DiarizationTurn]:
    """将多种 diarization 输入形态归一为 ``DiarizationTurn`` 列表。

    接受（可混用）：

    - 含 ``.speaker`` / ``.start_time`` / ``.end_time`` 属性的对象
      （``DiarizationSegment`` 鸭子类型，不 import 上游模块）；
    - ``(speaker, start, end)`` 三元组；
    - 含 ``speaker`` / ``start_time`` / ``end_time`` 键的 dict。
    """
    turns: List[DiarizationTurn] = []
    for entry in diarization:
        if isinstance(entry, dict):
            speaker = entry["speaker"]
            start_time = entry["start_time"]
            end_time = entry["end_time"]
        elif isinstance(entry, tuple):
            speaker, start_time, end_time = entry
        else:
            speaker = entry.speaker
            start_time = entry.start_time
            end_time = entry.end_time
        turns.append(DiarizationTurn(speaker=str(speaker), start_time=float(start_time), end_time=float(end_time)))
    return turns


# ---------------------------------------------------------------------------
# segment 构建
# ---------------------------------------------------------------------------

#: 说话人列入段 speakers 列表的最小重叠时长（秒，spec：重叠 >= 0.1s 全部列入）
_MIN_SPEAKER_OVERLAP = 0.1


def _split_groups(
    items: List[Tuple[str, float, float]],
    segment_gap_threshold: float,
    max_segment_seconds: float,
) -> List[List[Tuple[str, float, float]]]:
    """按时间间隙 / 段长上限切分对齐 item 序列。

    新 item 触发切分的条件（spec「Segment 切分与说话人归属」）：

    - ``item.start_time - 当前段末 item.end_time >= segment_gap_threshold``；
    - 或 ``item.end_time - 当前段首 item.start_time > max_segment_seconds``（段长强切）。
    """
    groups: List[List[Tuple[str, float, float]]] = []
    current: List[Tuple[str, float, float]] = []
    for item in items:
        if current:
            gap = item[1] - current[-1][2]
            span = item[2] - current[0][1]
            if gap >= segment_gap_threshold or span > max_segment_seconds:
                groups.append(current)
                current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def _extract_segment_text(
    items: List[Tuple[str, float, float]],
    full_text: str,
    cursor: int,
) -> Tuple[str, int]:
    """游标匹配从 ``full_text`` 截取段文本，返回 ``(段文本, 新游标)``。

    - 维护游标（单调不回退）：对段内每个 item 的文本从上次匹配末位置起
      ``find``；全部命中则段文本 = ``full_text[首匹配起点:末匹配终点]``，
      保留期间的标点与空格；
    - 任一 item 找不到 → 回退拼接：item 文本全 ASCII 用 ``" ".join``，
      否则 ``"".join``（中文等无空格语言）；此时游标保持不变。
    """
    pos = cursor
    first = -1
    last_end = -1
    for text, _, _ in items:
        idx = full_text.find(text, pos)
        if idx < 0:
            break
        if first < 0:
            first = idx
        last_end = idx + len(text)
        pos = last_end
    else:
        return full_text[first:last_end], last_end
    texts = [text for text, _, _ in items]
    joined = " ".join(texts) if all(t.isascii() for t in texts) else "".join(texts)
    return joined, cursor


def build_segment_response(
    align_items: List[Any],
    diarization: List[Any],
    full_text: str,
    language_name: str,
    duration: float,
    process_time: Optional[float] = None,
    segment_gap_threshold: float = 0.8,
    max_segment_seconds: float = 30.0,
) -> dict:
    """构建 segment 模式响应 dict（纯函数，无副作用）。

    :param align_items: 对齐 item 列表（鸭子类型 ``.text`` / ``.start_time`` /
        ``.end_time``），可为空；
    :param diarization: diarization 片段（对象 / 三元组 / dict，见 ``_to_turns``）；
    :param full_text: 完整 ASR 文本（段文本从中游标截取）；
    :param language_name: 内部语言名（经 ``language_name_to_code`` 输出码）；
    :param duration: 音频时长（秒）；
    :param process_time: 服务端总耗时（秒），``None`` 则响应中为 ``null``；
    :param segment_gap_threshold: 相邻 item 时间间隙切分阈值（秒，含）；
    :param max_segment_seconds: 段长上限（秒，超过强切）。
    """
    turns = _to_turns(diarization)
    items = [
        (str(it.text), float(it.start_time), float(it.end_time))
        for it in align_items
    ]

    cursor = 0
    segments: List[dict] = []
    # (speaker, 段时长) 原始值序列，供 speakerSummary 统计（避免二次遍历取整误差）
    dominant_records: List[Tuple[Optional[str], float]] = []
    for group in _split_groups(items, segment_gap_threshold, max_segment_seconds):
        seg_start = group[0][1]
        seg_end = group[-1][2]
        text, cursor = _extract_segment_text(group, full_text, cursor)
        # 说话人归属：segment [s, e] 与各说话人片段的重叠总时长 Σ max(0, min(e,te)-max(s,ts))
        overlap: Dict[str, float] = {}
        for turn in turns:
            ov = min(seg_end, turn.end_time) - max(seg_start, turn.start_time)
            if ov > 0:
                overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + ov
        # 重叠降序（并列按说话人 id 升序，保证确定性）：首者为 dominant
        ranked = sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0]))
        speaker = ranked[0][0] if ranked else None
        speakers = [sp for sp, ov in ranked if ov >= _MIN_SPEAKER_OVERLAP]
        segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": text,
            "speaker": speaker,
            "speakers": speakers,
        })
        dominant_records.append((speaker, seg_end - seg_start))

    # speakerSummary：覆盖 diarization 识别的全部说话人（从未 dominant 者为零值项）
    all_speakers = sorted({turn.speaker for turn in turns})
    totals = {sp: 0.0 for sp in all_speakers}
    counts = {sp: 0 for sp in all_speakers}
    for speaker, seg_duration in dominant_records:
        if speaker in totals:
            totals[speaker] += seg_duration
            counts[speaker] += 1
    summary_speakers = [
        {
            "id": sp,
            "totalDuration": round(totals[sp], 3),
            "segmentCount": counts[sp],
        }
        for sp in sorted(all_speakers, key=lambda sp: (-totals[sp], sp))
    ]

    return {
        "language": language_name_to_code(language_name),
        "duration": round(float(duration), 3),
        "text": full_text,
        "processTime": round(float(process_time), 3) if process_time is not None else None,
        "segments": segments,
        "speakerSummary": {
            "speakerCount": len(summary_speakers),
            "speakers": summary_speakers,
        },
    }


# ---------------------------------------------------------------------------
# 离线自测（example --self-test 复用；不依赖 GPU / 网络 / 第三方库）
# ---------------------------------------------------------------------------


def self_test() -> None:
    """内置断言自测：任一断言失败即抛异常，全部通过打印 ``pipeline self_test ok``。"""
    from types import SimpleNamespace

    def ali(text: str, start: float, end: float) -> Any:
        """构造鸭子类型对齐 item（模拟 DiarizationSegment/对齐结果，不 import 上游）。"""
        return SimpleNamespace(text=text, start_time=start, end_time=end)

    # ---- 1. 语言映射：30 项双向 + resolve 大小写/码/未知抛错 ----------------
    expected = {  # 逐项照抄 spec「语言码映射」表
        "Chinese": "zh", "English": "en", "Cantonese": "yue", "Arabic": "ar",
        "German": "de", "French": "fr", "Spanish": "es", "Portuguese": "pt",
        "Indonesian": "id", "Italian": "it", "Korean": "ko", "Russian": "ru",
        "Thai": "th", "Vietnamese": "vi", "Japanese": "ja", "Turkish": "tr",
        "Hindi": "hi", "Malay": "ms", "Dutch": "nl", "Swedish": "sv",
        "Danish": "da", "Finnish": "fi", "Polish": "pl", "Czech": "cs",
        "Filipino": "fil", "Persian": "fa", "Greek": "el", "Romanian": "ro",
        "Hungarian": "hu", "Macedonian": "mk",
    }
    assert len(expected) == 30
    assert LANGUAGE_NAME_TO_CODE == expected
    assert LANGUAGE_CODE_TO_NAME == {code: name for name, code in expected.items()}
    assert all(code == code.lower() for code in LANGUAGE_CODE_TO_NAME)  # 码小写
    for name, code in expected.items():  # 双向 + 名大小写归一 + 码匹配
        assert resolve_language_name(name) == name
        assert resolve_language_name(name.lower()) == name
        assert resolve_language_name(code) == name
        assert language_name_to_code(name) == code
    assert resolve_language_name("CHINESE") == "Chinese"
    assert resolve_language_name(" chinese ") == "Chinese"  # 去首尾空白
    assert resolve_language_name("yue") == "Cantonese"
    try:
        resolve_language_name("Klingon")
    except ValueError as exc:
        assert "不支持的语言" in str(exc)  # 中文错误消息含可选值概要
    else:
        raise AssertionError("未知语言应抛 ValueError")
    assert language_name_to_code("Klingon") == "klingon"  # 未匹配回退小写
    assert language_name_to_code("zh") == "zh"  # 已是码则原样（小写）

    # ---- 2. gap 0.8s 切分（含）与 <0.8s 不切 -------------------------------
    # 注：切分边界测试统一用二进制可精确表示的时间值（如 1.0/0.75），
    # 避免 2.8-2.0=0.7999... 这类浮点误差干扰语义验证
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世界", 1.0, 2.0), ali("欢迎", 3.0, 4.0), ali("光临", 4.0, 4.5)],
        diarization=[],
        full_text="你好，世界。欢迎光临。",
        language_name="Chinese",
        duration=4.5,
    )
    # 第 3 个 item 与段末间隙 3.0-2.0=1.0 >= 0.8 → 切分
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 2.0), (3.0, 4.5)]
    # 游标匹配保留标点：段文本 = full_text[0:5] / full_text[6:10]（中文无空格场景）
    assert resp["segments"][0]["text"] == "你好，世界"
    assert resp["segments"][1]["text"] == "欢迎光临"
    assert resp["speakerSummary"] == {"speakerCount": 0, "speakers": []}  # 空 diarization
    assert resp["segments"][0]["speaker"] is None and resp["segments"][0]["speakers"] == []

    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世", 1.75, 2.0)],  # 间隙 0.75 < 0.8
        diarization=[],
        full_text="你好世",
        language_name="Chinese",
        duration=2.0,
    )
    assert resp["segments"] == [
        {"start": 0.0, "end": 2.0, "text": "你好世", "speaker": None, "speakers": []},
    ]

    # 边界语义：间隙恰好等于阈值（1.0）→ 含边界（>=）切分；阈值调大则不切
    boundary_items = [ali("a", 0.0, 1.0), ali("b", 2.0, 3.0)]  # 间隙 1.0
    resp = build_segment_response(boundary_items, [], "a b", "English", 3.0, segment_gap_threshold=1.0)
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (2.0, 3.0)]
    resp = build_segment_response(boundary_items, [], "a b", "English", 3.0, segment_gap_threshold=1.25)
    assert len(resp["segments"]) == 1

    # 阈值参数透传：同一间隙 0.6，默认不切、threshold=0.5 切
    gap_items = [ali("a", 0.0, 1.0), ali("b", 1.6, 2.0)]
    assert len(build_segment_response(gap_items, [], "a b", "English", 2.0)["segments"]) == 1
    resp = build_segment_response(gap_items, [], "a b", "English", 2.0, segment_gap_threshold=0.5)
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (1.6, 2.0)]

    # ---- 3. 30s 段长强切 ----------------------------------------------------
    resp = build_segment_response(
        align_items=[ali("a", 0.0, 20.0), ali("b", 20.5, 30.5)],  # 间隙 0.5<0.8，但段长 30.5>30
        diarization=[],
        full_text="a b",
        language_name="English",
        duration=30.5,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 20.0), (20.5, 30.5)]
    # 边界：段长恰好 30.0（不 > 30）不切；英文空格保留
    resp = build_segment_response(
        align_items=[ali("a", 0.0, 20.0), ali("b", 20.5, 30.0)],
        diarization=[],
        full_text="a b",
        language_name="English",
        duration=30.0,
    )
    assert len(resp["segments"]) == 1
    assert resp["segments"][0]["end"] == 30.0
    assert resp["segments"][0]["text"] == "a b"

    # ---- 4. 游标匹配保留标点/空格 与 找不到回退拼接 --------------------------
    full = "Hello, world. Nice to meet you."
    resp = build_segment_response(
        align_items=[ali("Hello", 0.0, 0.5), ali("world", 0.6, 1.0), ali("Nice", 2.0, 2.4), ali("meet", 2.5, 2.8)],
        diarization=[],
        full_text=full,
        language_name="English",
        duration=3.0,
    )
    # world→Nice 间隙 1.0 >= 0.8 → 两段；段内标点/空格/未对齐词（to）均保留
    assert resp["segments"][0]["text"] == "Hello, world"
    assert resp["segments"][1]["text"] == "Nice to meet"

    resp = build_segment_response(
        align_items=[ali("hello", 0.0, 0.5), ali("world", 0.6, 1.0)],
        diarization=[],
        full_text="COMPLETELY DIFFERENT",
        language_name="English",
        duration=1.0,
    )
    assert resp["segments"][0]["text"] == "hello world"  # 全 ASCII → 空格拼接
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 0.5), ali("世界", 0.6, 1.0)],
        diarization=[],
        full_text="完全不同",
        language_name="Chinese",
        duration=1.0,
    )
    assert resp["segments"][0]["text"] == "你好世界"  # 含非 ASCII → 无空格拼接
    resp = build_segment_response(
        align_items=[ali("hello", 0.0, 0.5), ali("世界", 0.6, 1.0)],
        diarization=[],
        full_text="zzz",
        language_name="Chinese",
        duration=1.0,
    )
    assert resp["segments"][0]["text"] == "hello世界"  # 混合 → 无空格拼接

    # ---- 5. 说话人归属（spec 场景）+ 输入归一 --------------------------------
    diar = [
        SimpleNamespace(speaker="SPEAKER_00", start_time=0.0, end_time=1.5),  # 对象（鸭子类型）
        ("SPEAKER_00", 1.5, 2.0),                                            # 元组：同说话人多片段求和 → 2.0s
        ("SPEAKER_01", 2.5, 3.0),                                            # 元组：重叠 0.5s
        {"speaker": "SPEAKER_02", "start_time": 2.95, "end_time": 3.0},      # dict：重叠 0.05s < 0.1 不入 speakers
    ]
    resp = build_segment_response(
        align_items=[ali("测试", 0.0, 3.0), ali("静音", 4.0, 5.0)],  # 间隙 1.0 → 两段
        diarization=diar,
        full_text="测试静音",
        language_name="Chinese",
        duration=5.0,
        process_time=2.71828,
    )
    segs = resp["segments"]
    assert [(s["start"], s["end"]) for s in segs] == [(0.0, 3.0), (4.0, 5.0)]
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts)  # 天然按 start 升序
    # SPEAKER_00 重叠 2.0s vs SPEAKER_01 0.5s → dominant=SPEAKER_00，speakers 按重叠降序
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]  # 0.05s 未达 0.1s 阈值
    # 无重叠段：speaker=None、speakers=[]
    assert segs[1]["speaker"] is None and segs[1]["speakers"] == []
    # 响应外层字段与 3 位小数
    assert resp["language"] == "zh"
    assert resp["duration"] == 5.0
    assert resp["text"] == "测试静音"
    assert resp["processTime"] == 2.718
    # speakerSummary：零值项 + speakerCount==len + 排序（totalDuration 降序，并列 id 升序）
    summary = resp["speakerSummary"]
    assert summary["speakerCount"] == 3 == len(summary["speakers"])
    assert summary["speakers"] == [
        {"id": "SPEAKER_00", "totalDuration": 3.0, "segmentCount": 1},
        {"id": "SPEAKER_01", "totalDuration": 0.0, "segmentCount": 0},
        {"id": "SPEAKER_02", "totalDuration": 0.0, "segmentCount": 0},
    ]
    assert sum(s["totalDuration"] for s in summary["speakers"]) <= resp["duration"]  # Σ ≤ duration

    # 重叠并列 → 重叠降序第一个（并列按 id 升序，取 SPEAKER_00）
    resp = build_segment_response(
        align_items=[ali("x", 0.0, 1.0)],
        diarization=[("SPEAKER_01", 0.0, 1.0), ("SPEAKER_00", 0.0, 1.0)],
        full_text="x",
        language_name="Chinese",
        duration=1.0,
    )
    assert resp["segments"][0]["speaker"] == "SPEAKER_00"
    assert resp["segments"][0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]

    # ---- 6. 空 align_items：segments=[]，speakerSummary 仍全量 ----------------
    resp = build_segment_response(
        align_items=[],
        diarization=[("SPEAKER_00", 0.0, 1.0)],
        full_text="无对齐结果",
        language_name="Japanese",
        duration=1.5,
        process_time=0.5,
    )
    assert resp["segments"] == []
    assert resp["speakerSummary"] == {
        "speakerCount": 1,
        "speakers": [{"id": "SPEAKER_00", "totalDuration": 0.0, "segmentCount": 0}],
    }
    assert resp["language"] == "ja"
    assert resp["processTime"] == 0.5

    # ---- 7. _to_turns 归一：对象 / 元组 / dict（含混合）----------------------
    turns = _to_turns([
        DiarizationTurn("SPEAKER_00", 0.0, 1.0),
        ("SPEAKER_01", 1.0, 2.0),
        {"speaker": "SPEAKER_02", "start_time": 2.0, "end_time": 3.0},
        SimpleNamespace(speaker="SPEAKER_03", start_time=3, end_time=4),
    ])
    assert turns == [
        DiarizationTurn("SPEAKER_00", 0.0, 1.0),
        DiarizationTurn("SPEAKER_01", 1.0, 2.0),
        DiarizationTurn("SPEAKER_02", 2.0, 3.0),
        DiarizationTurn("SPEAKER_03", 3.0, 4.0),  # 数值统一 float
    ]
    assert _to_turns([]) == []

    print("pipeline self_test ok")
