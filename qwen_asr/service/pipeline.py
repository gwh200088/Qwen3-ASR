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

职责（对应 spec「Segment 切分与说话人归属」与「词级说话人归属」）：

- 语言名 ↔ BCP-47 风格码双向映射（30 项，逐项照抄 spec 表格）；
- 对齐 token 序列 → 句级 segment 切分（时间间隙阈值 / 段长上限强制切分）；
- 段文本游标匹配：从完整 ASR 文本截取（保留标点与空格），失败回退拼接；
- 说话人归属双模式：
  - ``segment``（段级投票，原有行为零改动）：segment 与 diarization 片段的
    时间重叠计算（dominant + speakers 列表）；
  - ``word``（词级归属，默认）：以 align_items 为词序列，词时间中点投票
    归属说话人，洞（无 turn 覆盖词）插值填充，按说话人变化切分 +
    同人二次聚合（含短插话保护）；
- 对齐失败块的粗粒度兜底段（coarse_chunks，两种归属模式均生效）；
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

#: 短插话保护阈值（秒）：同人两段合并前，间隙内其他 speaker 的 turn 覆盖达到
#: 该时长即认为存在"有 turn 但无词"的真实短插话，放弃合并（spec「短插话保护」）
_SHORT_INTERJECTION_SECONDS = 0.3


# ---------------------------------------------------------------------------
# 词级说话人归属（word 模式；词序列直接用 align_items，仅标准库）
# ---------------------------------------------------------------------------


def _attribute_words(
    items: List[Tuple[str, float, float]],
    turns: List[DiarizationTurn],
) -> List[Optional[str]]:
    """词中点投票归属，返回与 ``items`` 等长的 speaker 列表（洞为 ``None``）。

    - 词时间中点落入且仅落入一个 turn → 该 turn 的 speaker；
    - 中点落入多个重叠 turn（重叠语音区）→ 与**词时间区间**重叠时长最大者，
      并列取 speaker id 字典序最小者（保证确定性）；
    - 中点无 turn 覆盖 → ``None``（洞，diarization 漏检/间隙/ASR 幻觉词，
      由 ``_fill_gaps`` 按时间邻近性插值填充）。
    """
    attributions: List[Optional[str]] = []
    for _, start, end in items:
        mid = (start + end) / 2.0
        covering = [t for t in turns if t.start_time <= mid <= t.end_time]
        if not covering:
            attributions.append(None)
        elif len(covering) == 1:
            attributions.append(covering[0].speaker)
        else:
            best_speaker: Optional[str] = None
            best_key: Optional[Tuple[float, str]] = None
            for turn in covering:
                overlap = min(end, turn.end_time) - max(start, turn.start_time)
                key = (-overlap, turn.speaker)
                if best_key is None or key < best_key:
                    best_key = key
                    best_speaker = turn.speaker
            attributions.append(best_speaker)
    return attributions


def _speaker_at_point(t: float, turns: List[DiarizationTurn]) -> Optional[str]:
    """时刻 ``t`` 落入的 turn 的 speaker；多个重叠 turn 取 id 字典序最小（确定性）。"""
    speakers = [turn.speaker for turn in turns if turn.start_time <= t <= turn.end_time]
    if not speakers:
        return None
    return min(speakers)


def _fill_gaps(
    items: List[Tuple[str, float, float]],
    attributions: List[Optional[str]],
    turns: List[DiarizationTurn],
) -> List[Optional[str]]:
    """洞填充四规则（洞 = 词时间中点无 turn 覆盖的词；词本身有时间戳）。

    - 句中洞且前后已归属词同 speaker → 继承该 speaker（覆盖绝大多数情形）；
    - 句中洞且前后异 speaker（洞跨换人点）→ 取 ``[前词.end, 后词.start]``
      线性插值中点，落入谁的 turn 归谁；中点仍无 turn 覆盖归前词 speaker
      （保守继承）；
    - 开头洞（无前邻居）→ 后向继承首个已归属词；
    - 结尾洞（无后邻居）→ 前向继承末个已归属词；
    - 全序列无任何已归属词 → 全部保持 ``None``（不抛异常）。
    """
    filled = list(attributions)
    if not any(a is not None for a in filled):
        return filled
    total = len(filled)
    # 邻居查找基于原始 attributions（避免左侧洞被就地填充后污染邻居语义）
    for i in range(total):
        if filled[i] is not None:
            continue
        prev_i = i - 1
        while prev_i >= 0 and attributions[prev_i] is None:
            prev_i -= 1
        next_i = i + 1
        while next_i < total and attributions[next_i] is None:
            next_i += 1
        prev_speaker = attributions[prev_i] if prev_i >= 0 else None
        next_speaker = attributions[next_i] if next_i < total else None
        if prev_speaker is None:
            filled[i] = next_speaker  # 开头洞：后向继承首个已归属词
        elif next_speaker is None:
            filled[i] = prev_speaker  # 结尾洞：前向继承末个已归属词
        elif prev_speaker == next_speaker:
            filled[i] = prev_speaker  # 前后同人：继承
        else:
            # 前后异人：洞跨换人点，中点插值判定归属
            t1 = items[prev_i][2]  # 前词 end
            t2 = items[next_i][1]  # 后词 start
            if t1 < t2:
                speaker = _speaker_at_point((t1 + t2) / 2.0, turns)
                filled[i] = speaker if speaker is not None else prev_speaker
            else:
                filled[i] = prev_speaker  # 时间戳交叠等退化情形：保守继承前词
    return filled


def _split_by_speaker(
    pairs: List[Tuple[str, float, float, Optional[str]]],
    segment_gap_threshold: float,
    max_segment_seconds: float,
) -> List[List[Tuple[str, float, float, Optional[str]]]]:
    """word 模式切分前两步：间隙/段长切分 + 词归属 speaker 变化处追加切分。

    输入 ``pairs`` 为 ``(text, start, end, speaker)``；``_split_groups`` 仅按
    索引 ``[1]``/``[2]`` 取起止时间，故可直接复用于 4 元组。全 ``None``
    归属序列无 speaker 变化 → 仅按间隙/段长切分（spec「序列内无任何已归属词」）。
    """
    groups: List[List[Tuple[str, float, float, Optional[str]]]] = []
    for group in _split_groups(pairs, segment_gap_threshold, max_segment_seconds):
        current: List[Tuple[str, float, float, Optional[str]]] = []
        current_speaker: Optional[str] = None
        for pair in group:
            if current and pair[3] != current_speaker:
                groups.append(current)
                current = []
            current.append(pair)
            current_speaker = pair[3]
        if current:
            groups.append(current)
    return groups


def _has_other_speaker_turn(
    turns: List[DiarizationTurn],
    gap_start: float,
    gap_end: float,
    speaker: Optional[str],
) -> bool:
    """间隙区间 ``[gap_start, gap_end]`` 内是否存在其他 speaker 的 turn 覆盖 ≥ 0.3s。

    短插话保护判据：保护"间隙里有 turn 但无词"的真实短插话（B 插话未被
    ASR 转写时，间隙内只有 B 的 turn 而无 B 的词；若把两段 A 合并，B 的
    插话事件就无声消失了）。
    """
    if gap_end <= gap_start:
        return False
    coverage: Dict[str, float] = {}
    for turn in turns:
        if turn.speaker == speaker:
            continue
        overlap = min(gap_end, turn.end_time) - max(gap_start, turn.start_time)
        if overlap > 0:
            coverage[turn.speaker] = coverage.get(turn.speaker, 0.0) + overlap
    return any(value >= _SHORT_INTERJECTION_SECONDS for value in coverage.values())


def _gap_blocked(
    gap_start: float,
    gap_end: float,
    blocked: Optional[List[Tuple[float, float]]],
) -> bool:
    """间隙区间是否与任一粗粒度兜底块区间相交（相交则同人两段不合并）。"""
    if not blocked:
        return False
    return any(cs <= gap_end and ce >= gap_start for cs, ce in blocked)


def _merge_same_speaker(
    groups: List[List[Tuple[str, float, float, Optional[str]]]],
    turns: List[DiarizationTurn],
    speaker_merge_gap: float,
    max_segment_seconds: float,
    blocked: Optional[List[Tuple[float, float]]] = None,
) -> List[List[Tuple[str, float, float, Optional[str]]]]:
    """同人二次聚合：同 speaker 相邻段且间隙 < ``speaker_merge_gap`` 合并。

    - ``speaker_merge_gap <= 0``（含 0，即"不合并"）直接原样返回；
    - 短插话保护：间隙区间内存在其他 speaker turn 覆盖 ≥ 0.3s 不合并；
    - 粗段阻断（spec「粗段不参与同人二次聚合」）：间隙区间与任一对齐失败
      块区间（``blocked``）相交不合并——否则跨失败块合并出的正常段会与
      粗段在输出中相互重叠；
    - 合并后段长（末词 end - 首词 start）超过 ``max_segment_seconds`` 不合并；
    - ``speaker=None`` 的段不参与聚合（全 None 序列仅按间隙/段长切分）。
    """
    if speaker_merge_gap is None or speaker_merge_gap <= 0:
        return groups
    merged: List[List[Tuple[str, float, float, Optional[str]]]] = []
    for group in groups:
        if merged:
            prev = merged[-1]
            speaker = group[0][3]
            gap = group[0][1] - prev[-1][2]
            if (
                speaker is not None
                and speaker == prev[0][3]
                and gap < speaker_merge_gap
                and (group[-1][2] - prev[0][1]) <= max_segment_seconds
                and not _has_other_speaker_turn(turns, prev[-1][2], group[0][1], speaker)
                and not _gap_blocked(prev[-1][2], group[0][1], blocked)
            ):
                merged[-1] = prev + group
                continue
        merged.append(group)
    return merged


def _word_vote(group: List[Tuple[str, float, float, Optional[str]]]) -> Tuple[Optional[str], List[str]]:
    """word 模式段内词归属统计。

    - ``speaker``：段内词归属词数最多者，并列取 id 字典序最小（段按词归属
      切分后段内理论全同人，此规则为洞插值边界情形的确定性兜底）；
    - ``speakers``：段内词归属出现过的 speaker 去重集合（按词数降序、id 升序）；
      全 ``None`` 段为 ``(None, [])``。
    """
    counts: Dict[str, int] = {}
    for pair in group:
        if pair[3] is not None:
            counts[pair[3]] = counts.get(pair[3], 0) + 1
    if not counts:
        return None, []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0], [sp for sp, _ in ranked]


def _coarse_vote(turns: List[DiarizationTurn], start: float, end: float) -> Tuple[Optional[str], List[str]]:
    """粗粒度兜底段的块区间说话人投票（与段级投票同公式）。

    重叠降序（并列按 id 升序）首者为 dominant；重叠 ≥ 0.1s 者入 speakers。
    """
    overlap: Dict[str, float] = {}
    for turn in turns:
        ov = min(end, turn.end_time) - max(start, turn.start_time)
        if ov > 0:
            overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + ov
    ranked = sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0]))
    speaker = ranked[0][0] if ranked else None
    speakers = [sp for sp, ov in ranked if ov >= _MIN_SPEAKER_OVERLAP]
    return speaker, speakers


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
    speaker_attribution: str = "word",
    speaker_merge_gap: float = 2.0,
    coarse_chunks: Optional[List[Tuple[str, float, float]]] = None,
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
    :param max_segment_seconds: 段长上限（秒，超过强切）；
    :param speaker_attribution: 说话人归属模式——``word``（默认，词级归属：
        词中点投票 + 洞填充 + 说话人变化切分 + 同人二次聚合）或 ``segment``
        （段级重叠投票，原有行为代码路径零改动）；
    :param speaker_merge_gap: word 模式同人相邻段合并阈值（秒，默认 2.0；
        ``<= 0`` 表示不合并；仅 word 模式生效）；
    :param coarse_chunks: 对齐失败块的粗粒度兜底 ``(text, start, end)`` 列表
        （两种归属模式均生效）：每块产出一个块级粗段（块区间投票 + 块 ASR
        原文），与正常段混合产出时 ``segments[]`` 按 ``start`` 全局升序，
        粗段不参与同人二次聚合。
    """
    turns = _to_turns(diarization)
    items = [
        (str(it.text), float(it.start_time), float(it.end_time))
        for it in align_items
    ]
    coarse = [
        (str(text), float(start), float(end))
        for text, start, end in (coarse_chunks or [])
    ]

    cursor = 0
    segments: List[dict] = []
    # (speaker, 段时长) 原始值序列，供 speakerSummary 统计（避免二次遍历取整误差）
    dominant_records: List[Tuple[Optional[str], float]] = []

    if speaker_attribution == "word":
        # ---- word 模式：词级归属 → 洞填充 → 切分 → 同人二次聚合 ------------
        attributions = _fill_gaps(items, _attribute_words(items, turns), turns)
        pairs: List[Tuple[str, float, float, Optional[str]]] = [
            (text, start, end, speaker)
            for (text, start, end), speaker in zip(items, attributions)
        ]
        groups = _split_by_speaker(pairs, segment_gap_threshold, max_segment_seconds)
        groups = _merge_same_speaker(
            groups, turns, speaker_merge_gap, max_segment_seconds,
            blocked=[(start, end) for _, start, end in coarse],
        )
        for group in groups:
            seg_start = group[0][1]
            seg_end = group[-1][2]
            text, cursor = _extract_segment_text(
                [(pair[0], pair[1], pair[2]) for pair in group], full_text, cursor
            )
            speaker, speakers = _word_vote(group)
            segments.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": text,
                "speaker": speaker,
                "speakers": speakers,
            })
            dominant_records.append((speaker, seg_end - seg_start))
    else:
        # ---- segment 模式：段级重叠投票（原有行为，代码路径零改动）----------
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

    # ---- 粗粒度兜底段（对齐失败块；两种归属模式均生效）----------------------
    if coarse:
        for coarse_text, coarse_start, coarse_end in coarse:
            speaker, speakers = _coarse_vote(turns, coarse_start, coarse_end)
            segments.append({
                "start": round(coarse_start, 3),
                "end": round(coarse_end, 3),
                "text": coarse_text,
                "speaker": speaker,
                "speakers": speakers,
            })
            dominant_records.append((speaker, coarse_end - coarse_start))
        # 粗段与正常段混合产出：segments[] 按 start 全局升序（稳定排序，
        # 粗段插入其时间区间的正确位置，不允许简单追加导致乱序）
        order = sorted(range(len(segments)), key=lambda i: segments[i]["start"])
        segments = [segments[i] for i in order]
        dominant_records = [dominant_records[i] for i in order]

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
    # 注 2：以下 2~7 组为 segment 模式回归断言（显式 speaker_attribution="segment"，
    # 断言内容与升级前逐字节等价）；word 模式断言见第 8~10 组，容错/兜底见第 11 组
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世界", 1.0, 2.0), ali("欢迎", 3.0, 4.0), ali("光临", 4.0, 4.5)],
        diarization=[],
        full_text="你好，世界。欢迎光临。",
        language_name="Chinese",
        duration=4.5,
        speaker_attribution="segment",
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
        speaker_attribution="segment",
    )
    assert resp["segments"] == [
        {"start": 0.0, "end": 2.0, "text": "你好世", "speaker": None, "speakers": []},
    ]

    # 边界语义：间隙恰好等于阈值（1.0）→ 含边界（>=）切分；阈值调大则不切
    boundary_items = [ali("a", 0.0, 1.0), ali("b", 2.0, 3.0)]  # 间隙 1.0
    resp = build_segment_response(boundary_items, [], "a b", "English", 3.0, segment_gap_threshold=1.0, speaker_attribution="segment")
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (2.0, 3.0)]
    resp = build_segment_response(boundary_items, [], "a b", "English", 3.0, segment_gap_threshold=1.25, speaker_attribution="segment")
    assert len(resp["segments"]) == 1

    # 阈值参数透传：同一间隙 0.6，默认不切、threshold=0.5 切
    gap_items = [ali("a", 0.0, 1.0), ali("b", 1.6, 2.0)]
    assert len(build_segment_response(gap_items, [], "a b", "English", 2.0, speaker_attribution="segment")["segments"]) == 1
    resp = build_segment_response(gap_items, [], "a b", "English", 2.0, segment_gap_threshold=0.5, speaker_attribution="segment")
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (1.6, 2.0)]

    # ---- 3. 30s 段长强切 ----------------------------------------------------
    resp = build_segment_response(
        align_items=[ali("a", 0.0, 20.0), ali("b", 20.5, 30.5)],  # 间隙 0.5<0.8，但段长 30.5>30
        diarization=[],
        full_text="a b",
        language_name="English",
        duration=30.5,
        speaker_attribution="segment",
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 20.0), (20.5, 30.5)]
    # 边界：段长恰好 30.0（不 > 30）不切；英文空格保留
    resp = build_segment_response(
        align_items=[ali("a", 0.0, 20.0), ali("b", 20.5, 30.0)],
        diarization=[],
        full_text="a b",
        language_name="English",
        duration=30.0,
        speaker_attribution="segment",
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
        speaker_attribution="segment",
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
        speaker_attribution="segment",
    )
    assert resp["segments"][0]["text"] == "hello world"  # 全 ASCII → 空格拼接
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 0.5), ali("世界", 0.6, 1.0)],
        diarization=[],
        full_text="完全不同",
        language_name="Chinese",
        duration=1.0,
        speaker_attribution="segment",
    )
    assert resp["segments"][0]["text"] == "你好世界"  # 含非 ASCII → 无空格拼接
    resp = build_segment_response(
        align_items=[ali("hello", 0.0, 0.5), ali("世界", 0.6, 1.0)],
        diarization=[],
        full_text="zzz",
        language_name="Chinese",
        duration=1.0,
        speaker_attribution="segment",
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
        speaker_attribution="segment",
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
        speaker_attribution="segment",
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
        speaker_attribution="segment",
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

    # ---- 8. word 模式：词中点投票（单一 turn / 重叠 turn / 无覆盖洞）---------
    vote_turns = [
        DiarizationTurn("SPEAKER_00", 0.0, 2.0),
        DiarizationTurn("SPEAKER_01", 1.0, 3.0),  # 与 00 在 [1,2] 重叠
    ]
    vote_words = [
        ("a", 0.0, 0.4),  # 中点 0.2 仅入 00
        ("b", 0.8, 1.6),  # 中点 1.2 双覆盖；词区间重叠 00=0.8 > 01=0.6 → 00
        ("c", 2.2, 2.6),  # 中点 2.4 仅入 01
        ("d", 2.8, 3.2),  # 中点 3.0 入 01（端点含）
        ("e", 4.0, 4.4),  # 中点 4.2 无覆盖 → None（洞）
    ]
    assert _attribute_words(vote_words, vote_turns) == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", None,
    ]
    # 重叠区词区间重叠并列 → speaker id 字典序最小（保证确定性）
    assert _attribute_words(
        [("x", 0.0, 2.0)],
        [DiarizationTurn("SPEAKER_01", 0.0, 2.0), DiarizationTurn("SPEAKER_00", 0.0, 2.0)],
    ) == ["SPEAKER_00"]

    # ---- 9. word 模式：洞填充四规则 ----------------------------------------
    hole_items = [("a", 0.0, 1.0), ("b", 1.5, 2.0), ("c", 3.0, 3.5)]
    # 句中洞且前后同人 → 继承
    assert _fill_gaps(hole_items, ["SPEAKER_A", None, "SPEAKER_A"], []) == [
        "SPEAKER_A", "SPEAKER_A", "SPEAKER_A",
    ]
    # 句中洞且前后异人（洞跨换人点）：[前.end=1.0, 后.start=3.0] 中点 2.0
    # 落入 B turn → B；中点无 turn 覆盖 → 保守继承前词 A
    assert _fill_gaps(
        hole_items, ["SPEAKER_A", None, "SPEAKER_B"], [DiarizationTurn("SPEAKER_B", 1.6, 2.4)]
    ) == ["SPEAKER_A", "SPEAKER_B", "SPEAKER_B"]
    assert _fill_gaps(hole_items, ["SPEAKER_A", None, "SPEAKER_B"], []) == [
        "SPEAKER_A", "SPEAKER_A", "SPEAKER_B",
    ]
    # 开头洞后向继承 / 结尾洞前向继承
    assert _fill_gaps(hole_items, [None, "SPEAKER_A", None], []) == [
        "SPEAKER_A", "SPEAKER_A", "SPEAKER_A",
    ]
    # 全 None 序列：保持全 None（切分仅按间隙/段长，段 speaker=null）
    assert _fill_gaps(hole_items, [None, None, None], []) == [None, None, None]

    # ---- 10. word 模式：快速交锋切分 / 同人聚合 / 短插话保护 -----------------
    # 快速交锋（核心价值）：A 说至 5.0s，B 于 5.2s 接话（换人间隙 0.2s < 0.8s
    # 切分阈值）→ 按说话人变化切为两段（现状会单段整段归 A）；A 自身
    # 1.5s 停顿（≥0.8s 触发间隙切分）经同人聚合并回一段
    resp = build_segment_response(
        align_items=[
            ali("你好", 0.0, 1.0), ali("谢谢", 2.5, 5.0),  # SPEAKER_00（间隙 1.5 < 2.0 聚合）
            ali("没事", 5.2, 6.0), ali("好的", 6.5, 7.0),    # SPEAKER_01
        ],
        diarization=[("SPEAKER_00", 0.0, 5.0), ("SPEAKER_01", 5.1, 7.0)],
        full_text="你好，谢谢。没事，好的。",
        language_name="Chinese",
        duration=7.0,
    )
    segs = resp["segments"]
    assert [(s["start"], s["end"], s["speaker"]) for s in segs] == [
        (0.0, 5.0, "SPEAKER_00"),  # 段边界 = 首词 start / 末词 end
        (5.2, 7.0, "SPEAKER_01"),
    ]
    assert segs[0]["text"] == "你好，谢谢" and segs[1]["text"] == "没事，好的"
    assert segs[0]["speakers"] == ["SPEAKER_00"]  # word 模式 speakers=词归属去重集合
    # speakerSummary 按 dominant 段统计（word 模式 totalDuration 精度到词粒度）
    assert resp["speakerSummary"]["speakers"] == [
        {"id": "SPEAKER_00", "totalDuration": 5.0, "segmentCount": 1},
        {"id": "SPEAKER_01", "totalDuration": 1.8, "segmentCount": 1},
    ]

    # 同人自然停顿不分段：停顿 1.2s（≥0.8s 间隙切分）后同人继续，merge_gap=2.0
    # 且停顿区间无其他 speaker turn → 二次聚合为一段（现状会错误地切成两段）
    pause_items = [ali("今", 0.0, 0.4), ali("天", 0.4, 0.8), ali("不", 2.0, 2.4), ali("错", 2.4, 2.8)]
    resp = build_segment_response(
        align_items=pause_items,
        diarization=[("SPEAKER_00", 0.0, 3.0)],
        full_text="今天不错",
        language_name="Chinese",
        duration=3.0,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [(0.0, 2.8, "SPEAKER_00")]
    assert resp["segments"][0]["text"] == "今天不错"

    # 短插话保护：A 说 0-0.8s 与 2.0-2.8s，B 于 1.0-1.6s 插话但未被 ASR 转写
    # （间隙内仅存在 B 的 turn 覆盖 0.6s ≥ 0.3s）→ 两 A 段不合并
    resp = build_segment_response(
        align_items=pause_items,
        diarization=[("SPEAKER_00", 0.0, 0.8), ("SPEAKER_01", 1.0, 1.6), ("SPEAKER_00", 2.0, 2.8)],
        full_text="今天不错",
        language_name="Chinese",
        duration=3.0,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [
        (0.0, 0.8, "SPEAKER_00"), (2.0, 2.8, "SPEAKER_00"),
    ]

    # merge_gap=0 不聚合：同人两段保持切分
    resp = build_segment_response(
        align_items=pause_items,
        diarization=[("SPEAKER_00", 0.0, 3.0)],
        full_text="今天不错",
        language_name="Chinese",
        duration=3.0,
        speaker_merge_gap=0.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 0.8), (2.0, 2.8)]

    # 全 None 归属（无 diarization，默认 word 模式）：仅按间隙/段长切分，段 speaker=null
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世界", 3.0, 4.0)],
        diarization=[],
        full_text="你好世界",
        language_name="Chinese",
        duration=4.0,
    )
    assert [(s["start"], s["end"], s["speaker"], s["speakers"]) for s in resp["segments"]] == [
        (0.0, 1.0, None, []), (3.0, 4.0, None, []),
    ]

    # ---- 11. 容错/兜底：coarse_chunks 粗段（两种归属模式均生效）--------------
    # 单块失败：正常词级段 + 粗段混合产出，segments[] 按 start 升序全局排列；
    # 粗段不参与同人二次聚合（同人两段的间隙被粗段区间阻断，不跨粗段合并）
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 1.5, 2.0), ali("丙", 3.0, 3.5), ali("丁", 3.6, 4.0)],
        diarization=[("SPEAKER_00", 0.0, 4.0)],
        full_text="甲乙丙丁",
        language_name="Chinese",
        duration=4.0,
        coarse_chunks=[("失败块文本", 2.1, 2.9)],
    )
    segs = resp["segments"]
    # 词级段 [0,2] 与 [3,4] 同人且间隙 1.0 < 2.0，但间隙与粗段 [2.1,2.9] 相交
    # → 不合并；粗段插入其时间区间的正确位置（start 升序）
    assert [(s["start"], s["end"]) for s in segs] == [(0.0, 2.0), (2.1, 2.9), (3.0, 4.0)]
    assert segs[0]["speaker"] == "SPEAKER_00" and segs[0]["text"] == "甲乙"
    assert segs[2]["speaker"] == "SPEAKER_00" and segs[2]["text"] == "丙丁"
    # 粗段：块区间投票 dominant，文本取块 ASR 原文（不区分标记，与正常段同构）
    assert segs[1]["speaker"] == "SPEAKER_00" and segs[1]["speakers"] == ["SPEAKER_00"]
    assert segs[1]["text"] == "失败块文本"

    # 对齐全空但 diarization 可用：全部块按粗段方式产出（非空 segments）
    resp = build_segment_response(
        align_items=[],
        diarization=[("SPEAKER_00", 0.0, 90.0), ("SPEAKER_01", 90.0, 180.0)],
        full_text="第一块文本第二块文本",
        language_name="Chinese",
        duration=180.0,
        coarse_chunks=[("第一块文本", 0.0, 90.0), ("第二块文本", 90.0, 180.0)],
    )
    assert [(s["start"], s["end"], s["speaker"], s["text"]) for s in resp["segments"]] == [
        (0.0, 90.0, "SPEAKER_00", "第一块文本"),
        (90.0, 180.0, "SPEAKER_01", "第二块文本"),
    ]

    # segment 模式下粗段兜底同样生效（逐块容错与归属模式正交）
    resp = build_segment_response(
        align_items=[ali("x", 0.0, 1.0)],
        diarization=[("SPEAKER_00", 0.0, 1.0), ("SPEAKER_01", 2.0, 3.0)],
        full_text="x",
        language_name="Chinese",
        duration=3.0,
        speaker_attribution="segment",
        coarse_chunks=[("粗段", 1.5, 3.0)],
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [
        (0.0, 1.0, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01"),
    ]

    print("pipeline self_test ok")
