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
- 对齐 token 序列 → 句级 segment 切分（句末标点硬边界恒切分 / 无标点处
  时间间隙阈值 / 段长上限强制切分，spec「标点感知 Segment 切分」）；
- 段文本游标匹配：从完整 ASR 文本截取（保留标点与空格），失败回退拼接；
  句末标点硬边界处切分时，between-span 中的句末标点附前段 ``text`` 末尾
  （含末段尾部追加），客户端拼接 ``segments[].text`` 与 ``text`` 标点无损；
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
    hard_boundaries: Optional[List[bool]] = None,
) -> List[List[Tuple[str, float, float, Optional[str]]]]:
    """word 模式切分前两步：间隙/段长/硬边界切分 + 词归属 speaker 变化处追加切分。

    输入 ``pairs`` 为 ``(text, start, end, speaker)``；``_split_groups`` 仅按
    索引 ``[1]``/``[2]`` 取起止时间，故可直接复用于 4 元组（``hard_boundaries``
    与 4 元组序列等长同序，索引直接对齐）。全 ``None`` 归属序列无 speaker
    变化 → 仅按硬边界/间隙/段长切分（spec「序列内无任何已归属词」）；
    说话人变化切分逻辑不受标点影响（无标点处照切，word 模式核心价值）。
    """
    groups: List[List[Tuple[str, float, float, Optional[str]]]] = []
    for group in _split_groups(
        pairs, segment_gap_threshold, max_segment_seconds, hard_boundaries
    ):
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
    hard_boundaries: Optional[List[bool]] = None,
) -> List[List[Tuple[str, float, float, Optional[str]]]]:
    """同人二次聚合：同 speaker 相邻段且间隙 < ``speaker_merge_gap`` 合并。

    - ``speaker_merge_gap <= 0``（含 0，即"不合并"）直接原样返回；
    - 硬边界阻断（spec「同人聚合不跨越硬边界」）：同人相邻段之间的边界
      为句末标点硬边界时不合并——分组结果连续覆盖全部词，全局边界索引
      按累计词数反查（前段末词全局索引 = 累计词数 - 1）；
      ``hard_boundaries=None`` 时无阻断（兼容 ``punctuation_split=off`` 路径）；
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
    consumed = 0  # 已覆盖词数（分组连续覆盖全部词，反查全局边界索引）
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
                and not (hard_boundaries is not None and hard_boundaries[consumed - 1])
                and not _has_other_speaker_turn(turns, prev[-1][2], group[0][1], speaker)
                and not _gap_blocked(prev[-1][2], group[0][1], blocked)
            ):
                merged[-1] = prev + group
                consumed += len(group)
                continue
        merged.append(group)
        consumed += len(group)
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
    hard_boundaries: Optional[List[bool]] = None,
) -> List[List[Tuple[str, float, float]]]:
    """按句末标点硬边界 / 时间间隙 / 段长上限切分对齐 item 序列。

    新 item 触发切分的条件（spec「标点感知 Segment 切分」）：

    - ``hard_boundaries[i-1]``——前 item 与当前 item 之间为句末标点硬边界，
      无视时间间隙恒切分（``hard_boundaries=None`` 时全按非硬边界，行为
      与纯间隙切分完全一致）；
    - 或 ``item.start_time - 当前段末 item.end_time >= segment_gap_threshold``
      （无句末标点处的间隙切分阈值，默认 2.0）；
    - 或 ``item.end_time - 当前段首 item.start_time > max_segment_seconds``
      （段长强切，不受标点影响）。
    """
    groups: List[List[Tuple[str, float, float]]] = []
    current: List[Tuple[str, float, float]] = []
    for i, item in enumerate(items):
        if current:
            gap = item[1] - current[-1][2]
            span = item[2] - current[0][1]
            if (
                (hard_boundaries is not None and hard_boundaries[i - 1])
                or gap >= segment_gap_threshold
                or span > max_segment_seconds
            ):
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


#: 句末标点集合（spec「句末标点硬切分」）：``。！？；`` + ASCII ``.!?;``
#: （含句点——英文句末标点主力，缺失则英文场景标点切分失效）+ 换行符；
#: 逗号 / 顿号 / 冒号不在内（保留段内，符合"一句话一个分段"）
_SENTENCE_END_CHARS = set("。！？；.!?;\n")


def _sentence_end_boundaries(
    items: List[Tuple[str, float, float]],
    full_text: str,
    coarse_spans: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[List[bool], List[str], bool, int]:
    """计算相邻 item 之间的句末标点硬边界与边界标点序列。

    与 ``_extract_segment_text`` 同一 greedy ``find`` 游标语义把每个 item
    匹配到 ``full_text``；相邻 item 匹配区间之间的 between-span
    （``full_text[前item匹配终点:后item匹配起点]``）：

    - 含任一句末标点 → ``boundaries[i] = True``（硬边界，无视间隙恒切分），
      ``puncts[i]`` 为 between-span 中全部句末标点字符按出现顺序拼接
      （如 ``"。"``, ``"？！"``；空格 / 引号等非句末标点字符不收集）；
    - 不含 → ``boundaries[i] = False``，``puncts[i] = ""``。

    规则：

    - 跨失败块边界 puncts 置空（v3）：边界时间间隙区间 ``[items[i].end,
      items[i+1].start]`` 与任一 coarse 块区间相交（``_gap_blocked`` 同一
      时间域判定）→ ``puncts[i] = ""``，``boundaries[i]`` 保持原判定、切分
      照常——该 between-span 含整块失败文本（最长 180s），标点拼入会给
      前段追加垃圾后缀并与粗段原文标点重复；
    - 任一 item 匹配失败 → 全量回退（全 ``False`` + 全空串）：失败 item
      后游标位置不确定，部分保留已匹配前缀的边界可能使后续 between-span
      错位（标点误判），全量回退语义保守且可预测，不抛异常。

    Returns:
        ``(boundaries, puncts, matched, last_end)``：两个列表长度均为
        ``len(items) - 1``（``len(items) <= 1`` 时为空）；``matched`` 为全局
        匹配成功标志；``last_end`` 为末词匹配终点（匹配失败或空序列时为
        -1），供末段尾部句末标点追加。
    """
    total = len(items)
    pos = 0
    spans: List[Tuple[int, int]] = []  # 各 item 的 (匹配起点, 匹配终点)
    for text, _, _ in items:
        idx = full_text.find(text, pos)
        if idx < 0:
            return [False] * (total - 1), [""] * (total - 1), False, -1
        spans.append((idx, idx + len(text)))
        pos = idx + len(text)
    boundaries: List[bool] = []
    puncts: List[str] = []
    for i in range(total - 1):
        between = full_text[spans[i][1]:spans[i + 1][0]]
        collected = "".join(ch for ch in between if ch in _SENTENCE_END_CHARS)
        boundaries.append(bool(collected))
        if collected and _gap_blocked(items[i][2], items[i + 1][1], coarse_spans):
            collected = ""  # 跨失败块边界：失败块标点仅保留在粗段原文中
        puncts.append(collected)
    return boundaries, puncts, True, spans[-1][1] if spans else -1


def build_segment_response(
    align_items: List[Any],
    diarization: List[Any],
    full_text: str,
    language_name: str,
    duration: float,
    process_time: Optional[float] = None,
    segment_gap_threshold: float = 2.0,
    max_segment_seconds: float = 30.0,
    speaker_attribution: str = "word",
    speaker_merge_gap: float = 2.0,
    coarse_chunks: Optional[List[Tuple[str, float, float]]] = None,
    punctuation_split: bool = True,
) -> dict:
    """构建 segment 模式响应 dict（纯函数，无副作用）。

    :param align_items: 对齐 item 列表（鸭子类型 ``.text`` / ``.start_time`` /
        ``.end_time``），可为空；
    :param diarization: diarization 片段（对象 / 三元组 / dict，见 ``_to_turns``）；
    :param full_text: 完整 ASR 文本（段文本从中游标截取）；
    :param language_name: 内部语言名（经 ``language_name_to_code`` 输出码）；
    :param duration: 音频时长（秒）；
    :param process_time: 服务端总耗时（秒），``None`` 则响应中为 ``null``；
    :param segment_gap_threshold: 相邻 item 之间**无句末标点**时的时间间隙
        切分阈值（秒，含；默认 2.0）——句末标点处无视间隙恒切分；
    :param max_segment_seconds: 段长上限（秒，超过强切）；
    :param speaker_attribution: 说话人归属模式——``word``（默认，词级归属：
        词中点投票 + 洞填充 + 说话人变化切分 + 同人二次聚合）或 ``segment``
        （段级重叠投票，原有行为代码路径零改动）；
    :param speaker_merge_gap: word 模式同人相邻段合并阈值（秒，默认 2.0；
        ``<= 0`` 表示不合并；仅 word 模式生效）；
    :param coarse_chunks: 对齐失败块的粗粒度兜底 ``(text, start, end)`` 列表
        （两种归属模式均生效）：每块产出一个块级粗段（块区间投票 + 块 ASR
        原文），与正常段混合产出时 ``segments[]`` 按 ``start`` 全局升序，
        粗段不参与同人二次聚合；
    :param punctuation_split: 句末标点硬切分开关（默认 True）：True 时相邻
        item 在 ``full_text`` 中匹配区间之间存在句末标点（``。！？；.!?;``
        及换行）的边界恒切分（无视间隙）、切分处句末标点附前段 ``text``
        末尾、末段追加全文尾部句末标点；False 时跳过硬边界计算，纯间隙/
        段长（word 模式含说话人变化）切分，段文本截取行为不变。
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

    # 句末标点硬边界（spec「标点感知 Segment 切分」）：False 时跳过计算
    # （boundaries=None，纯间隙行为）；匹配失败全量回退（全 False，无追加）
    if punctuation_split:
        hard_boundaries, puncts, matched, last_end = _sentence_end_boundaries(
            items, full_text, coarse_spans=[(start, end) for _, start, end in coarse]
        )
    else:
        hard_boundaries, puncts, matched, last_end = None, [], False, -1

    cursor = 0
    segments: List[dict] = []
    # (speaker, 段时长) 原始值序列，供 speakerSummary 统计（避免二次遍历取整误差）
    dominant_records: List[Tuple[Optional[str], float]] = []
    # 已覆盖词数（两种模式的分组均连续覆盖全部词，反查全局边界索引用）
    consumed = 0

    if speaker_attribution == "word":
        # ---- word 模式：词级归属 → 洞填充 → 切分 → 同人二次聚合 ------------
        attributions = _fill_gaps(items, _attribute_words(items, turns), turns)
        pairs: List[Tuple[str, float, float, Optional[str]]] = [
            (text, start, end, speaker)
            for (text, start, end), speaker in zip(items, attributions)
        ]
        groups = _split_by_speaker(
            pairs, segment_gap_threshold, max_segment_seconds, hard_boundaries
        )
        groups = _merge_same_speaker(
            groups, turns, speaker_merge_gap, max_segment_seconds,
            blocked=[(start, end) for _, start, end in coarse],
            hard_boundaries=hard_boundaries,
        )
        for group in groups:
            seg_start = group[0][1]
            seg_end = group[-1][2]
            text, cursor = _extract_segment_text(
                [(pair[0], pair[1], pair[2]) for pair in group], full_text, cursor
            )
            consumed += len(group)
            # 段末边界为句末标点硬边界 → between-span 句末标点附前段末尾
            # （跨失败块边界 puncts 已置空故自然不追加；puncts 只含句末标点）
            if (
                hard_boundaries is not None
                and consumed < len(items)
                and hard_boundaries[consumed - 1]
            ):
                text += puncts[consumed - 1]
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
        for group in _split_groups(
            items, segment_gap_threshold, max_segment_seconds, hard_boundaries
        ):
            seg_start = group[0][1]
            seg_end = group[-1][2]
            text, cursor = _extract_segment_text(group, full_text, cursor)
            consumed += len(group)
            # 段末边界为句末标点硬边界 → between-span 句末标点附前段末尾
            if (
                hard_boundaries is not None
                and consumed < len(items)
                and hard_boundaries[consumed - 1]
            ):
                text += puncts[consumed - 1]
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

    # ---- 末段尾部句末标点追加（v3）------------------------------------------
    # 末词匹配终点之后的句末标点不属于任何 between-span，不追加则全文以
    # 句末标点结尾时末段丢标点（拼接无损失效）；匹配失败回退 /
    # punctuation_split=False 时不追加。粗段不参与（其 text 取块 ASR 原文，
    # 含自身标点），故在粗段混入前对最后产出的正常段追加。
    if matched and last_end >= 0 and segments:
        tail = "".join(ch for ch in full_text[last_end:] if ch in _SENTENCE_END_CHARS)
        if tail:
            segments[-1]["text"] += tail

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

    # ---- 2. 间隙切分与句末标点硬切分（segment 模式）------------------------
    # 注：切分边界测试统一用二进制可精确表示的时间值（如 1.0/0.75），
    # 避免 2.8-2.0=0.7999... 这类浮点误差干扰语义验证
    # 注 2：以下 2~7 组为 segment 模式断言（显式 speaker_attribution="segment"）；
    # 新默认 segment_gap_threshold=2.0 且句末标点硬切分开启，涉及切分/段文本
    # 的期望值已按"句末标点附前段 + 末段尾部追加"更新；word 模式断言见第
    # 8~10 组，容错/兜底见第 11 组，标点切分专项断言见第 12~17 组
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世界", 1.0, 2.0), ali("欢迎", 3.0, 4.0), ali("光临", 4.0, 4.5)],
        diarization=[],
        full_text="你好，世界。欢迎光临。",
        language_name="Chinese",
        duration=4.5,
        speaker_attribution="segment",
    )
    # "世界|欢迎"之间 between-span 为句号 → 硬边界切分（间隙 1.0 < 2.0 不触发
    # 间隙切分，切分由句号驱动）；句号附前段、末段尾部句号追加
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 2.0), (3.0, 4.5)]
    # 游标匹配保留段内标点：段文本 = full_text[0:5]+"。" / full_text[6:10]+"。"（末段尾部）
    assert resp["segments"][0]["text"] == "你好，世界。"
    assert resp["segments"][1]["text"] == "欢迎光临。"
    assert resp["speakerSummary"] == {"speakerCount": 0, "speakers": []}  # 空 diarization
    assert resp["segments"][0]["speaker"] is None and resp["segments"][0]["speakers"] == []

    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世", 1.75, 2.0)],  # 间隙 0.75 < 2.0（默认）不切
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
        align_items=[ali("a", 0.0, 20.0), ali("b", 20.5, 30.5)],  # 间隙 0.5<2.0，但段长 30.5>30
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
    # world→Nice 之间 between-span ". " 含 ASCII 句点 → 硬边界切分（间隙
    # 1.0 < 2.0，切分由句点驱动）；段内标点/空格/未对齐词（to）均保留，
    # 句点附前段、末段尾部句点追加（between 中空格为非句末标点不追加）
    assert resp["segments"][0]["text"] == "Hello, world."
    assert resp["segments"][1]["text"] == "Nice to meet."

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
        align_items=[ali("测试", 0.0, 3.0), ali("静音", 4.0, 5.0)],  # 间隙 1.0，无标点
        diarization=diar,
        full_text="测试静音",
        language_name="Chinese",
        duration=5.0,
        process_time=2.71828,
        speaker_attribution="segment",
        segment_gap_threshold=0.8,  # 显式旧阈值：间隙 1.0 >= 0.8 → 两段（新默认 2.0 下不切）
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
    # 快速交锋（核心价值）：A 说至 5.0s，B 于 5.2s 接话（换人间隙 0.2s < 2.0
    # 切分阈值）→ 按说话人变化切为两段（无标点处照切，不受标点影响）；
    # A 自身 1.5s 停顿无标点 < 2.0 → 不切分（升级前 0.8s 阈值下切分后
    # 靠同人聚合并回一段，断言结果一致）
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
    # "谢谢|没事"之间 between-span 为句号（与说话人变化重合）→ 句号附前段、
    # 末段尾部句号追加
    assert segs[0]["text"] == "你好，谢谢。" and segs[1]["text"] == "没事，好的。"
    assert segs[0]["speakers"] == ["SPEAKER_00"]  # word 模式 speakers=词归属去重集合
    # speakerSummary 按 dominant 段统计（word 模式 totalDuration 精度到词粒度）
    assert resp["speakerSummary"]["speakers"] == [
        {"id": "SPEAKER_00", "totalDuration": 5.0, "segmentCount": 1},
        {"id": "SPEAKER_01", "totalDuration": 1.8, "segmentCount": 1},
    ]

    # 同人自然停顿不分段：停顿 1.2s 无标点 < 2.0 → 根本不切分（升级前
    # 0.8s 阈值切分后靠 merge_gap=2.0 聚合回一段，断言结果一致）
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

    # 短插话保护：A 说 0-0.75s 与 2.75-3.5s（间隙 2.75-0.75=2.0 恰达默认阈值
    # → 切分发生），B 于 1.0-1.5s 插话但未被 ASR 转写（间隙内 B turn 覆盖
    # 0.5s ≥ 0.3s）；merge_gap=3.0 足以合并（间隙 2.0 < 3.0）→ 保护仍阻断合并
    interject_items = [ali("今", 0.0, 0.375), ali("天", 0.375, 0.75), ali("不", 2.75, 3.125), ali("错", 3.125, 3.5)]
    resp = build_segment_response(
        align_items=interject_items,
        diarization=[("SPEAKER_00", 0.0, 0.75), ("SPEAKER_01", 1.0, 1.5), ("SPEAKER_00", 2.75, 3.5)],
        full_text="今天不错",
        language_name="Chinese",
        duration=3.5,
        speaker_merge_gap=3.0,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [
        (0.0, 0.75, "SPEAKER_00"), (2.75, 3.5, "SPEAKER_00"),
    ]
    assert [s["text"] for s in resp["segments"]] == ["今天", "不错"]

    # merge_gap=0 不聚合：显式旧阈值 0.8 下同人 1.2s 停顿切分后不聚合
    resp = build_segment_response(
        align_items=pause_items,
        diarization=[("SPEAKER_00", 0.0, 3.0)],
        full_text="今天不错",
        language_name="Chinese",
        duration=3.0,
        segment_gap_threshold=0.8,
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
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 1.5, 2.0), ali("丙", 4.0, 4.5), ali("丁", 4.6, 5.0)],
        diarization=[("SPEAKER_00", 0.0, 5.0)],
        full_text="甲乙丙丁",
        language_name="Chinese",
        duration=5.0,
        coarse_chunks=[("失败块文本", 2.1, 2.9)],
        speaker_merge_gap=3.0,
    )
    segs = resp["segments"]
    # 词级段 [0,2] 与 [4,5] 同人且间隙 2.0 < merge_gap 3.0（若无阻断会合并），
    # 但间隙与粗段 [2.1,2.9] 相交 → 不合并；粗段插入其时间区间的正确位置
    # （start 升序）
    assert [(s["start"], s["end"]) for s in segs] == [(0.0, 2.0), (2.1, 2.9), (4.0, 5.0)]
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

    # ---- 12. _sentence_end_boundaries：硬边界计算 helper 级 ------------------
    # 句号/问号/叹号/分号/ASCII 句点/换行触发硬边界；逗号/顿号不触发
    helper_items = [
        ("甲", 0.0, 0.5), ("乙", 0.5, 1.0), ("丙", 1.0, 1.5), ("丁", 1.5, 2.0),
        ("戊", 2.0, 2.5), ("己", 2.5, 3.0), ("庚", 3.0, 3.5), ("辛", 3.5, 4.0),
    ]
    boundaries, puncts, matched, last_end = _sentence_end_boundaries(
        helper_items, "甲。乙？丙！丁；戊.己\n庚，辛"
    )
    assert boundaries == [True, True, True, True, True, True, False]
    assert puncts == ["。", "？", "！", "；", ".", "\n", ""]  # 逗号边界：False + 空串
    assert matched is True
    assert last_end == 15  # 末词"辛"匹配终点（full_text 共 15 字符）

    # 连续句末标点"？！"完整收集进 puncts；空格不收集
    boundaries, puncts, matched, last_end = _sentence_end_boundaries(
        [("甲", 0.0, 0.5), ("乙", 0.5, 1.0)], "甲？！ 乙"
    )
    assert boundaries == [True] and puncts == ["？！"]
    assert matched is True and last_end == 5

    # 跨失败块边界 puncts 置空（boundaries 保持原判定）：两词时间间隙
    # [1.0, 4.0] 与 coarse 块区间相交、between-span 含句号
    coarse_pair = [("甲", 0.0, 1.0), ("乙", 4.0, 5.0)]
    boundaries, puncts, _, _ = _sentence_end_boundaries(coarse_pair, "甲。乙", [(1.5, 3.5)])
    assert boundaries == [True] and puncts == [""]  # 置空，切分照常
    # 不相交的 coarse 区间不影响 puncts 收集
    boundaries, puncts, _, _ = _sentence_end_boundaries(coarse_pair, "甲。乙", [(10.0, 11.0)])
    assert boundaries == [True] and puncts == ["。"]

    # 匹配失败回退：某 item 文本不在 full_text → 全 False + 全空串 + matched=False
    boundaries, puncts, matched, last_end = _sentence_end_boundaries(
        [("甲", 0.0, 0.5), ("乙", 0.5, 1.0), ("丙", 1.0, 1.5)], "甲。丙"  # "乙"找不到
    )
    assert boundaries == [False, False] and puncts == ["", ""]
    assert matched is False and last_end == -1

    # 单 item 序列：返回空边界（matched=True、last_end 为末词匹配终点）
    boundaries, puncts, matched, last_end = _sentence_end_boundaries([("甲", 0.0, 0.5)], "甲。")
    assert boundaries == [] and puncts == []
    assert matched is True and last_end == 1

    # ---- 13. 句末标点硬切分：快问快答 / 英文句点 / 逗号不切 ------------------
    # 快问快答（spec Scenario）：两句间隙仅 0.3s（< 阈值）→ 句号处切分两段，
    # 各自独立归属；句号附前段、末段尾部"？"追加（v3，不丢失）
    resp = build_segment_response(
        align_items=[ali("说号就行", 0.0, 1.0), ali("啊", 1.3, 1.6)],
        diarization=[],
        full_text="说号就行。啊？",
        language_name="Chinese",
        duration=1.6,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (1.3, 1.6)]
    assert resp["segments"][0]["text"] == "说号就行。"
    assert resp["segments"][1]["text"] == "啊？"
    assert "".join(s["text"] for s in resp["segments"]) == "说号就行。啊？" == resp["text"]

    # 英文句末句点切分（ASCII `.` 属于句末标点集合）：间隙 0.3s → 两段
    resp = build_segment_response(
        align_items=[
            ali("Nice", 0.0, 0.5), ali("to", 0.5, 0.8), ali("meet", 0.85, 1.2),
            ali("you", 1.2, 1.5), ali("See", 1.8, 2.1), ali("you", 2.1, 2.4),
        ],
        diarization=[],
        full_text="Nice to meet you. See you.",
        language_name="English",
        duration=2.4,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.5), (1.8, 2.4)]
    assert resp["segments"][0]["text"] == "Nice to meet you."  # 句点附前段
    assert resp["segments"][1]["text"] == "See you."          # 末段尾部句点追加

    # 逗号不切分（spec Scenario）：小间隙 0.2s + 仅有逗号 → 不切，逗号保留段内
    resp = build_segment_response(
        align_items=[ali("不记分", 0.0, 1.0), ali("罚款二十", 1.2, 2.0)],
        diarization=[],
        full_text="不记分，罚款二十",
        language_name="Chinese",
        duration=2.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 2.0)]
    assert resp["segments"][0]["text"] == "不记分，罚款二十"

    # ---- 14. 段文本标点附前段 / 跨失败块不追加 / 混合场景拼接无损 -------------
    # "说号就行。啊？说吧"（spec Scenario 段文本标点无损）→ 三段 text 各带
    # 句末标点，"".join(segments[].text) 与 text 字段一致
    resp = build_segment_response(
        align_items=[ali("说号就行", 0.0, 1.0), ali("啊", 1.2, 1.5), ali("说吧", 1.7, 2.0)],
        diarization=[],
        full_text="说号就行。啊？说吧",
        language_name="Chinese",
        duration=2.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (1.2, 1.5), (1.7, 2.0)]
    assert [s["text"] for s in resp["segments"]] == ["说号就行。", "啊？", "说吧"]
    assert "".join(s["text"] for s in resp["segments"]) == resp["text"]

    # 连续句末标点"？！ "（spec Scenario）：前段追加"？！"，空格不追加
    resp = build_segment_response(
        align_items=[ali("啊", 0.0, 0.5), ali("什么", 1.0, 1.5)],
        diarization=[],
        full_text="啊？！ 什么",
        language_name="Chinese",
        duration=1.5,
    )
    assert [s["text"] for s in resp["segments"]] == ["啊？！", "什么"]

    # 跨失败块边界不追加（spec Scenario）：甲|乙 隔失败块（块文本含多个句末
    # 标点）→ 前段无垃圾标点后缀；粗段 text 含自身原文标点、无重复；混合
    # 场景拼接无损：按 start 排序后 "".join(segments[].text) == text
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 4.0, 5.0)],  # 间隙 [1.0, 4.0]
        diarization=[("SPEAKER_00", 0.0, 5.0)],
        full_text="甲失败。块转写！乙。",
        language_name="Chinese",
        duration=5.0,
        coarse_chunks=[("失败。块转写！", 1.5, 3.5)],  # 与间隙相交 → puncts 置空
    )
    segs = resp["segments"]
    assert [(s["start"], s["end"]) for s in segs] == [(0.0, 1.0), (1.5, 3.5), (4.0, 5.0)]
    # between-span"失败。块转写！"跨失败块 → 置空不追加，前段无"。。？。"垃圾后缀
    assert segs[0]["text"] == "甲"
    # 粗段取块 ASR 原文（含自身句末标点，不与任何段重复）
    assert segs[1]["text"] == "失败。块转写！"
    # 末段尾部句号追加
    assert segs[2]["text"] == "乙。"
    assert "".join(s["text"] for s in segs) == resp["text"]

    # ---- 15. 无标点间隙阈值 2.0：用户场景回归与边界值 -------------------------
    # 用户场景回归 1（spec Scenario 句中停顿不再切碎）：无标点 0.875s 停顿
    # （用户 0.88s 场景，取二进制精确值）→ 不切分
    resp = build_segment_response(
        align_items=[ali("想", 0.0, 1.0), ali("负责", 1.875, 2.5)],
        diarization=[],
        full_text="想负责",
        language_name="Chinese",
        duration=2.5,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 2.5)]
    assert resp["segments"][0]["text"] == "想负责"

    # 用户场景回归 2（spec Scenario 单字碎片聚合）：等距单字序列间隙 1.9375s
    # （用户 1.943s 场景，取二进制精确值 < 2.0）且无标点 → 不逐字成段；
    # span 34.625s > 30s 按段长上限强切一次 → 恰两段
    single_items = [
        ali(ch, i * 2.4375, i * 2.4375 + 0.5)
        for i, ch in enumerate("甲乙丙丁戊己庚辛壬癸子丑寅卯辰")
    ]
    resp = build_segment_response(
        align_items=single_items,
        diarization=[],
        full_text="甲乙丙丁戊己庚辛壬癸子丑寅卯辰",
        language_name="Chinese",
        duration=34.625,
    )
    # 注：第二段 start 31.6875 经 round(..., 3) 银行家舍入为 31.688
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 29.75), (31.688, 34.625)]
    assert [s["text"] for s in resp["segments"]] == ["甲乙丙丁戊己庚辛壬癸子丑寅", "卯辰"]

    # 长静音仍切分（spec Scenario）：无标点间隙恰好 2.0（含边界值）→ 切分
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 3.0, 4.0)],
        diarization=[],
        full_text="甲乙",
        language_name="Chinese",
        duration=4.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (3.0, 4.0)]

    # ---- 16. 同人聚合不跨硬边界 / 默认零触发 / punctuation_split 开关 ---------
    # 同人聚合不跨越硬边界（spec Scenario）：同人快速连接（间隙 0.2s），
    # gap 阈值调小 0.5 + merge_gap 调大 5.0（若无硬边界阻断则会合并）→ 仍两段
    fast_items = [ali("今天不错", 0.0, 1.0), ali("明天更好", 1.2, 2.0)]
    resp = build_segment_response(
        align_items=fast_items,
        diarization=[("SPEAKER_00", 0.0, 2.0)],
        full_text="今天不错。明天更好。",
        language_name="Chinese",
        duration=2.0,
        segment_gap_threshold=0.5,
        speaker_merge_gap=5.0,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [
        (0.0, 1.0, "SPEAKER_00"), (1.2, 2.0, "SPEAKER_00"),
    ]
    assert [s["text"] for s in resp["segments"]] == ["今天不错。", "明天更好。"]
    # 对照：关掉标点切分后同参数（间隙 0.2 < 0.5 不切）→ 一段，
    # 证明上面两段确由硬边界切分且被聚合阻断保持
    resp = build_segment_response(
        align_items=fast_items,
        diarization=[("SPEAKER_00", 0.0, 2.0)],
        full_text="今天不错。明天更好。",
        language_name="Chinese",
        duration=2.0,
        segment_gap_threshold=0.5,
        speaker_merge_gap=5.0,
        punctuation_split=False,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [(0.0, 2.0, "SPEAKER_00")]

    # 默认参数下聚合零触发（spec 推导结论）：默认 gap 2.0 / merge_gap 2.0，
    # 同人间隙 2.5s 切分两段后 gap ≥ 2.0 不满足合并条件 gap < 2.0 → 不合并
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 3.5, 4.5)],
        diarization=[("SPEAKER_00", 0.0, 4.5)],
        full_text="甲乙",
        language_name="Chinese",
        duration=4.5,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [
        (0.0, 1.0, "SPEAKER_00"), (3.5, 4.5, "SPEAKER_00"),
    ]

    # punctuation_split=False（spec Scenario 线上误切即时关闭）：有标点小间隙
    # → 跳过硬边界计算不切分；句号随合并自然保留段内，末段尾部不追加
    resp = build_segment_response(
        align_items=[ali("说号就行", 0.0, 1.0), ali("啊", 1.3, 1.6)],
        diarization=[],
        full_text="说号就行。啊？",
        language_name="Chinese",
        duration=1.6,
        punctuation_split=False,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.6)]
    assert resp["segments"][0]["text"] == "说号就行。啊"

    # 完整旧行为回归（spec Scenario）：off + 显式 gap 0.8，同人间隙 1.0 →
    # 切分后同人聚合并回一段（升级前 0.8s 阈值 + merge_gap 2.0 的行为）
    resp = build_segment_response(
        align_items=[ali("今天不错", 0.0, 1.0), ali("明天更好", 2.0, 3.0)],  # 间隙 1.0
        diarization=[("SPEAKER_00", 0.0, 3.0)],
        full_text="今天不错。明天更好。",
        language_name="Chinese",
        duration=3.0,
        segment_gap_threshold=0.8,
        punctuation_split=False,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [(0.0, 3.0, "SPEAKER_00")]
    assert resp["segments"][0]["text"] == "今天不错。明天更好"

    # ---- 17. match 失败回退 / 说话人变化无标点照切 / segment 模式标点切分 -----
    # match 失败回退（spec Scenario 标点信息缺失回退）：item 文本不在
    # full_text → 纯间隙行为（无标点切分、无标点追加），不抛异常
    resp = build_segment_response(
        align_items=[ali("甲乙", 0.0, 1.0), ali("丙丁", 4.0, 5.0)],  # 间隙 3.0 ≥ 2.0
        diarization=[],
        full_text="完全不同的文本",
        language_name="Chinese",
        duration=5.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (4.0, 5.0)]
    assert [s["text"] for s in resp["segments"]] == ["甲乙", "丙丁"]  # 回退拼接，无追加
    # 匹配失败 + 小间隙 + full_text 含句末标点 → 无标点信息，不切分
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 1.2, 2.0)],  # 间隙 0.2 < 2.0
        diarization=[],
        full_text="丙。丁",
        language_name="Chinese",
        duration=2.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 2.0)]
    assert resp["segments"][0]["text"] == "甲乙"  # "甲"匹配失败 → 全量回退

    # 说话人变化无标点处照切（word 模式核心价值，不受标点影响）
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("没事", 1.2, 2.0)],  # 间隙 0.2，无标点
        diarization=[("SPEAKER_00", 0.0, 1.0), ("SPEAKER_01", 1.1, 2.0)],
        full_text="你好没事",
        language_name="Chinese",
        duration=2.0,
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [
        (0.0, 1.0, "SPEAKER_00"), (1.2, 2.0, "SPEAKER_01"),
    ]
    assert [s["text"] for s in resp["segments"]] == ["你好", "没事"]

    # segment 模式：标点切分与标点追加（含末段尾部）同样生效
    resp = build_segment_response(
        align_items=[ali("说号就行", 0.0, 1.0), ali("啊", 1.2, 1.5), ali("说吧", 1.7, 2.0)],
        diarization=[],
        full_text="说号就行。啊？说吧。",
        language_name="Chinese",
        duration=2.0,
        speaker_attribution="segment",
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (1.2, 1.5), (1.7, 2.0)]
    assert [s["text"] for s in resp["segments"]] == ["说号就行。", "啊？", "说吧。"]
    assert "".join(s["text"] for s in resp["segments"]) == resp["text"]

    print("pipeline self_test ok")
