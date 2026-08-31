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
- segment 切分采用**文本优先的三层解耦架构**（切点与对齐输出无关）：
  - Layer 1（纯文本，零 aligner 依赖）：直接扫描 ``full_text`` 按句末标点
    划分，粗粒度兜底块的字符区间边界强制切分。产出首尾相接、完整覆盖全文
    的字符区间；
  - Layer 2（时间映射）：把 align item 映射回字符区间（**局部回退**——单个
    token 失配只影响它自身，不再全量回退），区间时间取覆盖它的 items 的
    min/max；
  - Layer 3（item 维度细分）：文本段之内的段长强切，以及 ``hybrid`` 模式的
    间隙/说话人变化切分与同人二次聚合。``punctuation``（默认）跳过后者；
- 段文本 = ``full_text[c_start:c_end]`` **划分**（非截取）：相邻区间首尾相接，
  ``"".join(segments[].text) == text`` 由构造保证，无需标点追加/补偿逻辑；
- 说话人归属双模式：
  - ``segment``（段级投票，原有行为零改动）：segment 与 diarization 片段的
    时间重叠计算（dominant + speakers 列表）；
  - ``word``（词级归属，默认）：以 align_items 为词序列，词时间中点投票
    归属说话人，洞（无 turn 覆盖词）插值填充，按说话人变化切分 +
    同人二次聚合（含短插话保护）；
- 对齐失败块的粗粒度兜底段（coarse_chunks，两种归属模式均生效）：块字符
  区间内的标点同样触发切分，各子区间时间按字符偏移在块区间上线性分摊；
- speakerSummary 汇总（覆盖全部识别说话人，含零值项）。

本模块供 middleware 调用，也可被 example ``--self-test`` 离线自测，
不 import torch / pyannote / vLLM 等任何重依赖。
"""

import bisect
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


@dataclasses.dataclass
class _TextSpan:
    """Layer 1 产物：文本层切分单元（首尾相接，完整覆盖 ``full_text``）。

    Attributes:
        c_start: 字符区间起点（含）。
        c_end: 字符区间终点（不含）；恒有 ``c_end > c_start``。
        coarse_index: 落在粗粒度兜底块字符区间内时为对应 ``coarse_chunks``
            下标，否则为 ``None``。
    """

    c_start: int
    c_end: int
    coarse_index: Optional[int] = None

    @property
    def is_coarse(self) -> bool:
        return self.coarse_index is not None


def _resolve_coarse_char_spans(
    coarse: List[Tuple[str, float, float]],
    coarse_char_spans: Optional[List[Tuple[int, int]]],
    full_text: str,
) -> List[Tuple[int, int]]:
    """确定每个粗段在 ``full_text`` 中的字符区间（与 ``coarse`` 等长对齐）。

    优先用 middleware 基于 ``per_chunk`` 文本长度算出的精确区间。下标缺失、
    长度不匹配、越界或区间无效时**逐下标**退回游标 ``find`` 兜底（按 ``start``
    排序避免乱序错位）——必须兜底的原因：未能定位的粗段无法在 Layer 1 的字符
    域被切出去，其文本会被相邻正常段的字符区间一并卷入，与粗段自身 ``text``
    重复（拼接有损）。

    真的定位不到的下标以 ``(-1, -1)`` 占位，由调用方按"未定位"处理（该粗段
    退化为整块段，沿用既有行为）。

    Returns:
        与 ``coarse`` 一一对应的 ``(char_start, char_end)`` 列表。
    """
    limit = len(full_text)
    resolved: List[Optional[Tuple[int, int]]] = []
    for i in range(len(coarse)):
        pair = (
            coarse_char_spans[i]
            if coarse_char_spans is not None and i < len(coarse_char_spans)
            else None
        )
        if pair is not None:
            s, e = int(pair[0]), int(pair[1])
            if 0 <= s < e <= limit:
                resolved.append((s, e))
                continue
        resolved.append(None)
    # 未提供 / 无效的下标按时间顺序 find 兜底
    pending = [i for i, r in enumerate(resolved) if r is None]
    if pending:
        pos = 0
        for i in sorted(pending, key=lambda k: coarse[k][1]):
            coarse_text = coarse[i][0]
            if not coarse_text:
                continue
            idx = full_text.find(coarse_text, pos)
            if idx >= 0:
                resolved[i] = (idx, idx + len(coarse_text))
                pos = idx + len(coarse_text)
    return [(r if r is not None else (-1, -1)) for r in resolved]


def _split_text_spans(
    full_text: str,
    coarse_char_spans: Optional[List[Tuple[int, int]]] = None,
    split_on_punctuation: bool = True,
) -> List[_TextSpan]:
    """Layer 1：按句末标点划分 ``full_text``（纯文本，零 aligner 依赖）。

    切点（位置 ``p`` 表示 ``full_text[:p]`` 结束一个文本段）来自两类：

    - **句末标点连续串 + 其后空白**之后：``。`` / ``？！`` 等整串连同后续空白
      归入前一段，切点落在空白之后——等价既有「句末标点附前段」语义，故
      ``puncts`` 列表与「末段尾部标点追加」逻辑整体退场；
    - **粗粒度兜底块字符区间的起点与终点**：正常段不得横跨兜底块文本，替代
      既有 ``_gap_blocked`` 的**时间域**判定，改由字符域精确切割。

    段文本即 ``full_text[c_start:c_end]`` 原样切片。既有实现是**截取**
    （``full_text[首匹配起点:末匹配终点]``），between-span 中的字符必然丢失
    或错位（遗留 ❸ 的根因）；此处改为**划分**——相邻区间首尾相接且整体覆盖
    ``[0, len(full_text))``，``"".join(seg["text"]) == full_text`` 由构造保证。

    Args:
        full_text: 完整 ASR 文本。
        coarse_char_spans: 各粗段的 ``(char_start, char_end)``，与
            ``coarse_chunks`` 一一对应；``(-1, -1)`` 表示未定位，忽略。
        split_on_punctuation: False 时只按粗段边界切分（对应
            ``punctuation_split=False`` 的纯间隙行为）。

    Returns:
        首尾相接的 ``_TextSpan`` 列表；``full_text`` 为空时返回空列表。
    """
    n = len(full_text)
    if n == 0:
        return []
    coarse_ranges = [
        (int(cs), int(ce), idx)
        for idx, (cs, ce) in enumerate(coarse_char_spans or [])
        if cs is not None and ce is not None and 0 <= cs < ce
    ]
    cuts: set = set()
    for cs, ce, _idx in coarse_ranges:
        if 0 < cs < n:
            cuts.add(cs)
        if 0 < ce < n:
            cuts.add(ce)
    if split_on_punctuation:
        i = 0
        while i < n:
            if full_text[i] in _SENTENCE_END_CHARS:
                j = i
                while j < n and full_text[j] in _SENTENCE_END_CHARS:
                    j += 1
                while j < n and full_text[j].isspace():
                    j += 1
                if j < n:
                    cuts.add(j)
                i = j
            else:
                i += 1
    spans: List[_TextSpan] = []
    prev = 0
    for p in sorted(cuts) + [n]:
        if p <= prev or p > n:
            continue
        cidx = next(
            (idx for cs, ce, idx in coarse_ranges if cs <= prev < ce), None
        )
        spans.append(_TextSpan(prev, p, cidx))
        prev = p
    return spans


#: 句末标点集合（spec「句末标点硬切分」）：``。！？；`` + ASCII ``.!?;``
#: （含句点——英文句末标点主力，缺失则英文场景标点切分失效）+ 换行符；
#: 逗号 / 顿号 / 冒号不在内（保留段内，符合"一句话一个分段"）
_SENTENCE_END_CHARS = set("。！？；.!?;\n")


def _map_items_to_chars(
    items: List[Tuple[str, float, float]],
    full_text: str,
) -> List[Optional[Tuple[int, int]]]:
    """Layer 2：把每个对齐 item 映射到 ``full_text`` 的字符区间。

    沿用既有 greedy ``find`` 游标语义，但失败语义从**全量回退**改为
    **局部回退**：

    - 命中 → 记录 ``(起点, 终点)`` 并推进游标；
    - 未命中 → 记 ``None`` 且**游标不推进**（后续 item 仍按原游标继续匹配）。

    既有实现在任一 item 未命中时立即返回全 ``False`` 边界，使整份音频的
    标点切分失效（实测：196s 音频因单个 token 未命中，75 个句末标点全部
    失效，只切出 7 段）。局部回退把影响面收敛到该 token 自身——其余 item
    的字符位置信息完整保留，切分由 Layer 1 的文本扫描独立决定，不受牵连。

    aligner 的 token 经 ``clean_token`` 剥离标点与空白后与 ASR 原文可能不再
    逐字相等（``0.35``→``035``、``2024-01-01``→``20240101``、
    ``三零二X。GTDCH``→``XGTDCH``），此类 token 必然未命中；局部回退使其
    代价从"整段失效"降为"少一个时间锚点"。
    """
    spans: List[Optional[Tuple[int, int]]] = []
    pos = 0
    for text, _, _ in items:
        idx = full_text.find(text, pos)
        if idx < 0:
            spans.append(None)
            continue
        spans.append((idx, idx + len(text)))
        pos = idx + len(text)
    return spans


def _assign_item_buckets(
    text_spans: List[_TextSpan],
    item_chars: List[Optional[Tuple[int, int]]],
) -> List[int]:
    """Layer 2：判定每个 item 归属哪个文本段（按字符起点二分）。

    未命中（``None``）的 item 无法定位，继承前一个 item 的归属以保序；
    首个 item 即未命中时归 0。这保证 aligner 整体失配时全部 item 仍落在
    同一文本段，退化行为与既有「纯间隙/段长切分」一致。
    """
    if not text_spans:
        return []
    starts = [span.c_start for span in text_spans]
    last = len(text_spans) - 1
    buckets: List[int] = []
    prev = 0
    for chars in item_chars:
        if chars is None:
            buckets.append(prev)
            continue
        idx = bisect.bisect_right(starts, chars[0]) - 1
        if idx < 0:
            idx = 0
        elif idx > last:
            idx = last
        buckets.append(idx)
        prev = idx
    return buckets


def _absorb_orphan_buckets(
    text_spans: List[_TextSpan],
    bucket_items: List[List[int]],
) -> Tuple[List[_TextSpan], List[List[int]]]:
    """把无 item 覆盖的文本段（孤儿文本）并入相邻段。

    孤儿文本没有时间锚点，单独成段会产出零时长段，并入相邻段更合理：
    优先并入**前一个非粗段**（文本接续语义更自然），前一段不存在或是粗段时
    并入**后一个非粗段**。两向搜索**均不得跨越粗段**——否则正常段的字符区间
    会把粗段原文一并卷入，与粗段自身 ``text`` 重复。粗段自身有块区间时间，
    既不并入也不被并入。
    """
    n = len(text_spans)
    if n == 0:
        return [], []
    has_items = [bool(idxs) for idxs in bucket_items]
    target = list(range(n))

    def _search(i: int, step: int) -> int:
        """沿 step 方向找最近的可并入目标；遇粗段即止，找不到返回 -1。"""
        j = i + step
        while 0 <= j < n:
            if text_spans[j].is_coarse:
                return -1
            if has_items[j]:
                return j
            j += step
        return -1

    for i in range(n):
        if has_items[i] or text_spans[i].is_coarse:
            continue
        j = _search(i, -1)
        if j < 0:
            j = _search(i, 1)
        if j >= 0:
            target[i] = j

    out_spans: List[_TextSpan] = []
    out_items: List[List[int]] = []
    new_index: Dict[int, int] = {}
    for i in range(n):
        if target[i] == i:
            new_index[i] = len(out_spans)
            out_spans.append(text_spans[i])
            out_items.append(list(bucket_items[i]))
    for i in range(n):
        if target[i] == i:
            continue
        k = new_index.get(target[i])
        if k is None:
            continue
        prev = out_spans[k]
        out_spans[k] = _TextSpan(
            min(prev.c_start, text_spans[i].c_start),
            max(prev.c_end, text_spans[i].c_end),
            prev.coarse_index,
        )
    return out_spans, out_items


def _coarse_span_time(
    span: _TextSpan,
    coarse: List[Tuple[str, float, float]],
    coarse_char_spans: List[Tuple[int, int]],
) -> Tuple[float, float]:
    """粗段内子区间的起止时间：按字符偏移在块区间上线性分摊。

    Layer 1 会在粗段内部按标点继续切分（既有实现把整块 180s 文本作为单段
    产出，段内标点完全不切分），故此处为各子区间估出时间。块区间
    ``[t_start, t_end]`` 由 middleware 给出，分摊比例取子区间在块字符区间
    中的相对位置。
    """
    idx = span.coarse_index
    if idx is None or idx >= len(coarse) or idx >= len(coarse_char_spans):
        return 0.0, 0.0
    _text, t_start, t_end = coarse[idx]
    cs, ce = coarse_char_spans[idx]
    total = ce - cs
    if total <= 0:
        return float(t_start), float(t_end)
    frac_s = min(1.0, max(0.0, (span.c_start - cs) / float(total)))
    frac_e = min(1.0, max(0.0, (span.c_end - cs) / float(total)))
    return (
        t_start + (t_end - t_start) * frac_s,
        t_start + (t_end - t_start) * frac_e,
    )


def _subgroup_char_bounds(
    span: _TextSpan,
    groups: List[List[Any]],
    group_item_chars: List[List[Optional[Tuple[int, int]]]],
) -> Optional[List[Tuple[int, int]]]:
    """把文本段字符区间按子组划界；任一子组无已映射 item 时返回 ``None``。

    子组来自 Layer 3 的段长强切 / 间隙 / 说话人变化切分。划界规则：首子组
    起点为段起点、末子组终点为段终点、中间边界取后一子组首个已映射 item 的
    字符起点——保证各子组区间首尾相接、合起来恰好覆盖整个文本段。
    """
    if not groups:
        return []
    starts: List[int] = []
    for chars in group_item_chars:
        first = next((c[0] for c in chars if c is not None), None)
        if first is None:
            return None
        starts.append(first)
    bounds: List[Tuple[int, int]] = []
    cursor = span.c_start
    for k in range(1, len(starts)):
        cut = min(max(starts[k], cursor), span.c_end)
        bounds.append((cursor, cut))
        cursor = cut
    bounds.append((cursor, span.c_end))
    return bounds


def _group_text(
    group: List[Any],
    bounds: Optional[Tuple[int, int]],
    full_text: str,
) -> str:
    """子组文本：优先取字符区间切片，无法定位时回退拼接 item 文本。

    回退路径与既有实现一致：item 文本全 ASCII 用 ``" ".join``，否则
    ``"".join``（中文等无空格语言）。
    """
    if bounds is not None:
        return full_text[bounds[0]:bounds[1]]
    texts = [pair[0] for pair in group]
    joined = " ".join(texts) if all(t.isascii() for t in texts) else "".join(texts)
    return joined


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
    segment_split_mode: str = "punctuation",
    coarse_char_spans: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """构建 segment 模式响应 dict（纯函数，无副作用）。

    :param align_items: 对齐 item 列表（鸭子类型 ``.text`` / ``.start_time`` /
        ``.end_time``），可为空；
    :param diarization: diarization 片段（对象 / 三元组 / dict，见 ``_to_turns``）；
    :param full_text: 完整 ASR 文本（段文本从中直接切片，**划分**而非截取）；
    :param language_name: 内部语言名（经 ``language_name_to_code`` 输出码）；
    :param duration: 音频时长（秒）；
    :param process_time: 服务端总耗时（秒），``None`` 则响应中为 ``null``；
    :param segment_split_mode: 切分维度模式（仅 ``punctuation_split=True``
        时生效，False 时被忽略）——``punctuation``（默认）只按句末标点
        硬边界 + 段长上限切分：静音间隙完全不切（阈值视为无穷大）、
        word 模式句中说话人变化不切分（段 ``speaker`` 取段内词归属投票
        dominant、``speakers`` 为去重集合）、同人二次聚合跳过；
        ``hybrid`` 为上一代行为（标点 + 间隙 + word 模式说话人变化切分
        + 同人二次聚合）。跨失败块边界两模式均恒切分（遗留 ❶ 修复）；
    :param segment_gap_threshold: 相邻 item 之间**无句末标点**时的时间间隙
        切分阈值（秒，含；默认 2.0）——句末标点处无视间隙恒切分。
        **仅 ``hybrid`` 模式生效**（punctuation 模式下为无操作参数）；
    :param max_segment_seconds: 段长上限（秒，超过强切；两模式均生效，
        punctuation 模式下唯一的非标点切分来源）；
    :param speaker_attribution: 说话人归属模式——``word``（默认，词级归属：
        词中点投票 + 洞填充 + hybrid 模式说话人变化切分 + 同人二次聚合）或
        ``segment``（段级重叠投票，原有行为代码路径零改动）；
    :param speaker_merge_gap: word 模式同人相邻段合并阈值（秒，默认 2.0；
        ``<= 0`` 表示不合并）。**仅 ``hybrid`` 模式生效**（punctuation
        模式下同人二次聚合整体跳过，为无操作参数）；
    :param coarse_chunks: 对齐失败块的粗粒度兜底 ``(text, start, end)`` 列表
        （两种归属模式均生效）：块字符区间内的标点同样触发切分，各子区间
        时间按字符偏移在块区间上线性分摊（既有实现把整块作为单段产出，段内
        标点完全不切分）；与正常段混合产出时 ``segments[]`` 按 ``start``
        全局升序，粗段不参与同人二次聚合；
    :param punctuation_split: 句末标点硬切分开关（默认 True）：True 时按
        ``。！？；.!?;`` 及换行**直接扫描全文**划分文本段（切点与对齐输出
        无关，标点连同其后空白归入前段）；False 时只按粗段字符区间切分，
        退化为纯间隙/段长（word 模式含说话人变化）行为，``segment_split_mode``
        被忽略（由 serve 启动校验输出组合告警）。
    :param coarse_char_spans: 每个粗段在 ``full_text`` 中的精确字符区间
        ``[(char_start, char_end), ...]``，与 ``coarse_chunks`` 一一对应。
        **已由优化项升级为主路径依赖**——Layer 1 据此在字符域精确切割粗段
        边界（替代既有 ``_gap_blocked`` 的时间域判定）。None 时 pipeline 用
        ``coarse_text`` 在 ``full_text`` 中游标 ``find`` 兜底。
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

    # ---- Layer 1：纯文本切分（切点与对齐输出彻底解耦）----------------------
    # 切点 = 句末标点连续串（含其后空白）之后 + 粗段字符区间边界。段文本为
    # full_text 的直接切片，相邻区间首尾相接且整体覆盖全文，
    # "".join(seg["text"]) == full_text 由构造保证——既有「puncts 附前段 /
    # 末段尾部追加 / tail 排除粗段字符区间」三套补偿逻辑（含遗留 ❷）整体退场。
    coarse_spans = _resolve_coarse_char_spans(coarse, coarse_char_spans, full_text)
    text_spans = _split_text_spans(
        full_text, coarse_spans, split_on_punctuation=bool(punctuation_split)
    )
    if not text_spans and items:
        # full_text 为空却存在对齐 item（理论上不应发生——item 由文本对齐而来）：
        # 无文本可划分，退化为单一空文本段，段文本走回退拼接路径，避免整份
        # 响应丢段（既有实现在此场景亦产出回退拼接段）
        text_spans = [_TextSpan(0, 0)]

    # ---- Layer 2：item → 字符区间映射（局部回退）+ 文本段归属 --------------
    item_chars = _map_items_to_chars(items, full_text)
    item_bucket = _assign_item_buckets(text_spans, item_chars)
    bucket_items: List[List[int]] = [[] for _ in text_spans]
    for i, bucket in enumerate(item_bucket):
        if 0 <= bucket < len(bucket_items):
            bucket_items[bucket].append(i)
    text_spans, bucket_items = _absorb_orphan_buckets(text_spans, bucket_items)

    # 已在文本中定位的粗段（字符域已切分）；未定位者（字符区间反查失败 /
    # find 兜底失败）退化为**时间域**强制切分——正常段不得在时间上横跨失败块，
    # 沿用既有「跨失败块恒切分」语义（遗留 ❶）的兜底路径。
    located_coarse: set = {
        span.coarse_index for span in text_spans if span.is_coarse
    }
    unlocated_time: List[Tuple[float, float]] = [
        (start, end)
        for i, (_text, start, end) in enumerate(coarse)
        if i not in located_coarse
    ]
    time_forced: List[bool] = [False] * max(0, len(items) - 1)
    if unlocated_time:
        for i in range(len(items) - 1):
            if _gap_blocked(items[i][2], items[i + 1][1], unlocated_time):
                time_forced[i] = True

    # 切分维度模式（spec「segment 切分维度模式」）：punctuation（默认）=
    # 纯标点模式——间隙阈值视为无穷大、word 模式说话人变化不切分、同人二次
    # 聚合跳过；hybrid = 上一代三维混合行为，在 Layer 1 文本段**内部**细分。
    punctuation_only = bool(punctuation_split) and str(segment_split_mode) == "punctuation"
    gap_threshold_eff = float("inf") if punctuation_only else float(segment_gap_threshold)

    # 词级归属（仅 word 模式消费；两模式共用同一份 items 时间戳）
    attributions = _fill_gaps(items, _attribute_words(items, turns), turns) if items else []
    pairs: List[Tuple[str, float, float, Optional[str]]] = [
        (text, start, end, speaker)
        for (text, start, end), speaker in zip(items, attributions)
    ]

    segments: List[dict] = []
    # (speaker, 段时长) 原始值序列，供 speakerSummary 统计（避免二次遍历取整误差）
    dominant_records: List[Tuple[Optional[str], float]] = []
    coarse_time_blocked = [(start, end) for _, start, end in coarse]

    def _segment_vote(seg_start: float, seg_end: float) -> Tuple[Optional[str], List[str]]:
        """segment 归属模式：段级重叠投票（既有行为，公式零改动）。

        重叠总时长 Σ max(0, min(e,te)-max(s,ts)) 降序（并列按 id 升序）首者
        为 dominant，重叠 ≥ 0.1s 者入 speakers。
        """
        overlap: Dict[str, float] = {}
        for turn in turns:
            ov = min(seg_end, turn.end_time) - max(seg_start, turn.start_time)
            if ov > 0:
                overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + ov
        ranked = sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0]))
        return (
            ranked[0][0] if ranked else None,
            [sp for sp, ov in ranked if ov >= _MIN_SPEAKER_OVERLAP],
        )

    for b, span in enumerate(text_spans):
        if span.is_coarse:
            # 粗段：Layer 1 已在块内按标点切分，各子区间按字符偏移在块区间
            # 上线性分摊时间（既有实现整块 180s 单段产出，段内标点不切分）。
            # **刻意不施加 max_segment_seconds 强切**：粗段时间戳本就是线性
            # 估算，再按段长切碎只会制造虚假的时间精度；且块长上限为
            # MAX_FORCE_ALIGN_INPUT_SECONDS(180s)，按标点切分后通常远小于此。
            # 已知局限：块内完全无标点时仍产出与块等长的单段。
            seg_start, seg_end = _coarse_span_time(span, coarse, coarse_spans)
            speaker, speakers = _coarse_vote(turns, seg_start, seg_end)
            segments.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": full_text[span.c_start:span.c_end],
                "speaker": speaker,
                "speakers": speakers,
            })
            dominant_records.append((speaker, seg_end - seg_start))
            continue

        idxs = bucket_items[b]
        if not idxs:
            continue  # 孤儿文本已由 _absorb_orphan_buckets 并入相邻段

        # 段内 item 间的时间域强制切分（未定位粗段的遗留 ❶ 兜底）
        sub_boundaries = [
            any(time_forced[j] for j in range(idxs[k], min(idxs[k + 1], len(time_forced))))
            for k in range(len(idxs) - 1)
        ]

        # ---- Layer 3：文本段之内的 item 维度细分（段长强切 / 间隙 / 说话人）
        if speaker_attribution == "word":
            sub_pairs = [pairs[i] for i in idxs]
            if punctuation_only:
                # 纯标点模式：段内仅段长强切（说话人变化不拆段、不聚合）
                groups = _split_groups(
                    sub_pairs, gap_threshold_eff, max_segment_seconds, sub_boundaries
                )
            else:
                groups = _split_by_speaker(
                    sub_pairs, gap_threshold_eff, max_segment_seconds, sub_boundaries
                )
                groups = _merge_same_speaker(
                    groups, turns, speaker_merge_gap, max_segment_seconds,
                    blocked=coarse_time_blocked, hard_boundaries=sub_boundaries,
                )
        else:
            groups = _split_groups(
                [items[i] for i in idxs], gap_threshold_eff, max_segment_seconds,
                sub_boundaries,
            )

        # 子组 → 字符区间：按各子组首个已映射 item 的字符起点划界，保证各
        # 子组区间首尾相接且合起来恰好覆盖本文本段（拼接无损）
        group_chars: List[List[Optional[Tuple[int, int]]]] = []
        cursor_i = 0
        for group in groups:
            k = len(group)
            group_chars.append([item_chars[i] for i in idxs[cursor_i:cursor_i + k]])
            cursor_i += k
        bounds_list = _subgroup_char_bounds(span, groups, group_chars)

        for gi, group in enumerate(groups):
            seg_start = group[0][1]
            seg_end = group[-1][2]
            bounds = bounds_list[gi] if bounds_list is not None else None
            if speaker_attribution == "word":
                speaker, speakers = _word_vote(group)
            else:
                speaker, speakers = _segment_vote(seg_start, seg_end)
            segments.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": _group_text(group, bounds, full_text),
                "speaker": speaker,
                "speakers": speakers,
            })
            dominant_records.append((speaker, seg_end - seg_start))

    # ---- 未在 full_text 中定位到的粗段（字符区间反查失败 / find 兜底失败）---
    # 退化为既有「整块单段 + 块区间投票」行为，保证兜底文本不丢
    for ci, (coarse_text, cs_time, ce_time) in enumerate(coarse):
        if ci in located_coarse:
            continue
        speaker, speakers = _coarse_vote(turns, cs_time, ce_time)
        segments.append({
            "start": round(cs_time, 3),
            "end": round(ce_time, 3),
            "text": coarse_text,
            "speaker": speaker,
            "speakers": speakers,
        })
        dominant_records.append((speaker, ce_time - cs_time))

    # segments[] 按 start 全局升序（稳定排序，粗段插入其时间区间的正确位置）
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
    # （间隙切分仅 hybrid 模式，显式传 mode="hybrid"）
    boundary_items = [ali("a", 0.0, 1.0), ali("b", 2.0, 3.0)]  # 间隙 1.0
    resp = build_segment_response(boundary_items, [], "a b", "English", 3.0, segment_gap_threshold=1.0, speaker_attribution="segment", segment_split_mode="hybrid")
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (2.0, 3.0)]
    resp = build_segment_response(boundary_items, [], "a b", "English", 3.0, segment_gap_threshold=1.25, speaker_attribution="segment", segment_split_mode="hybrid")
    assert len(resp["segments"]) == 1

    # 阈值参数透传：同一间隙 0.6，默认不切、threshold=0.5 切（hybrid 模式）
    gap_items = [ali("a", 0.0, 1.0), ali("b", 1.6, 2.0)]
    assert len(build_segment_response(gap_items, [], "a b", "English", 2.0, speaker_attribution="segment", segment_split_mode="hybrid")["segments"]) == 1
    resp = build_segment_response(gap_items, [], "a b", "English", 2.0, segment_gap_threshold=0.5, speaker_attribution="segment", segment_split_mode="hybrid")
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

    # ---- 4. 文本划分保留标点/空格 与 找不到回退拼接 --------------------------
    full = "Hello, world. Nice to meet you."
    resp = build_segment_response(
        align_items=[ali("Hello", 0.0, 0.5), ali("world", 0.6, 1.0), ali("Nice", 2.0, 2.4), ali("meet", 2.5, 2.8)],
        diarization=[],
        full_text=full,
        language_name="English",
        duration=3.0,
        speaker_attribution="segment",
    )
    # 文本层按 ASCII 句点切分（间隙 1.0 < 2.0，切分由句点驱动）；段文本为
    # full_text 原样切片——句点连同其后空格归入前段，末段保留未对齐词 you
    # （既有实现只截取到末 item 匹配终点，"you" 被丢弃、末段仅剩追加句点）
    assert resp["segments"][0]["text"] == "Hello, world. "
    assert resp["segments"][1]["text"] == "Nice to meet you."
    assert "".join(s["text"] for s in resp["segments"]) == full  # 拼接无损

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
        segment_split_mode="hybrid",  # 间隙切分仅 hybrid 模式
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
        segment_split_mode="hybrid",  # 短插话保护依赖间隙切分 + 聚合阻断
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
        segment_split_mode="hybrid",  # 间隙切分 + 聚合均仅 hybrid 模式
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 0.8), (2.0, 2.8)]

    # 全 None 归属（无 diarization，默认 word 模式）：仅按间隙/段长切分，
    # 段 speaker=null（间隙 2.0 恰达阈值 → 切分，hybrid 模式）
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世界", 3.0, 4.0)],
        diarization=[],
        full_text="你好世界",
        language_name="Chinese",
        duration=4.0,
        segment_split_mode="hybrid",
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

    # ---- 12. Layer 1/2 helper 级：_split_text_spans / _map_items_to_chars ----
    # 句号/问号/叹号/分号/ASCII 句点/换行均触发切分；逗号/顿号不触发
    full_text12 = "甲。乙？丙！丁；戊.己\n庚，辛"
    spans12 = _split_text_spans(full_text12)
    assert [(s.c_start, s.c_end) for s in spans12] == [
        (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 15),
    ]
    assert [s.is_coarse for s in spans12] == [False] * 7
    # 段文本即原样切片：相邻区间首尾相接 → 拼接无损由构造保证
    assert "".join(full_text12[s.c_start:s.c_end] for s in spans12) == full_text12

    # 连续句末标点"？！"连同其后空白归入前一段（切点落在空白之后）
    spans12b = _split_text_spans("甲？！ 乙")
    assert [(s.c_start, s.c_end) for s in spans12b] == [(0, 4), (4, 5)]
    assert "甲？！ 乙"[spans12b[0].c_start:spans12b[0].c_end] == "甲？！ "
    assert "甲？！ 乙"[spans12b[1].c_start:spans12b[1].c_end] == "乙"

    # 粗段字符区间边界强制切分，且粗段内部标点照常切分（既有实现整块单段产出）
    spans12c = _split_text_spans("甲失败。块转写！乙。", [(1, 8)])
    assert [(s.c_start, s.c_end, s.is_coarse) for s in spans12c] == [
        (0, 1, False), (1, 4, True), (4, 8, True), (8, 10, False),
    ]
    assert [s.coarse_index for s in spans12c] == [None, 0, 0, None]

    # split_on_punctuation=False：只按粗段边界切分（punctuation_split=False 路径）
    spans12d = _split_text_spans("甲。乙。丙", [(2, 4)], split_on_punctuation=False)
    assert [(s.c_start, s.c_end, s.coarse_index) for s in spans12d] == [
        (0, 2, None), (2, 4, 0), (4, 5, None),
    ]
    # 空文本 → 空列表
    assert _split_text_spans("") == []

    # _map_items_to_chars：全部命中 → 逐 item 字符区间
    assert _map_items_to_chars(
        [("甲", 0.0, 0.5), ("乙", 0.5, 1.0)], "甲。乙"
    ) == [(0, 1), (2, 3)]

    # **核心回归**：单 token 未命中 → 局部回退（记 None、游标不推进），其余
    # item 位置完整保留。既有实现在此全量回退，使整份音频标点切分失效
    # （实测 196s 音频因单个 token 未命中，75 个句末标点全部失效只切出 7 段）
    assert _map_items_to_chars(
        [("甲", 0.0, 0.5), ("035", 0.5, 1.0), ("乙", 1.0, 1.5)], "甲0.35乙"
    ) == [(0, 1), None, (5, 6)]
    # 连续多个未命中互不影响，后续仍可恢复匹配
    assert _map_items_to_chars(
        [("甲", 0.0, 0.5), ("X", 0.5, 1.0), ("Y", 1.0, 1.5), ("乙", 1.5, 2.0)],
        "甲。乙",
    ) == [(0, 1), None, None, (2, 3)]

    # _assign_item_buckets：按字符起点归属；未命中继承前一个 item
    spans12e = _split_text_spans("甲。乙。丙")
    assert _assign_item_buckets(spans12e, [(0, 1), None, (2, 3), (4, 5)]) == [0, 0, 1, 2]
    assert _assign_item_buckets([], []) == []

    # _absorb_orphan_buckets：无 item 的段并入相邻段（优先前一段）
    spans12f = _split_text_spans("甲。乙。丙。丁")
    m_spans, m_items = _absorb_orphan_buckets(spans12f, [[0], [], [2], [3]])
    assert [(s.c_start, s.c_end) for s in m_spans] == [(0, 4), (4, 6), (6, 7)]
    assert m_items == [[0], [2], [3]]
    # 首段为孤儿时并入后一段
    m_spans, m_items = _absorb_orphan_buckets(spans12f, [[], [1], [2], [3]])
    assert [(s.c_start, s.c_end) for s in m_spans] == [(0, 4), (4, 6), (6, 7)]
    assert m_items == [[1], [2], [3]]
    # 粗段不参与并入，且并入不得跨越粗段（否则粗段原文会被卷进正常段）
    m_spans, m_items = _absorb_orphan_buckets(
        _split_text_spans("甲。乙。丙", [(2, 4)]), [[], [], [2]]
    )
    assert [(s.c_start, s.c_end, s.is_coarse) for s in m_spans] == [
        (0, 2, False), (2, 4, True), (4, 5, False),
    ]
    assert m_items == [[], [], [2]]  # 首段孤儿无法跨越粗段并入，保持独立

    # _coarse_span_time：块内子区间按字符偏移线性分摊块时间
    spans12g = _split_text_spans("甲失败。块转写！乙。", [(1, 8)])
    assert _coarse_span_time(spans12g[1], [("失败。块转写！", 10.0, 20.0)], [(1, 8)]) == (
        10.0, 10.0 + 10.0 * 3 / 7,
    )
    assert _coarse_span_time(spans12g[2], [("失败。块转写！", 10.0, 20.0)], [(1, 8)]) == (
        10.0 + 10.0 * 3 / 7, 20.0,
    )

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
    # 句点连同其后空格归入前段；末段延伸至文末，尾部句点天然含在段内
    assert resp["segments"][0]["text"] == "Nice to meet you. "
    assert resp["segments"][1]["text"] == "See you."
    assert "".join(s["text"] for s in resp["segments"]) == "Nice to meet you. See you."

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

    # 连续句末标点"？！ "（spec Scenario）：切点落在标点串及其后空白之后，
    # "？！ " 整体归入前段（既有实现丢弃空格，"".join 拼接有损）
    resp = build_segment_response(
        align_items=[ali("啊", 0.0, 0.5), ali("什么", 1.0, 1.5)],
        diarization=[],
        full_text="啊？！ 什么",
        language_name="Chinese",
        duration=1.5,
    )
    assert [s["text"] for s in resp["segments"]] == ["啊？！ ", "什么"]
    assert "".join(s["text"] for s in resp["segments"]) == "啊？！ 什么"

    # 跨失败块边界（spec Scenario）：甲|乙 隔失败块（块文本含多个句末标点）→
    # 正常段不横跨失败块；**粗段内部按标点继续切分**（既有实现整块单段产出，
    # 段内标点不切分），各子区间按字符偏移线性分摊块时间；混合场景拼接无损
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 4.0, 5.0)],  # 间隙 [1.0, 4.0]
        diarization=[("SPEAKER_00", 0.0, 5.0)],
        full_text="甲失败。块转写！乙。",
        language_name="Chinese",
        duration=5.0,
        coarse_chunks=[("失败。块转写！", 1.5, 3.5)],  # 字符区间由 find 定位
    )
    segs = resp["segments"]
    # 粗段 [1.5, 3.5] 按块内标点切成 2 段，时间按字符偏移分摊（10 字符中占 3 + 4）
    assert [(s["start"], s["end"]) for s in segs] == [
        (0.0, 1.0), (1.5, 2.357), (2.357, 3.5), (4.0, 5.0),
    ]
    assert segs[0]["text"] == "甲"          # 正常段不含失败块文本
    assert segs[1]["text"] == "失败。"       # 粗段子区间 1（原样切片，含自身标点）
    assert segs[2]["text"] == "块转写！"     # 粗段子区间 2
    assert segs[3]["text"] == "乙。"         # 末段延伸至文末
    assert "".join(s["text"] for s in segs) == resp["text"]  # 拼接无损

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

    # 长静音仍切分（spec Scenario，hybrid 模式）：无标点间隙恰好 2.0
    # （含边界值）→ 切分（punctuation 模式下间隙不切，见组 18 专项断言）
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 3.0, 4.0)],
        diarization=[],
        full_text="甲乙",
        language_name="Chinese",
        duration=4.0,
        segment_split_mode="hybrid",
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

    # 默认参数下聚合零触发（spec 推导结论，hybrid 模式）：默认 gap 2.0 /
    # merge_gap 2.0，同人间隙 2.5s 切分两段后 gap ≥ 2.0 不满足合并条件
    # gap < 2.0 → 不合并
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 3.5, 4.5)],
        diarization=[("SPEAKER_00", 0.0, 4.5)],
        full_text="甲乙",
        language_name="Chinese",
        duration=4.5,
        segment_split_mode="hybrid",
    )
    assert [(s["start"], s["end"], s["speaker"]) for s in resp["segments"]] == [
        (0.0, 1.0, "SPEAKER_00"), (3.5, 4.5, "SPEAKER_00"),
    ]

    # punctuation_split=False（spec Scenario 线上误切即时关闭）：不按标点切分，
    # 纯间隙/段长行为（word 模式含说话人变化）；段文本仍为 full_text 原样切片
    resp = build_segment_response(
        align_items=[ali("说号就行", 0.0, 1.0), ali("啊", 1.3, 1.6)],
        diarization=[],
        full_text="说号就行。啊？",
        language_name="Chinese",
        duration=1.6,
        punctuation_split=False,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.6)]
    # 既有实现末段丢尾部"？"（matched=False 时不追加）；新实现末段延伸至文末
    assert resp["segments"][0]["text"] == "说号就行。啊？"

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
    assert resp["segments"][0]["text"] == "今天不错。明天更好。"  # 末段延伸至文末

    # ---- 17. match 失败回退 / 说话人变化无标点照切 / segment 模式标点切分 -----
    # match 失败回退（spec Scenario 标点信息缺失回退，hybrid 模式）：item
    # 文本不在 full_text → 纯间隙行为（无标点切分、无标点追加），不抛异常
    resp = build_segment_response(
        align_items=[ali("甲乙", 0.0, 1.0), ali("丙丁", 4.0, 5.0)],  # 间隙 3.0 ≥ 2.0
        diarization=[],
        full_text="完全不同的文本",
        language_name="Chinese",
        duration=5.0,
        segment_split_mode="hybrid",
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

    # 说话人变化无标点处照切（word 模式核心价值，不受标点影响；hybrid 模式）
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("没事", 1.2, 2.0)],  # 间隙 0.2，无标点
        diarization=[("SPEAKER_00", 0.0, 1.0), ("SPEAKER_01", 1.1, 2.0)],
        full_text="你好没事",
        language_name="Chinese",
        duration=2.0,
        segment_split_mode="hybrid",
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

    # ---- 18. punctuation 模式专项（新默认：间隙/说话人变化均不切分）----------
    # 长停顿不拆段（spec Scenario 一句话中间长停顿不再拆段，用户问题 1）：
    # 无标点间隙 3.0s > 2.0 阈值 → punctuation 模式不切（gap 阈值视为无穷大）
    pause3_items = [ali("想", 0.0, 1.0), ali("负责", 4.0, 5.0)]  # 间隙 3.0
    resp = build_segment_response(
        align_items=pause3_items,
        diarization=[],
        full_text="想负责",
        language_name="Chinese",
        duration=5.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 5.0)]
    assert resp["segments"][0]["text"] == "想负责"
    # hybrid 对照：同一 fixture 间隙 3.0 ≥ 2.0 → 切为两段
    resp = build_segment_response(
        align_items=pause3_items,
        diarization=[],
        full_text="想负责",
        language_name="Chinese",
        duration=5.0,
        segment_split_mode="hybrid",
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (4.0, 5.0)]

    # 句中说话人变化不拆段（spec Scenario）：word 模式、无标点、说话人变化
    # → 不切分；段 speaker=词归属票数多者（dominant），speakers=去重集合
    resp = build_segment_response(
        align_items=[
            ali("你", 0.0, 0.3), ali("好", 0.3, 0.6), ali("啊", 0.6, 0.9),  # SPEAKER_00 ×3
            ali("没", 1.0, 1.3), ali("事", 1.3, 1.6),                        # SPEAKER_01 ×2
        ],
        diarization=[("SPEAKER_00", 0.0, 0.9), ("SPEAKER_01", 1.0, 1.6)],
        full_text="你好啊没事",
        language_name="Chinese",
        duration=1.6,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.6)]
    assert resp["segments"][0]["speaker"] == "SPEAKER_00"  # 3 票 > 2 票
    assert resp["segments"][0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]  # 词数降序
    assert resp["segments"][0]["text"] == "你好啊没事"
    # hybrid 对照：同 fixture 说话人变化处切分（组 17 已有同类断言，此处仅对照段数）
    resp = build_segment_response(
        align_items=[
            ali("你", 0.0, 0.3), ali("好", 0.3, 0.6), ali("啊", 0.6, 0.9),
            ali("没", 1.0, 1.3), ali("事", 1.3, 1.6),
        ],
        diarization=[("SPEAKER_00", 0.0, 0.9), ("SPEAKER_01", 1.0, 1.6)],
        full_text="你好啊没事",
        language_name="Chinese",
        duration=1.6,
        segment_split_mode="hybrid",
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 0.9), (1.0, 1.6)]

    # 段长兜底强切（punctuation 模式下唯一的非标点切分来源）：无标点长 span
    long_items = [ali("甲", 0.0, 15.0), ali("乙", 15.5, 30.5)]  # span 30.5 > 30
    resp = build_segment_response(
        align_items=long_items,
        diarization=[],
        full_text="甲乙",
        language_name="Chinese",
        duration=30.5,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 15.0), (15.5, 30.5)]

    # match 失败回退仅剩段长（spec Scenario 整体匹配失败回退退化）：item 文本
    # 不在 full_text → 标点信息缺失，punctuation 模式下小间隙不切（单段）；
    # 30s 以上长 span 仅段长强切（cut 兜底不受匹配失败影响）
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0), ali("乙", 1.5, 2.0)],  # 间隙 0.5
        diarization=[],
        full_text="完全不同的文本",
        language_name="Chinese",
        duration=2.0,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 2.0)]
    miss_items = [ali("甲", 0.0, 15.0), ali("乙", 15.5, 30.5)]
    resp = build_segment_response(
        align_items=miss_items,
        diarization=[],
        full_text="完全不同的文本",
        language_name="Chinese",
        duration=30.5,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 15.0), (15.5, 30.5)]

    # off 时 mode 无效（spec Scenario punctuation-split off 组合）：off +
    # mode=punctuation（默认）→ mode 被忽略，纯间隙/段长行为（间隙 3.0 ≥ 2.0 切）
    resp = build_segment_response(
        align_items=pause3_items,
        diarization=[],
        full_text="想负责",
        language_name="Chinese",
        duration=5.0,
        punctuation_split=False,
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (4.0, 5.0)]

    # segment 归属模式：punctuation 模式同样生效（间隙 3.0 不切，标点照切）
    resp = build_segment_response(
        align_items=[ali("你好", 0.0, 1.0), ali("世界", 4.0, 5.0)],
        diarization=[],
        full_text="你好。世界",
        language_name="Chinese",
        duration=5.0,
        speaker_attribution="segment",
    )
    assert [(s["start"], s["end"]) for s in resp["segments"]] == [(0.0, 1.0), (4.0, 5.0)]
    assert [s["text"] for s in resp["segments"]] == ["你好。", "世界"]

    # 跨失败块强制切分（遗留 ❶ 回归，spec Scenario 无标点跨失败块恒切分）：
    # 交界无句末标点（fixture 避开遗留 ❸ 交界标点丢失场景）+ 间隙 < 2.0s +
    # 跨 coarse 块 → punctuation 与 hybrid 两模式均切分；段文本不含失败块
    # 文本（失败块文本仅计入粗段一次），交界无标点下按 start 排序拼接一致
    coarse_fixture = [ali("甲", 0.0, 1.0), ali("乙", 1.9, 2.5)]  # 间隙 [1.0, 1.9] < 2.0
    for mode in ("punctuation", "hybrid"):
        resp = build_segment_response(
            align_items=coarse_fixture,
            diarization=[("SPEAKER_00", 0.0, 2.5)],
            full_text="甲失败块文本乙",
            language_name="Chinese",
            duration=2.5,
            coarse_chunks=[("失败块文本", 1.1, 1.8)],  # 与间隙 [1.0, 1.9] 相交
            segment_split_mode=mode,
        )
        segs = resp["segments"]
        assert [(s["start"], s["end"]) for s in segs] == [(0.0, 1.0), (1.1, 1.8), (1.9, 2.5)]
        # 段文本不含失败块文本：若无强制切分，首段会截成"甲失败块文本"（重复）
        assert segs[0]["text"] == "甲" and segs[2]["text"] == "乙"
        assert segs[1]["text"] == "失败块文本"  # 粗段取块 ASR 原文（仅计入一次）
        assert "".join(s["text"] for s in segs) == resp["text"]  # 交界无标点 → 拼接一致

    # ---- 19. P0 场景：尾部空 items 块标点不重复（coarse_char_spans 精确区间）--
    # 尾部空 items 块进 coarse 后，其文本含句末标点。既有实现靠独立的 tail 逻辑
    # 跳过 coarse 字符区间来避免重复追加；新实现由 Layer 1 在**字符域**直接把
    # 粗段切出去，末段天然不会延伸到粗段文本，无需任何补偿逻辑。
    # 场景：正常块"甲" + 尾部空 items 块"乙说完了。"
    # 精确区间：coarse_char_spans=[(1, 6)]（"甲"1 字符，"乙说完了。"5 字符）
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0)],
        diarization=[],
        full_text="甲乙说完了。",
        language_name="Chinese",
        duration=3.0,
        coarse_chunks=[("乙说完了。", 1.5, 3.0)],
        coarse_char_spans=[(1, 6)],
    )
    segs = resp["segments"]
    assert [(s["start"], s["end"]) for s in segs] == [(0.0, 1.0), (1.5, 3.0)]
    assert segs[0]["text"] == "甲"  # 字符域切割，前段不含粗段文本与其标点
    assert segs[1]["text"] == "乙说完了。"  # 粗段含自身标点
    assert "".join(s["text"] for s in segs) == resp["text"]  # 拼接无损

    # 同场景不传 coarse_char_spans（None → pipeline find 兜底）
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0)],
        diarization=[],
        full_text="甲乙说完了。",
        language_name="Chinese",
        duration=3.0,
        coarse_chunks=[("乙说完了。", 1.5, 3.0)],
    )
    segs = resp["segments"]
    assert segs[0]["text"] == "甲"  # find 兜底同样排除 coarse 区间
    assert segs[1]["text"] == "乙说完了。"
    assert "".join(s["text"] for s in segs) == resp["text"]

    # 无 coarse 块时 tail 行为不变（既有行为回归）
    resp = build_segment_response(
        align_items=[ali("甲", 0.0, 1.0)],
        diarization=[],
        full_text="甲。",
        language_name="Chinese",
        duration=1.0,
    )
    assert resp["segments"][0]["text"] == "甲。"  # 末段延伸至文末

    # ---- 20. 脱敏回归：证件号念读导致 token 失配（局部回退核心场景）----------
    # 真实故障：196.544s 执法音频中身份证号被重复念读（"三零二X。三零二X。"），
    # 经 aligner 的 clean_token 剥掉句号后拉丁/数字连串被合并为 token "XGTDCH"
    # （原文实为 "X。GTDCH"）→ find 失败 → 既有实现**全量回退**，全文 75 个
    # 句末标点全部失效、只剩 30s 段长强切（仅切出 7 段，拼接还少 5 字）。
    # 此处用**脱敏**的同类结构假数据复现该模式（真实证件号/车牌不进仓库），
    # 验证局部回退后：标点切分不再受单 token 失配牵连，且拼接无损。
    import unicodedata

    def _aligner_tokens(text: str) -> List[str]:
        """复刻 Qwen3ForceAlignProcessor 分词：剥掉标点与空白后拉丁/数字连串合并。"""

        def _kept(ch: str) -> bool:
            return ch == "'" or unicodedata.category(ch).startswith(("L", "N"))

        toks: List[str] = []
        for seg in text.split():
            cleaned = "".join(ch for ch in seg if _kept(ch))
            if not cleaned:
                continue
            buf: List[str] = []
            for ch in cleaned:
                if 0x4E00 <= ord(ch) <= 0x9FFF:  # CJK 逐字成 token
                    if buf:
                        toks.append("".join(buf))
                        buf = []
                    toks.append(ch)
                else:
                    buf.append(ch)
            if buf:
                toks.append("".join(buf))
        return toks

    case_text = (
        "我找他，找他了。问他是不是变道，是吧？身份证。先说身份证号。"
        "身幺三幺。幺三幺。五幺七K。五幺七K。XQPZW三九。"
        "看这个样子，应该是你变道，变道了。啊。看这个，这个，这个从你这个角度啊。"
        "啊。我看是应该是你变道，没没注意看他打车。啊，对对对。是吧？对对。"
        "要是打车，要是他要是他要是变道的话，他不会发生，不会像你这么歪歪着这么歪。"
        "啊，对，是我往这边并的，是吧？他顶着我了。对。"
        "这个我们到达现场以后，你们发生交通事故，后方未摆放警告标志。"
        "发生事故以后，后边必须放三角架，知道吗？"
        "你们没有放，我们到了以后你们都没有。我是想放，我卸不下来。"
    )
    case_toks = _aligner_tokens(case_text)
    # 复现真实根因：存在"清洗后在原文中不连续存在"的 token
    assert "KXQPZW" in case_toks
    assert case_text.find("KXQPZW") < 0
    case_dur = 120.0
    step20 = case_dur / (len(case_toks) + 1)
    case_items = [
        ali(t, round(step20 * (i + 1), 3), round(step20 * (i + 1) + step20 * 0.6, 3))
        for i, t in enumerate(case_toks)
    ]
    resp = build_segment_response(case_items, [], case_text, "Chinese", case_dur)
    segs20 = resp["segments"]
    # 全量回退下 120s 只会被 30s 段长切成约 5 段；局部回退后回到标点量级
    assert len(segs20) == 25, len(segs20)
    assert "".join(s["text"] for s in segs20) == case_text  # 拼接无损
    # 证件号两句各自独立成段（既有实现会把它们并进 30s 粗块且不切分）
    texts20 = [s["text"] for s in segs20]
    assert "五幺七K。" in texts20 and "XQPZW三九。" in texts20

    print("pipeline self_test ok")
