# 文本优先 Segment 切分解耦 Spec

## Why

部署 `cu128-punct2` 后，执法记录仪音频出现「有句末标点却不切分，全部聚合在一个超长 segment 段」的偶发问题。

**真实案例**（196.544s 执法音频）：

| 指标 | 观测值 |
|---|---|
| 全文 `text` | 815 字，**75 个句末标点** |
| 实际段数 | **7 段**（正常应约 76 段） |
| 最长段时长 | 29.893s（卡在 `--max-segment-seconds` 默认 30s 上限） |
| `join(segments[].text)` | 810 字，**少 5 字** |

**根因链**：

```
民警念身份证号并重复了一遍
  ↓
ASR: "...三零二X。三零二X。GTDCH八五。看这个样子..."
  ↓
aligner `clean_token` 剥掉全部标点 → 拉丁/数字连串被并成单 token
  ↓
token = 'XGTDCH'（原文实为 "X。GTDCH"）
  ↓
full_text.find('XGTDCH') = -1
  ↓
_sentence_end_boundaries 单 token 失败即**全量回退**
  ↓
75 个句末标点全部失效 → 只剩 30s 段长强切 → 7 段
```

**结构性问题**：既有架构把「在哪里切」（纯文本问题）与「切点的时间戳是多少」（对齐问题）耦合成了同一套 greedy `find` 机制。标点切分本应是 100% 确定性的文本操作，却退化成了依赖 aligner 输出质量的概率操作。

**为什么是"偶发"**：触发条件是"文本中出现清洗后无法在原文中连续匹配的 token"。执法场景最高发的恰恰是念证件号/车牌/金额/日期（`0.35`→`035`、`2024-01-01`→`20240101`、`10,000`→`10000`）；且 ASR 念读数串时极易重复，重复会让 greedy 游标提前消耗首次出现，第二次必然失配。196s 音频 ≈ 20 个 180s 块，单块失配概率累积后"偶发但不罕见"。

## What Changes

### 架构：三层解耦（划分代替截取）

| | 既有 | 新架构 |
|---|---|---|
| 段文本 | `full_text[首匹配:末匹配]` **截取** | `full_text[c_start:c_end]` **划分** |
| 切点来源 | token 匹配位置的 between-span | **文本扫描**（零 aligner 依赖） |
| 单 token 失配 | 全量回退，75 个标点全失效 | 仅该 token 少一个时间锚点 |
| 段末标点 | `puncts` 列表追加 | 含在字符区间内 |
| 尾部标点 | 独立 tail 逻辑 + coarse 排除 | 末段延伸至文末 |
| 粗段 | 整块 180s 不切 | 按标点切分 + 线性分摊时间 |

**Layer 1（文本层，零 aligner 依赖）**：新增 `_split_text_spans()`，按 `_SENTENCE_END_CHARS` 扫描 `full_text` 划分；粗段字符区间边界强制切分。产出首尾相接、完整覆盖 `[0, len(full_text))` 的字符区间，``"".join(seg["text"]) == full_text`` 由构造保证。

**Layer 2（时间映射层）**：
- 新增 `_map_items_to_chars()`：item → 字符区间，**局部回退**——单 token 失配记 `None` 且游标不推进，继续匹配后续 token（替代全量回退）。
- 新增 `_assign_item_buckets()`：item 按字符起点归属文本段；失配 item 继承前一个 item 的归属以保序。
- 新增 `_absorb_orphan_buckets()`：无 item 覆盖的文本段（孤儿文本）并入相邻段，**并入不得跨越粗段**。

**Layer 3（item 维度细分）**：文本段之内的段长强切，以及 `hybrid` 模式的间隙/说话人变化切分与同人二次聚合。`punctuation`（默认）跳过后者。

**退场的逻辑**：`puncts` 列表、末段尾部标点追加、`_sentence_end_boundaries`（全量回退载体）、`_extract_segment_text`（截取载体）全部移除。

### 附带修复

| 缺陷 | 说明 |
|---|---|
| **路径① 全局匹配失败** | 本案例主因，由局部回退根治 |
| **路径② 粗段不切分** | 粗段字符区间内的标点在 Layer 1 同样触发切分，各子区间时间按字符偏移线性分摊（既有实现整块 180s 单段产出）。**局限见下** |
| **路径④ 游标错位（遗留 ❸）** | 段文本改为划分而非截取，between-span 丢失/错位问题从根上消失；实测真实案例丢失的 5 字不再出现 |
| **遗留 ❷ 尾部标点重复** | 末段字符区间延伸至文末，独立 tail 逻辑不再需要 |

### 已知局限（刻意保留，不修复）

**粗段内无标点时仍产出与块等长的单段**（最长 `MAX_FORCE_ALIGN_INPUT_SECONDS` = 180s）。

曾尝试对超长粗段按 `max_segment_seconds` 在字符域二次切分，但**已撤销**，原因：

1. 粗段的时间戳本就是按字符偏移线性估算，再按段长切碎只会**制造虚假的时间精度**——下游会把估算值当作可信时间戳使用；
2. 实测会把 90s 的兜底块切成 3 个 30s 段，破坏既有「整块兜底」语义与 `speakerSummary` 统计；
3. 块长上限恒为 180s，按标点切分后通常远小于此；ASR 转写连续 3 分钟完全无句末标点属极端场景。

若后续确需处理该场景，正确方向是让 aligner 对兜底块也产出时间戳，而非在 pipeline 侧切分估算。

### 保留的能力（行为语义零变化）

- `build_segment_response` 签名、`segments[]` 字段名与类型、响应 JSON 结构
- `punctuation` / `hybrid` 两模式、`word` / `segment` 两归属模式
- 词级归属算法（`_attribute_words` / `_fill_gaps` / `_word_vote` / `_coarse_vote`）零改动
- `_split_groups` / `_split_by_speaker` / `_merge_same_speaker` 下沉为 Layer 3 段内细分
- 未定位粗段（字符区间反查失败）退化为既有「整块单段 + 块区间投票」，并保留时间域强制切分（遗留 ❶）兜底

## Impact

- Affected specs:
  - `punctuation-aware-segmentation`（切分判定从 token 匹配改为文本扫描）
  - `add-punct-split-mode-and-diarization-tuning`（遗留 ❷/❸ 随本 spec 一并消除；「整体匹配失败回退退化」Scenario 作废）
  - `fix-align-empty-items-fallback`（`coarse_char_spans` 由优化项升级为主路径依赖）
- Affected code:
  - `qwen_asr/service/pipeline.py`：新增 `_TextSpan` / `_resolve_coarse_char_spans` / `_split_text_spans` / `_map_items_to_chars` / `_assign_item_buckets` / `_absorb_orphan_buckets` / `_coarse_span_time` / `_subgroup_char_bounds` / `_group_text`；重构 `build_segment_response`；移除 `_sentence_end_boundaries` / `_extract_segment_text`
  - `qwen_asr/service/middleware.py`：仅注释更新（标注 `coarse_char_spans` 语义升级）
- 不影响：
  - `qwen_asr/inference/qwen3_forced_aligner.py`（token 清洗逻辑保持现状，由解耦架构规避）
  - 启动参数默认值与响应 JSON 结构

## ADDED Requirements

### Requirement: 文本层切分（Layer 1）

系统 SHALL 在 Layer 1 直接扫描 `full_text` 按 `_SENTENCE_END_CHARS` 划分文本段，切点位于「句末标点连续串 + 其后空白」之后；粗段字符区间的起点与终点 SHALL 强制切分。产出的字符区间 SHALL 首尾相接且完整覆盖 `[0, len(full_text))`。

#### Scenario: 句末标点切分不依赖对齐输出

- **WHEN** `full_text` 含 N 个句末标点，且存在 aligner token 在 `full_text` 中无法连续匹配
- **THEN** 切分仍按 N 个标点进行，段数回到标点量级
- **AND** 失配 token 仅损失自身的时间锚点，不牵连其余 item 的字符位置

#### Scenario: 拼接无损由构造保证

- **WHEN** 任意输入（含粗段、含失配 token）
- **THEN** `"".join(seg["text"] for seg in segments) == text`

#### Scenario: 粗段内部按标点切分

- **WHEN** 粗段文本含句末标点
- **THEN** 该粗段被切分为多个段，各段时间按字符偏移在块区间上线性分摊
- **AND** 不再产出整块 180s 不切分的超长段

### Requirement: item 字符映射局部回退（Layer 2）

系统 SHALL 在 item → 字符区间映射中，对单个 token 匹配失败记 `None` 且游标不推进，继续匹配后续 token。**禁止**任一 item 失配即全量回退。

#### Scenario: 证件号念读导致 token 失配

- **WHEN** ASR 文本含重复念读的证件号（如 `五幺七K。五幺七K。XQPZW三九。`）
- **THEN** token `KXQPZW` 失配记为 `None`，其余 item 位置完整保留
- **AND** 标点切分不受影响（自测第 20 组断言：120s 文本切出 25 段，非 30s 强切的约 5 段）

### Requirement: 孤儿文本并入相邻段

无 item 覆盖且非粗段的文本段 SHALL 并入相邻有 item 的段（优先前一段），且并入**不得跨越粗段**——否则正常段字符区间会把粗段原文一并卷入，与粗段自身 `text` 重复。

### Requirement: 未定位粗段的时间域兜底

粗段字符区间无法定位时（middleware 反查失败或 `find` 兜底失败），系统 SHALL：
1. 该粗段退化为整块单段（块区间投票 + 块 ASR 原文），保证兜底文本不丢；
2. 在 Layer 3 对 item 间隙与该粗段时间区间相交处强制切分（遗留 ❶ 的时间域兜底），保证正常段不在时间上横跨失败块。

## MODIFIED Requirements

### Requirement: 段文本游标匹配（源自 punctuation-aware-segmentation）

原需求：段文本从 `full_text` 游标截取（保留标点与空格），失败回退拼接；句末标点附前段、末段尾部追加。

修改后：段文本 = `full_text[c_start:c_end]` **直接切片**。划分替代截取，标点与空白天然含在区间内，`puncts` 列表与 tail 追加逻辑整体移除。回退拼接仅保留给「子组内无任何已映射 item」的退化场景。

**行为变更**：句末标点**之后**的空白归入前一段（既有实现丢弃该空白，导致 `join` 有损）。

### Requirement: 跨失败块边界处理（源自 add-punct-split-mode-and-diarization-tuning）

原需求：`_gap_blocked(...)` 为 True 时 `puncts[i]` 置空 + `boundaries[i]` 强制 True（时间域判定）。

修改后：粗段边界的强制切分改由 **Layer 1 字符域**精确切割（依赖 `coarse_char_spans`）；时间域判定降级为「未定位粗段」的兜底路径。

## REMOVED Requirements

- **整体匹配失败回退退化**（`add-punct-split-mode-and-diarization-tuning`）：全量回退机制随 `_sentence_end_boundaries` 一并移除，由局部回退替代。
- **遗留 ❷**（末段尾部标点与粗段尾部标点重复）：tail 逻辑移除后不再存在。
- **遗留 ❸**（交界标点丢失）：划分替代截取后不再存在。
