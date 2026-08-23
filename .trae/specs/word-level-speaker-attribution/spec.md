# 词级说话人归属（Word-Level Speaker Attribution）Spec

> v2 修订：根据实现核查修正三处前提错误——(a) 对齐器单块成功时每词必有时间戳（输入侧 mask 结构保证），原"无时间戳洞填充"对象不存在，洞重定义为"无 diarization turn 覆盖的词"；(b) 新增逐块对齐容错与失败块兜底（真实痛点：单块异常目前导致整请求 500、全空导致 segments=[]）；(c) 删除"映射回 ASR 词序列"层（build_segment_response 无独立 ASR 词序列，且 pipeline.py 受仅标准库约束，不能引入 nagisa/soynlp）——词级归属直接以 align_items 为词序列。

## Why

当前 segment 模式的说话人归属是**段级重叠投票**（段区间与 diarization 片段的重叠时长取最大者）：当两人快速交锋（换人间隙 < 0.8s 切分阈值）时，两个说话人被合进同一 segment，次要说话人的文本整段错配给主导者，`totalDuration` 随之失真。对齐器输出的本来就是字/词级时间戳，把归属粒度从"段"降到"词"，段边界可跟随说话人切换点，**显著缓解**该盲区（非根治——精度上限受 diarization 边界误差（典型 ±0.5s 量级）与重叠语音区词中点投票固有误差限制）。

同时，实机暴露的对齐链路脆弱性问题（单块对齐异常拖垮整个长音频请求、对齐全空时响应退化为空 segments）随本需求一并修复。

## What Changes

- `qwen_asr/service/pipeline.py` 新增**词级归属**纯逻辑层（仅标准库约束不变）：词中点投票归属、无 turn 覆盖词（洞）插值填充、按说话人变化切分 + 同人二次聚合（含短插话保护）
- `qwen_asr/service/middleware.py`：
  - `_run_asr_align` **逐块对齐容错**：单个对齐 batch 异常不再传播为整请求 500，失败块的词无时间戳，走块级粗粒度兜底
  - 对齐结果为空（全块失败或 merge 返回 None）时，若 diarization 可用，按 180s 块区间做粗粒度段级兜底（而非返回空 segments）
- `build_segment_response()` 新增 `speaker_attribution` 模式参数：`word`（新，默认）/ `segment`（现有段级投票，代码路径零改动）
- `qwen_asr/cli/serve.py` 新增启动参数 `--speaker-attribution {word,segment}`（默认 `word`）与 `--speaker-merge-gap`（同人相邻段合并阈值，秒，默认 `2.0`，`0` 表示不合并）
- **响应格式零变化**：`segments[]`/`speakerSummary` 结构、字段名、类型完全不变；`speakers` 字段在两种模式下**统计口径不同**（见字段语义），结构兼容但语义有差异，文档明示
- 零 GPU 开销：归属/切分/聚合均为纯后处理（毫秒级 CPU 计算），不新增模型、不动调度器/显存预算/并发架构

## Impact

- Affected specs: `add-segment-speaker-api`（其「Segment 切分与说话人归属」需求被修改为可切换的双模式）
- Affected code:
  - `qwen_asr/service/pipeline.py`（核心：词级归属纯函数 + 粗粒度兜底路径）
  - `qwen_asr/service/middleware.py`（逐块对齐容错 ~30 行 + 参数透传 ~5 行 + 失败块区间传递）
  - `qwen_asr/cli/serve.py`（新增 2 个启动参数）
  - `docs/deployment-guide.md`（参数表 + 字段口径说明 + 容错行为说明）
- 不影响：SDK 层（`qwen3_speaker_diarizer.py`、`qwen3_forced_aligner.py`）、调度器（`scheduler.py`）、显存预算、Docker 镜像、模型清单、API 请求格式

---

## ADDED Requirements

### Requirement: 词级说话人归属模式

系统 SHALL 提供 `word` 说话人归属模式（默认启用）：以 `align_items`（对齐器输出的字/词级时间戳序列）为词序列——**不引入独立 ASR 词序列，不复刻对齐器分词逻辑**（CJK 逐字/日文 nagisa/韩文 soynlp 等留在了对齐器内部）——按词时间中点投票归属说话人，再按说话人变化切分 segment。

前提事实（实现核查确认）：对齐器单块成功执行时，每个词结构上必然获得一对时间戳（时间戳 token 从输入侧 mask 提取，数量恒等于 2×词数；乱序输出由 LIS 修复强制单调）。因此词级归属**不存在"无时间戳的词"**；文本与时间戳的对应关系以 align_items 为准，`full_text` 仅用于段文本游标截取（现状不变）。

#### Scenario: 词中点落入单一说话人片段
- **WHEN** 某词的时间中点恰好落入且仅落入一个 diarization turn
- **THEN** 该词归属该 turn 的 speaker

#### Scenario: 词中点落入多个重叠 turn（重叠语音区）
- **WHEN** 词中点同时落入多个 turn（pyannote 输出的重叠片段）
- **THEN** 该词归属与**词时间区间重叠时长最大**的 turn；并列时取 speaker id 字典序最小者（保证确定性）

#### Scenario: 快速交锋正确切分（核心价值）
- **WHEN** SPEAKER_A 说到 5.0s，SPEAKER_B 于 5.2s 接话（换人间隙 0.2s < 0.8s 切分阈值），两段话均被对齐且 diarization 边界正确
- **THEN** 产出两个 segment：A 的词归 A、B 的词归 B；而非现状的单段整段归 A
- **NOTE** 该场景的切分精度受 diarization 边界误差（典型 ±0.5s）影响，换人点附近的个别词可能错归属——本方案是粒度改进，不承诺词级全对

### Requirement: 洞填充（无 diarization turn 覆盖的词）

系统 SHALL 对**词时间中点未落入任何 diarization turn** 的词（diarization 漏检、间隙、ASR 幻觉词等）按时间邻近性插值推断归属。词本身有时间戳（前提事实），仅归属缺失。

#### Scenario: 句中洞且前后同 speaker
- **WHEN** 某无覆盖词的前后已归属词同为 SPEAKER_A
- **THEN** 该洞词归属 SPEAKER_A（继承，覆盖绝大多数情形）

#### Scenario: 句中洞且前后异 speaker（洞跨换人点）
- **WHEN** 前词归 A（end=t1）、后词归 B（start=t2）且 t1 < t2
- **THEN** 取区间 `[t1, t2]` 内线性插值中点 `t=(t1+t2)/2`，落入谁的 turn 归谁；中点仍无 turn 覆盖时归前词 speaker（保守继承）

#### Scenario: 边界洞
- **WHEN** 序列开头的词无归属（无前邻居）或结尾的词无归属（无后邻居）
- **THEN** 分别后向继承（首个已归属词）或前向继承（末个已归属词）

#### Scenario: 序列内无任何已归属词
- **WHEN** 整个 align_items 序列所有词均无 turn 覆盖（如 diarization 整段漏检）
- **THEN** 全部词 speaker=None，切分仅按间隙/段长进行（间隙切分产出段级结果，段 speaker=null）

### Requirement: 逐块对齐容错（middleware）

系统 SHALL 使单个对齐 batch 的异常不传播为整请求失败：`_run_asr_align` 中对每个对齐 batch 独立捕获异常，失败 batch 对应块的词无时间戳，该块文本走**块级粗粒度兜底**；其余块正常词级归属。

#### Scenario: 单块对齐异常
- **WHEN** 1 小时音频（20 块）中第 7 块的对齐 batch 抛异常（OOM/音频异常等）
- **THEN** 请求正常返回 200：其余 19 块走词级归属；第 7 块产生粗粒度兜底段（见下一需求）；日志记录失败块序号与异常摘要
- **NOT** 整请求 500（现状行为）

#### Scenario: 客户端断连与对齐异常区分
- **WHEN** cancel_event 已置位（客户端断连）
- **THEN** 仍按现有取消语义中止（不吞取消异常）；逐块容错仅捕获对齐计算异常

### Requirement: 失败块与全空对齐的粗粒度兜底

系统 SHALL 对对齐失败（单块或全部）的音频区间提供粗粒度段级兜底，而非返回空 segments：

- **单块失败**：该块区间 `[offset, offset+块长]` 作为一个粗段，文本为该块 ASR 文本，speaker/segments 按块区间与 turns 重叠投票（与现有段级投票同公式），并在响应中不区分标记（与正常段同构）
- **align_items 全空**（全块失败或 merge 返回 None）：全部块按上述粗段方式产出；若无有效 ASR 文本块，则 segments=[]（与现状一致）
- **混合排序**：正常词级段与粗段混合产出时，`segments[]` SHALL 按 `start` 升序全局排列（粗段插入到其时间区间的正确位置，不允许简单追加导致乱序）
- **粗段不参与同人二次聚合**：粗段（无论与相邻正常段 speaker 是否相同）不与任何段合并，避免 180s 块级区间被并入正常段产生超长怪段
- **范围声明**：本容错仅覆盖对齐 batch 异常；ASR 分块生成异常（`future.result()` 抛出）**不在本次容错范围**，仍按现状整请求失败——留待后续需求
- **失真方向提示**：粗段全部时长计入 dominant speaker，minor speaker 时长被整段吞掉（段长最长 180s，失真幅度大于现状段级投票），部署文档 SHALL 写明该限制

#### Scenario: 对齐全空但 diarization 可用
- **WHEN** 所有对齐 batch 失败但 ASR 文本与 diarization 正常
- **THEN** 每个 180s 块产出一个粗段（speaker 按块区间投票，text 为块文本）；**NOT** 空segments（现状行为）
- **NOTE** 粗段粒度为块级（最长 180s），文档写明精度限制；不做块内文本切分（无词级时间戳无法定位切点）

#### Scenario: 正常段与粗段混合排序
- **WHEN** 20 块音频中第 7 块对齐失败（粗段 [1020, 1200]），其余块产出词级段（如 [990, 1015]、[1203, 1250]）
- **THEN** segments[] 按 start 升序：…、[990,1015] 词级段、[1020,1200] 粗段、[1203,1250] 词级段、…

### Requirement: 词级切分与同人二次聚合（含短插话保护）

系统 SHALL 在 word 模式下按以下顺序处理：先沿用现有间隙切分（≥ `segment_gap_threshold`）与段长上限（> `max_segment_seconds`），再在词归属 speaker 变化处追加切分点；随后将**同 speaker 且相邻段间隙 < `speaker_merge_gap`** 的段二次合并。

**段边界时间定义**：段 `start` = 首词 `start_time`，段 `end` = 末词 `end_time`（词必有时间戳，前提事实）；粗粒度兜底段的 `start`/`end` = 块区间边界。

**短插话保护**：两同人段合并前，检查两段间隙区间内是否存在**其他 speaker 的 diarization turn 覆盖 ≥ 0.3s**——存在则不合并。保护对象是**间隙里"有 turn 但无词"的真实短插话**（B 插话未被 ASR 转写或未产生对齐词时，间隙内只有 B 的 turn 而无 B 的词；若把两段 A 合并，B 的插话就无声消失了）。合并后段长超 `max_segment_seconds` 亦不合并。

#### Scenario: 同人自然停顿不分段
- **WHEN** SPEAKER_A 说话中途停顿 1.2s（≥ 0.8s 间隙阈值触发切分）后继续，`speaker_merge_gap=2.0`，且停顿区间无其他 speaker turn
- **THEN** 两段同人段被二次聚合为一段（现状会错误地切成两段）

#### Scenario: 短插话保护（未转写的真实插话）
- **WHEN** SPEAKER_A 说 0-5s 与 6-10s，SPEAKER_B 在 5.2-5.8s 插话但该插话未被 ASR 转写（无词产出，间隙内仅存在 B 的 diarization turn 覆盖 0.6s ≥ 0.3s），merge_gap=2.0
- **THEN** 两 A 段**不合并**——保留间隙对应的可疑区间，B 的插话事件不因合并而无声消失

#### Scenario: turn 边界抖动不产生碎片
- **WHEN** pyannote turn 边界抖动导致同一人连续词序列中出现个别词归属跳变，产生极短段
- **THEN** 跳变段与前后同人段间隙 < merge_gap 且间隙内无其他 speaker turn 覆盖 → 被聚合规则吸收

### Requirement: 启动参数与模式切换

系统 SHALL 提供启动参数控制归属模式，且不影响请求 API。

- `--speaker-attribution {word,segment}`：默认 `word`；`segment` 时走现有段级投票逻辑（代码路径零改动），**逐块对齐容错与粗粒度兜底在两种模式下均生效**（与归属模式正交）
- `--speaker-merge-gap <seconds>`：同人相邻段合并阈值，默认 `2.0`；`0` 表示不合并；仅 word 模式生效

#### Scenario: 回退到旧行为
- **WHEN** 运维以 `--speaker-attribution segment` 启动服务
- **THEN** 归属结果与升级前版本一致，**除**逐块容错与粗粒度兜底带来的可用性提升（对齐异常不再 500、不再空 segments）

### Requirement: word 模式下的字段语义

- `segments[].speaker`：段内词归属的 speaker（段按词归属切分，段内理论全同人；洞插值边界情况下取词数最多者，并列取 id 字典序最小者）；粗粒度兜底段为块区间投票结果
- `segments[].speakers`：**两模式口径不同**——word 模式 = 段内**词归属**出现过的 speaker 去重集合（按词数降序、id 升序）；segment 模式 = 段区间重叠 ≥ 0.1s 的说话人（现状不变）。结构兼容（均为字符串数组），语义差异在部署文档明示
- `speakerSummary`：仍按 dominant 段统计（word 模式下因段按人切分，`totalDuration` 精度提升到词粒度）
- 段文本仍从 full_text 游标截取（保留标点/空格）；粗粒度兜底段文本取该块 ASR 原文

#### Scenario: 响应格式兼容
- **WHEN** 客户端以相同请求访问升级前后服务
- **THEN** 响应 JSON 结构、字段名、类型完全一致，无需任何客户端改造

---

## MODIFIED Requirements

### Requirement: Segment 切分与说话人归属（原 add-segment-speaker-api spec）

说话人归属由「仅段级重叠投票」修改为「双模式：词级归属（默认）+ 段级投票（回退）」，由 `--speaker-attribution` 启动参数控制；`segment` 模式归属行为与原需求逐字保持不变。原需求的 0.8s 间隙切分、30s 段长上限、`speakers` 0.1s 重叠阈值（segment 模式）、`speakerSummary` 零值项覆盖等规则全部保留。

## REMOVED Requirements

（无——`segment` 模式作为回退路径完整保留，不删除任何行为）
