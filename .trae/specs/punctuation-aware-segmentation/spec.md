# 标点感知 Segment 切分（Punctuation-Aware Segmentation）Spec

> v3 修订（用户审阅反馈）：❻ 跨失败块（coarse）边界的 puncts 置空——失败块前后相邻词的 between-span 含整块失败文本（最长 180s），其句末标点若拼入 puncts 会给前段追加垃圾后缀并与粗段原文标点重复；❼ 末段追加尾部句末标点——full_text 末词匹配终点之后的句末标点不属于任何 between-span，不追加则"拼接无损"承诺对末段失效。
>
> v2 修订（用户审阅反馈）：❶ 句末标点集合补 ASCII 句点 `.`（英文句末标点占绝大多数，缺失则英文场景标点切分失效）；❷ 明示同人二次聚合在默认参数下不再触发（事实死代码，非删除）；❸ 新增「句末标点附前段末尾」规则（旧版"现状已如此"表述错误——现状段内句号保留，恒切后句号落入 between-span，须显式定义归属）；❹ 新增 `--punctuation-split` 运行时开关（消除"无回退路径"风险）；❺ 补全量回退理由。

## Why

实测暴露两类切分问题（用户提供的执法记录仪音频输出，经当前代码核查确认仍会复现）：

1. **句中切碎 / 单字成段**：切分完全依赖时间间隙（≥ 0.8s 即切），无视 ASR 文本中已有的句法边界。示例：句中 0.88s 停顿把"想|负责"劈开、"身份证"被劈成"身|份证"跨两段；更极端的是一段疑似非语音区（时间戳精确等距 1.943s，对齐伪影特征）产生 9 个连续单字段。当前 word 模式的同人聚合（merge_gap 2.0s）只能愈合"同说话人"的切分——词归属为 None（diarization 无 turn 覆盖，单字示例即此情况）、间隙 ≥ 2.0s、或 segment 模式下，碎片依旧。
2. **一段多人**：word 模式（当前默认）已按词归属切分大幅缓解，但 diarization 漏检（归属 None，退化为纯间隙切分）或 `segment` 模式下，快问快答（"说号就行。啊？"）仍会合进一段多人。

ASR 输出文本自带标点（句号/问号/感叹号是明确的句法边界），当前管道完全未利用。引入**句末标点硬切分 + 无标点处间隙阈值提高**可同时根治两类问题。

**用户确认的产品决策**：
- 切分粒度取**仅句末标点**（逗号/顿号保留段内，符合"一句话一个分段"）；
- 无句末标点处的间隙切分阈值由 0.8s **提高到 2.0s**（覆盖示例中 0.88s/1.943s 两类碎片间隙，并与同人合并阈值 2.0s 对齐）；
- **切分处句末标点附前段末尾**（客户端拼接 `segments[].text` 与 `text` 标点无损）；
- **提供 `--punctuation-split` 运行时开关**（默认开启，线上误切可即时关闭）。

## What Changes

- `qwen_asr/service/pipeline.py`：
  - **NEW** `_sentence_end_boundaries()`：基于游标匹配（与 `_extract_segment_text` 同一 greedy `find` 语义）把 align_items 全局匹配到 `full_text`，相邻 item 匹配区间之间的 between-span 含句末标点（**`。！？；.!?;`** 及换行符）→ 该边界为**硬边界**；返回 `(boundaries, puncts)`——`puncts[i]` 为边界 i 处 between-span 中的句末标点字符序列（如 `"。"`、`"？！"`、`""`），供段文本标点追加；**跨失败块边界**（边界时间间隙区间与任一 coarse 块时间区间相交，判断逻辑同 `_gap_blocked`）的 `puncts[i]` 置空串——该 between-span 含整块失败文本（最长 180s），标点拼入会产生垃圾后缀并与粗段原文标点重复；切分照常（跨失败块间隙 ≥ 失败块时长 ≥ 2.0s，间隙规则必然切分）；任一 item 匹配失败 → 全部回退为非硬边界（`boundaries` 全 False、`puncts` 全空串，纯间隙行为）
  - **MODIFIED** `_split_groups()`：新增可选参数 `hard_boundaries`，切分条件改为"硬边界处恒切分（无视间隙）**或** 间隙 ≥ 阈值 **或** 段长 > 上限"；word 模式的 `_split_by_speaker()` 透传该参数，说话人变化切分逻辑不变（无标点处照切）
  - **MODIFIED** `_merge_same_speaker()`：同人相邻段之间的边界为句末标点硬边界时**不合并**（分组结果连续覆盖全部词，按累计词数反查全局边界索引）
  - **MODIFIED** `build_segment_response()`：签名默认 `segment_gap_threshold` 0.8 → **2.0**，语义收窄为"无句末标点处的间隙切分阈值"；新增 `punctuation_split: bool = True` 参数（False 时跳过硬边界计算，纯间隙切分）；**段文本构建（两种归属模式均生效）**：① 段末边界为硬边界时，`puncts[边界]` 追加到前段 `text` 末尾（连续句末标点如"？！"全量追加；空格/引号等非句末标点字符不追加；跨失败块边界 puncts 已置空故不追加）；② **末段追加尾部句末标点**——全局匹配成功时，full_text 末词匹配终点之后的句末标点字符追加到末段 `text` 末尾（末词之后的标点不属于任何 between-span，不追加则丢失）；粗段（coarse_chunks）不参与标点追加（其 text 取块 ASR 原文，含自身标点）
- `qwen_asr/cli/serve.py`：`--segment-gap-threshold` 默认 0.8 → **2.0**（help 更新为新语义）；**NEW** `--punctuation-split {on,off}`（默认 `on`）
- `qwen_asr/service/extensions.py` + `middleware.py`：`ExtensionState` 新增 `punctuation_split` 字段并透传至 `build_segment_response`（与既有 `segment_gap_threshold` 等参数同一透传链）
- `docs/deployment-guide.md`：参数语义、默认行为变化、回退组合（见开关需求）、已知 trade-off 说明
- **响应格式零变化**：`segments[]`/`speakerSummary` 结构、字段名、类型完全不变；仅切分位置、段数与段文本标点归属变化

## Impact

- Affected specs:
  - `add-segment-speaker-api`（「Segment 切分与说话人归属」的切分规则被修改）
  - `word-level-speaker-attribution`（「词级切分与同人二次聚合」的切分前提与聚合边界被修改）
- Affected code:
  - `qwen_asr/service/pipeline.py`（核心：标点边界计算 + 切分/聚合接入 + 段文本标点追加，约 +100/-15 行）
  - `qwen_asr/cli/serve.py`（1 个默认值 + 1 个新参数）
  - `qwen_asr/service/extensions.py`、`qwen_asr/service/middleware.py`（`punctuation_split` 存储与透传，~5 行）
  - `docs/deployment-guide.md`（参数表 + 行为说明）
- **默认行为变化（明示）**：`--segment-gap-threshold` 默认值 0.8 → 2.0 且语义收窄（仅无句末标点边界）；无标点输出（ASR 不出标点的语言/模型）的段会变长（仍受 30s 上限约束）。显式传参的部署保持其值不受默认值变化影响
- **同人二次聚合默认失效（明示，行为推导非删除）**：默认参数（gap 2.0 / merge_gap 2.0）下 `_merge_same_speaker` 不再合并任何段——间隙切分边界（gap ≥ 2.0）天然不满足合并条件 `gap < 2.0`，硬边界被阻断，段长切分合并后必超上限，说话人变化段异人。`--speaker-merge-gap` 仅在显式配置为**大于** gap 阈值时具有实际效果（如旧配置组合 gap 0.8 + merge_gap 2.0，或 `--punctuation-split off --segment-gap-threshold 0.8` 的回退部署）。机制与参数保留，不删除
- **回退路径**：`--punctuation-split off` 关闭标点切分（纯间隙行为，阈值仍由 `--segment-gap-threshold` 控制）；完整旧行为 = `--punctuation-split off --segment-gap-threshold 0.8`
- 不影响：调度器/显存预算、SDK 层、Docker 镜像、API 请求格式、粗段兜底逻辑

---

## 关键设计决策

### 句末标点集合与硬边界机制
- 句末标点：`。！？；` + ASCII **`.!?;`**（含句点 `.`——英文句末标点占绝大多数，缺失则英文场景标点切分失效）+ 换行符。逗号（`，、,`）、冒号等**不是**切分点（保留段内）
- 实现为**硬边界**：先按句末标点把 align_items 序列切成"句块"，块内再走既有间隙/段长/说话人切分与聚合；聚合（同人合并）**永不跨越**硬边界。比"标点仅作为切分条件之一"更自洽——否则同人聚合会把标点切开的段合回去，规则互相抵消
- **段文本句末标点附前段末尾**：硬边界处 between-span 中的句末标点字符（含连续标点如"？！"）追加到前段 `text` 末尾，非句末标点字符（空格/引号）不追加。块内逗号/空格仍由游标截取自然保留。（现状对比：间隙 < 0.8 合并时句号在段内保留——恒切后若无此规则，句号将丢失于 between-span，`segments[].text` 拼接缺标点）
- **跨失败块边界 puncts 置空（v3）**：失败块（coarse）无 align items，其前后相邻词的 between-span 包含整块失败文本（最长 180s）——内含句末标点若拼入 puncts，会给前段追加垃圾后缀（如"。。？。"）且与粗段原文中的同一批标点重复。规则：边界时间间隙区间 `[前词.end, 后词.start]` 与任一 coarse 块时间区间相交 → `puncts[i] = ""`（判断逻辑与聚合的粗段阻断 `_gap_blocked` 同一方式，时间域而非文本域）；**切分照常**——跨失败块间隙 ≥ 失败块时长（180s 分块）≥ 2.0s，间隙规则必然切分，硬边界判定值不影响结果
- **末段尾部句末标点追加（v3）**：full_text 以句末标点结尾时（如"说号就行。啊？"），末词匹配终点之后的"？"不属于任何 between-span。规则：全局匹配成功时，full_text 末词匹配终点之后的句末标点字符追加到末段 `text` 末尾；匹配失败回退 / `punctuation_split=False` 时不追加
- **拼接无损的完整口径**：匹配成功时 `"".join(segments[].text) == text`（`text` 全文字段）全域成立，含混合粗段场景——正常段携带 between-span 标点 + 末段尾部标点，粗段 text 取块 ASR 原文（含自身全部标点），跨失败块边界标点已置空不重复，按 `start` 排序拼接恰好还原全文
- 匹配失败全量回退的理由：失败 item 后游标位置不确定，若部分保留已匹配前缀的边界，后续 item 从未推进游标继续 `find` 可能错位，导致 between-span 计算错误（标点误判）；匹配失败本身罕见（align item 文本源自同一 ASR 输出），全量回退语义保守且可预测

### 无标点间隙阈值 2.0s 与同人聚合的关系（行为推导，明示）
- 语义：相邻词之间**无**句末标点时，静音间隙 ≥ 2.0s 才切分（原 0.8s）。有句末标点处无视间隙恒切分
- 取 2.0 依据：(a) 覆盖用户示例两类碎片间隙（0.88s / 1.943s 均不再切）；(b) 与 `speaker_merge_gap`（默认 2.0）对齐——旧 word 模式下同人 0.8~2.0s 间隙"先切后合"，本方案"根本不切"，净效果一致；(c) None 归属与 segment 模式同样受益（旧版这两条路径无聚合兜底）
- **推导结论（运维须知）**：上述 (b) 的对齐意味着默认参数下"先切后合"的合半边永不触发（切分需 gap ≥ 2.0，合并需 gap < 2.0，条件互斥）；叠加硬边界阻断、段长上限、说话人变化异人，`_merge_same_speaker` 在默认配置下**不合并任何段**。这不是缺陷而是阈值对齐的必然结果——碎片段在切分阶段就不产生，无需聚合愈合。`--speaker-merge-gap` 保留给显式调低 gap 阈值的部署（此时聚合恢复活性）
- 单字成段示例回归验证：9 字间隙均 1.943s < 2.0 且无标点 → 聚合为一段（span 33s > 30s 触发一次强切 → 两段），不再逐字成段

### 标点切分开关（--punctuation-split）
- 默认 `on`；`off` 时跳过硬边界计算（`_sentence_end_boundaries` 不调用），切分退化为纯间隙/段长（+word 模式说话人变化），阈值仍由 `--segment-gap-threshold` 控制（默认 2.0）
- `off` 不改变段文本截取行为（`_extract_segment_text` 原样）；句末标点附前段仅在 `on` 时发生（off 时句号按现状语义随间隙合并自然保留段内）
- 运维回退组合：仅关标点切分 → `--punctuation-split off`；完整旧行为 → `--punctuation-split off --segment-gap-threshold 0.8`（此组合下同人聚合亦恢复旧活性）

### 已知 trade-off（文档明示）
- **英文缩写句点误切**：`Mr. Smith` 类缩写句点被视为句末标点 → 误切（`Mr` token 匹配成功 + `Smith` 匹配成功，between-span `". "` 含句点）。**词内含句点的 token 不受影响**：`U.S.A.`/`3.14` 经对齐器 `clean_token` 清洗为 `USA`/`314`，在 full_text 中 `find` 失败 → 整序列走匹配失败回退（无标点切分）。ASR 语音输出中缩写句点罕见，接受；线上出现可 `--punctuation-split off` 即时关闭
- **短插话保护区间收窄**：间隙 0.8~2.0s 且无标点处不再切分，原"未转写短插话保护"（间隙内他人 turn ≥ 0.3s 阻止合并）在该区间不再触发——间隙里的未转写插话不再以"段间隙"形式保留。保护仍在实际发生的切分（间隙 ≥ 2.0s、标点边界、说话人变化）后的合并判定中生效
- **无标点文本段变长**：ASR 不输出标点时全部边界按 2.0s 阈值（原 0.8s），段更长，仍受 30s 上限约束
- **匹配失败回退**：item 文本在 full_text 中找不到（`_extract_segment_text` 回退拼接的场景）→ 无标点信息 → 全部按无标点间隙规则（2.0s）；理由见「句末标点集合与硬边界机制」
- 说话人变化切分**不受标点影响**：无标点处照切（word 模式核心价值，快速交锋场景）

---

## ADDED Requirements

### Requirement: 句末标点硬切分

系统 SHALL 基于 ASR 文本中的句末标点（`。！？；.!?;` 及换行）切分 segment：相邻对齐词在 `full_text` 中的匹配区间之间存在句末标点时，该边界为硬边界——恒切分（无视时间间隙），且同人二次聚合不得跨越。

#### Scenario: 快问快答正确切分（中文）
- **WHEN** 同一音频内"说号就行。啊？"两个短句间隙仅 0.3s（< 间隙阈值）
- **THEN** 句号处切分为两段，各自独立归属说话人；**NOT** 合成一段多人

#### Scenario: 英文句末句点切分
- **WHEN** 英文音频"Nice to meet you. See you tomorrow."两句间隙 < 间隙阈值
- **THEN** 句点处切分为两段（ASCII `.` 属于句末标点集合）

#### Scenario: 逗号不切分
- **WHEN** 相邻词之间仅有逗号/顿号（"不记分，罚款二十"）且间隙小于阈值
- **THEN** 不因逗号切分（逗号保留在段文本内）

#### Scenario: 同人聚合不跨越硬边界
- **WHEN** 同一说话人快速连续说"今天不错。明天更好。"（句号处间隙 0.2s < merge_gap 2.0）
- **THEN** 两段（句号硬边界阻断聚合）；而无标点的同人自然停顿（间隙 < 2.0s）本就不切分，无需聚合

#### Scenario: 标点信息缺失回退
- **WHEN** 任一 align item 文本在 full_text 中游标匹配失败
- **THEN** 全部边界按无标点规则处理（间隙 ≥ 2.0s 切分），不抛异常

### Requirement: 切分处句末标点附前段末尾

系统 SHALL 在硬边界切分时，把边界 between-span 中的句末标点字符（含连续标点如"？！"）追加到前段 `text` 末尾；空格/引号等非句末标点字符不追加。**跨失败块边界**（边界时间间隙区间与任一 coarse 块时间区间相交）SHALL 不追加（puncts 置空），切分照常。**末段** SHALL 追加 full_text 末词匹配终点之后的尾部句末标点（全局匹配成功时）。

#### Scenario: 段文本标点无损
- **WHEN** full_text 为"说号就行。啊？说吧"，句号与问号处各产生硬边界切分
- **THEN** 三段 text 为"说号就行。"、"啊？"、"说吧"；`"".join(segments[].text)` 与 `text` 字段一致

#### Scenario: 末段尾部标点追加
- **WHEN** full_text 为"说号就行。啊？"（以句末标点结尾），末词"啊"匹配终点在"？"之前
- **THEN** 末段 text 为"啊？"（尾部"？"追加到末段，不丢失）

#### Scenario: 连续句末标点全量追加
- **WHEN** 边界 between-span 为"？！ "（连续问叹号 + 空格）
- **THEN** 前段 text 追加"？！"，空格不追加

#### Scenario: 跨失败块边界不追加
- **WHEN** 词 A（成功块末词）与词 B（下一成功块首词）之间隔着一个对齐失败块（块文本含多个句末标点），A/B 为 items 序列相邻词
- **THEN** 该边界处切分照常发生（间隙 ≥ 失败块时长 ≥ 2.0s）；前段 text **无**垃圾标点后缀（如"。。？。"）；失败块标点仅出现在粗段 text（块 ASR 原文）中，无重复

#### Scenario: 混合场景拼接无损
- **WHEN** 正常词级段与粗段混合产出（含跨失败块边界与全文尾部标点）
- **THEN** 按 `start` 排序后 `"".join(segments[].text) == text`（正常段携带 between-span 标点 + 末段尾部标点，粗段含自身原文标点，跨块标点不重复）

### Requirement: 无句末标点处的间隙切分阈值

系统 SHALL 将无句末标点边界的间隙切分阈值默认设为 **2.0s**（`--segment-gap-threshold`，可配），句末标点处无视间隙恒切分；段长上限（`--max-segment-seconds`，默认 30s）强切不受影响。

#### Scenario: 句中停顿不再切碎（问题 1 核心场景）
- **WHEN** 句中 0.88s 自然停顿且停顿处无句末标点（"…副驾驶。想负责那搞…"，"想|负责"之间）
- **THEN** 不切分（原 0.8s 阈值会切碎）；词归属说话人变化处仍照切（word 模式）

#### Scenario: 单字碎片聚合
- **WHEN** 连续单字词间隙 1.943s（< 2.0s）且无句末标点（用户单字成段示例）
- **THEN** 聚合为一段；聚合后 span 超过 30s 时按段长上限强切（如 33s → 两段），不再逐字成段

#### Scenario: 长静音仍切分
- **WHEN** 无句末标点边界处静音间隙 ≥ 2.0s（不同话语、无标点输出）
- **THEN** 切分（阈值含边界值 2.0）

#### Scenario: 段长强切保留
- **WHEN** 无任何标点的长块（如对齐伪影产生的 33s 连续词序列）
- **THEN** 段长 > 30s 处强切，行为与现状一致

### Requirement: 标点切分开关

系统 SHALL 提供 `--punctuation-split {on,off}` 启动参数（默认 `on`）：`off` 时跳过句末标点硬边界计算，切分退化为纯间隙/段长（word 模式说话人变化切分保留），间隙阈值仍由 `--segment-gap-threshold` 控制。

#### Scenario: 线上误切即时关闭
- **WHEN** 运维发现缩写句点等误切，以 `--punctuation-split off` 重启服务
- **THEN** 标点切分不再发生（无需降级版本）；段文本句末标点恢复现状语义（随间隙合并自然保留段内）

#### Scenario: 完整旧行为回退
- **WHEN** 以 `--punctuation-split off --segment-gap-threshold 0.8` 启动
- **THEN** 切分/聚合行为与升级前版本一致（间隙 0.8s 切分 + 同人 merge_gap 2.0 聚合恢复活性）

---

## MODIFIED Requirements

### Requirement: Segment 切分与说话人归属（原 add-segment-speaker-api / word-level-speaker-attribution）

切分规则由「间隙 ≥ 0.8s 或段长 > 30s（word 模式追加：说话人变化切分 + 同人 merge_gap 聚合）」修改为：

1. **句末标点硬边界 → 恒切分**（`--punctuation-split on` 时，默认；两种归属模式均生效；聚合不跨越；句末标点附前段段尾）；
2. 无句末标点边界 → 间隙 ≥ `segment_gap_threshold`（默认 **2.0**，原 0.8）切分；
3. 段长 > `max_segment_seconds`（默认 30s）强切（不变）；
4. word 模式说话人变化切分（不变，无标点处照切）；
5. 同人聚合：机制保留（含短插话保护、粗段阻断），**追加硬边界阻断**；**默认参数（gap 2.0 / merge_gap 2.0）下不再触发任何合并**（间隙切分与合并条件互斥所致，见设计决策），仅显式配置 merge_gap > gap 阈值时具有实际效果。

`segment` 模式归属投票、`speakers` 口径（两模式差异）、`speakerSummary` 口径、段文本游标截取（块间句末标点附前段、块内标点保留）、粗段兜底均不变。

### Requirement: qwen-asr-serve 入口参数

`--segment-gap-threshold` SHALL 默认 **2.0**（原 0.8），语义为"相邻词之间无句末标点时触发切分的静音间隙阈值（秒，含边界值）；句末标点处恒切分"；**NEW** `--punctuation-split {on,off}` SHALL 默认 `on`（off 时纯间隙切分，完整旧行为回退组合为 `off` + `--segment-gap-threshold 0.8`）；其余参数不变。

## REMOVED Requirements

（无——间隙切分、同人聚合、标点切分开关路径完整保留，仅阈值默认值与切分条件扩展）
