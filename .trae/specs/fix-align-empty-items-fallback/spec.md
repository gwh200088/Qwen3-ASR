# 对齐空 items 兜底修复 Spec v2

## Why

部署 punct2 后，执法记录仪音频出现"对话中存在清晰语句但结果中缺失该句的 segment 段"现象。根因是对齐器对某块返回空 items（`ForcedAlignResult(items=[])`）时既不抛异常也不进 `coarse_chunks`，该块文本进入 `full_text` 但无任何时间戳信息，成为"孤儿文本"。在 punctuation 模式下（不看间隙），[_extract_segment_text](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py#L468-L496) 的游标 `str.find` 跳过孤儿文本，将其并入相邻段末尾或开头，**不形成独立 segment 段**——用户表现为"这句话消失了"。

现有 [_flush_align_batch()](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L450-L469) 的 `try/except` 只捕获**整批异常**（OOM / GPU 报错），不处理**逐块空返回**；[merge_align_results](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/inference/utils.py#L525-L544) 也只跳过 `None` 不跳过空 items，无法兜底。

生产现象必然是"部分块空 items"——全部块空时 `merge_align_results` 返回 `None`，既有 [merged is None 分支](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L482-L489) 已能整体兜底。本 spec 修的正是"部分块空"这个缺口。

## What Changes

### Change 1：`_flush_align_batch()` 逐块空 items 检测

在 [_flush_align_batch()](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L450-L469) 内对齐返回后，逐块检测 `result is None` 或 `not list(result.items)`。对空 items 的块，调用 `_coarse_interval()` 加入 `coarse_chunks`，与整批异常兜底语义一致。

**结构调整**：原 `aligned.extend(_align_batch(ext, payload))` 改为先 `batch_results = _align_batch(ext, payload)`，再 `aligned.extend(batch_results)`，拿到逐块结果做空 items 检测（**不是为了避免重复调用**——原代码本来就只调用一次，重构目的是拿到逐块结果）。

### Change 2：`_run_asr_align()` 收尾时间覆盖率双保险

在 [_run_asr_align()](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L479-L489) 的 `merged = merge_align_results(...)` 之后、`if merged is None:` 之前，增加覆盖率校验分支（仅当 `merged is not None` 时执行）。

遍历 `per_chunk` 中的非空文本块，检查 `merged.items` 中是否有任一 item 的 `start_time` 落在该块 `[offset, offset+块长)` 区间内（**左闭右开**：`item.start_time` 恰好等于块终点算下一块的覆盖，不算本块）。无覆盖的块补进 `coarse_chunks`。

**去重**：Change 1 触发空 items 兜底时，记录块索引 `idx` 到集合 `coarse_idx_set`；Change 2 对未覆盖块检查 `idx in coarse_idx_set`，已在集合则跳过。块索引集合语义明确，避免浮点时间近似相等去重的脆弱性。

**已知局限**（防御性校验，可接受）：覆盖率校验只看"任一 item `start_time` 落入区间"。若前一块对齐器幻觉出超长 `end_time` 跨入本块区间，或 `start_time` 错位跨入本块，本块会被误判"已覆盖"而漏兜底。这是防御性校验的固有局限，spec 明确记录——该场景理论上不应发生（aligner 产出时间戳应在块区间内），发生时由 Change 1 的逐块空 items 检测兜底（空 items 块不会产生幻觉时间戳）。

### Change 3：pipeline tail 逻辑排除 coarse 字符区间（P0 修复）

本 spec 的 Change 1 把"尾部空 items 块"从假设场景变成现实场景——尾部空 items 块进 `coarse_chunks` 后，[pipeline.py:746-749](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py#L746-L749) 的末段尾部标点追加逻辑 `full_text[last_end:]` 会取到该块的句末标点，追加到最后一个正常段末尾；而粗段自己的 `text` 也含同样标点 → **标点重复 + 拼接无损承诺破坏**。

修复方向：tail 逻辑取标点时，**排除 coarse 块覆盖的字符区间**。

### Change 4：middleware 层提供精确 coarse 字符区间（双保险）

middleware 层有 `per_chunk = [(cwav, txt, lang, offset), ...]`，`full_text = "".join(txt for _, txt, _, _ in per_chunk)`，可**精确计算**每个 coarse 块的字符区间（块 i 的 `char_start = sum(len(per_chunk[j][1]) for j < i)`，`char_end = char_start + len(per_chunk[i][1])`），避免 pipeline 层 `find` 误匹配。

**coarse 条目 → 块索引映射**（实现关键）：`coarse_chunks` 条目来自三条路径——Change 1 逐块空 items、整批异常（[L461-463](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L461-L463)）、`merged is None` 重建（[L484-488](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L484-L488)）。其中**只有 Change 1 的块进 `coarse_idx_set`**（spec 声明既有行为不变）。因此计算 `coarse_char_spans` 时**不能用 `coarse_idx_set` 反查**（会漏掉整批异常和 `merged is None` 重建的块）。

正确映射机制：[_coarse_interval()](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L441-L443) 返回的 `start` 就是 `offset` 原值（`min()` 截断只作用于 `end`，`start` 不截断），三条路径通用。计算 `coarse_char_spans` 时对每个 `coarse_chunks` 条目，用 `start == per_chunk[i][3]`（offset）反查块索引 `i`，再按公式算字符区间。

`build_segment_response` 新增可选参数 `coarse_char_spans: Optional[List[Tuple[int, int]]]`，与 `coarse_chunks` 一一对应。middleware 层在 `_run_asr_align` 返回时计算并传入；现有调用点不传时为 `None`，pipeline 自己 `find` 兜底。

**find 兜底的乱序脆弱性**（实现提示）：Change 2 补进的块追加在 `coarse_chunks` 末尾，列表不保证时间升序（如块 5 被 Change 1 兜底、块 3 被 Change 2 补进 → 降序）。`coarse_char_spans=None` 时 pipeline 游标 `find` 在乱序 + 重复文本下可能定位失败或错位。生产路径走 Change 4 精确区间不受影响（spec 已声明风险可接受），但 pipeline find 兜底实现时应：先按时间 `start` 排序再 `find`，或各条 `coarse_text` 独立从 0 `find` 并防御重叠区间。

### Change 5：兜底日志可定位

Change 1/2 触发兜底时输出 WARNING 日志，包含块序号、`[start, end]` 时间区间、文本长度（Change 1）或"未被 item 覆盖"（Change 2），便于部署机定位。

## Impact

- Affected specs:
  - `punctuation-aware-segmentation`（标点感知分段的文本拼接前提：所有非空文本块必须有 align item 或 coarse 段覆盖）
  - `add-punct-split-mode-and-diarization-tuning`（punctuation 模式下游标匹配跳过孤儿文本的副作用放大）
- Affected code:
  - [qwen_asr/service/middleware.py](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py) — `_flush_align_batch()`、`_run_asr_align()` 收尾段、`_run_asr_align()` 返回值 4 元组改 5 元组（新增 `coarse_char_spans`）、[L857 解包点](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L857) 同步改 5 元组解包、[L861 生产调用点](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L861) 透传新参数
  - [qwen_asr/service/pipeline.py](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py) — `build_segment_response()` 新增 `coarse_char_spans` 参数 + [tail 逻辑 L746-749](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py#L746-L749) 改
- 不影响：
  - [qwen_asr/inference/utils.py](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/inference/utils.py) `merge_align_results` 行为不变（仍跳过 `None`，空 items 由 middleware 层兜底转 coarse）
  - [qwen_asr/inference/qwen3_forced_aligner.py](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py) 不改 aligner 本身
  - pipeline.py 的 51 处 self_test 调用点（`coarse_char_spans` 默认 None，pipeline 自己 find 兜底；grep "build_segment_response(" 得 52 处，其中 1 处是 [def 定义行 L571](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py#L571)，实际调用 51 处）

## ADDED Requirements

### Requirement: 对齐逐块空 items 兜底（Change 1）

系统 SHALL 在 `_flush_align_batch()` 内对齐批次返回后，逐块检测 `ForcedAlignResult.items` 是否为空。对 `result is None` 或 `not list(result.items)` 的块，系统 SHALL 将其 `(文本, offset, offset+块长)` 加入 `coarse_chunks`，与整批异常兜底语义一致，并记录块索引到 `coarse_idx_set` 供 Change 2 去重。

#### Scenario: 单块对齐返回空 items

- **WHEN** aligner 对某块返回 `ForcedAlignResult(items=[])`（不抛异常）
- **THEN** 该块的 `(文本, offset, offset+块长)` 进入 `coarse_chunks`，块索引进入 `coarse_idx_set`，输出 WARNING 日志（块序号 + 文本长度 + 时间区间）
- **AND** 该块的文本在最终 segments 中以**独立 coarse 段**出现，不并入相邻段

#### Scenario: 整批异常仍走既有兜底

- **WHEN** aligner 整批抛异常（OOM / GPU 报错）
- **THEN** 批内全部块走 `coarse_chunks`（既有行为不变），逐块空 items 检测不重复触发（`except` 分支内不执行逐块检测）
- **AND** 批内全部块的索引加入 `coarse_idx_set`（供 Change 2 覆盖率双保险去重——否则部分成功 + 整批异常场景下，Change 2 会判定这些块"未被覆盖"且 `idx not in coarse_idx_set` 为真，重复补进 `coarse_chunks`，导致文本重复）
- **AND** 覆盖率双保险不重复兜底这些块（`idx in coarse_idx_set` 检查跳过）

### Requirement: 对齐覆盖率双保险（Change 2）

系统 SHALL 在 `_run_asr_align()` 收尾处，对所有非空文本块做时间覆盖率校验：检查 `merged.items` 中是否有任一 item 的 `start_time` 落在该块 `[offset, offset+块长)` 区间内（**左闭右开**：`item.start_time == offset+块长` 算下一块的覆盖，不算本块）。无覆盖的块 SHALL 补进 `coarse_chunks`（若该块索引已在 `coarse_idx_set` 则跳过，防重复）。

#### Scenario: aligner 漏检某块

- **WHEN** aligner 对块 A 返回非空 items 但所有 item 的 `start_time` 都不在块 A 的 `[offset, offset+块长)` 区间内
- **THEN** 块 A 补进 `coarse_chunks`，输出 WARNING 日志（块区间 + "未被 item 覆盖"）
- **AND** 后续 `build_segment_response` 为该块生成独立 coarse 段

#### Scenario: merged 为 None

- **WHEN** 全部块对齐失败或逐块均空 → `merged is None`
- **THEN** 走既有"全部非空文本块整体兜底"分支（既有逻辑不变），覆盖率双保险跳过（`merged is None` 时不执行覆盖率校验）

#### Scenario: 幻觉时间戳跨块（已知局限）

- **WHEN** 前一块对齐器幻觉出超长 `end_time` 或 `start_time` 错位跨入本块区间
- **THEN** 本块可能被误判"已覆盖"而漏兜底——这是防御性校验的固有局限，spec 明确记录
- **AND** 该场景理论上不应发生（aligner 产出时间戳应在块区间内）；若发生且该块是空 items，由 Change 1 的逐块空 items 检测兜底

### Requirement: pipeline tail 逻辑排除 coarse 字符区间（Change 3）

系统 SHALL 在 [pipeline.py:746-749](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py#L746-L749) 的末段尾部标点追加逻辑中，排除 coarse 块覆盖的字符区间。tail 取 `full_text[last_end:]` 中的句末标点时，SHALL 跳过落在任一 coarse 块字符区间 `[char_start, char_end)` 内的字符。

#### Scenario: 尾部空 items 块标点不重复

- **WHEN** 空 items 块在音频尾部，进了 `coarse_chunks`，其文本含句末标点
- **THEN** tail 逻辑跳过该块的字符区间，不把它的标点追加到最后一个正常段
- **AND** 粗段自己的 `text` 含完整原文含标点
- **AND** `"".join(seg["text"] for seg in segments) == full_text` 拼接无损成立（标点不重复）

#### Scenario: 无 coarse 块时 tail 行为不变

- **WHEN** 没有 coarse 块（对齐全部成功）
- **THEN** tail 逻辑行为不变（无字符区间需排除，取 `full_text[last_end:]` 全部句末标点）

### Requirement: middleware 层提供精确 coarse 字符区间（Change 4）

系统 SHALL 在 `_run_asr_align()` 返回时，计算每个 coarse 块在 `full_text` 中的精确字符区间 `[char_start, char_end)`，通过 `coarse_char_spans` 返回。`build_segment_response` 新增可选参数 `coarse_char_spans: Optional[List[Tuple[int, int]]]`，与 `coarse_chunks` 一一对应。

#### Scenario: middleware 提供精确字符区间

- **WHEN** `_run_asr_align` 产出 coarse 块
- **THEN** 基于 `per_chunk` 的文本长度精确计算每个 coarse 块的 `[char_start, char_end)`，通过 `coarse_char_spans` 传给 `build_segment_response`
- **AND** pipeline tail 逻辑优先用 `coarse_char_spans` 排除字符区间

#### Scenario: 调用点未提供字符区间（双保险兜底）

- **WHEN** `coarse_char_spans` 为 `None`（self_test 调用或旧调用点）
- **THEN** pipeline 自己用 `coarse_text` 在 `full_text` 中游标 `find` 确定字符区间，tail 逻辑用 find 结果排除
- **AND** find 误匹配风险存在（coarse_text 在 full_text 中重复出现时），但作为双保险兜底可接受

### Requirement: 兜底日志可定位（Change 5）

系统 SHALL 在 Change 1/2 触发兜底时输出 WARNING 日志：
- Change 1（空 items 兜底）：块序号 + 文本长度 + `[start, end]` 时间区间
- Change 2（覆盖率双保险）：块序号 + `[start, end]` 时间区间 + "未被 item 覆盖"

## MODIFIED Requirements

### Requirement: 逐块对齐容错（既有）

逐块对齐容错的失败语义扩展：除了"整批异常 → 批内全部兜底"和"全空 → 全部兜底"外，新增"单块返回空 items → 该块兜底"（Change 1）和"merged 生成后单块未被覆盖 → 该块补兜底"（Change 2）两个分支。`coarse_chunks` 的语义从"异常兜底"扩展为"任何无 item 时间戳的块兜底"，统一作为 `build_segment_response` 的粗粒度段来源。

### Requirement: 末段尾部句末标点追加（既有，v3）

tail 逻辑从"取 `full_text[last_end:]` 全部句末标点"修改为"取 `full_text[last_end:]` 中**不在 coarse 块字符区间内**的句末标点"。新增 `coarse_char_spans` 参数作为主字符区间来源，`None` 时 pipeline 自己 `find` 兜底。

## REMOVED Requirements

无。
