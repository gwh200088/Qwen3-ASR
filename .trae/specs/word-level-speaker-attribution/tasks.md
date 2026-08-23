# Tasks

- [ ] Task 1: `pipeline.py` 词级归属核心（纯函数，仅标准库）
  - [ ] SubTask 1.1: `_attribute_words()`——词中点投票：中点落入的 turn 即归属；落入多个重叠 turn 时取与词时间区间重叠最大者（并列取 id 字典序最小）；中点无 turn 覆盖标记为待填充洞。**词序列直接用 align_items**（每词必有时间戳，见 spec 前提事实；不做 ASR 词序列映射，不引入 nagisa/soynlp）
  - [ ] SubTask 1.2: `_fill_gaps()`——洞填充四规则（洞=无 turn 覆盖词，非无时间戳词）：前后同人继承 / 前后异人取 `[前.end, 后.start]` 中点插值（仍无覆盖归前词）/ 开头洞后向继承 / 结尾洞前向继承；全序列无已归属词时全部 None

- [ ] Task 2: word 模式段构建与 `build_segment_response` 集成
  - [ ] SubTask 2.1: `_split_by_speaker()`——在 `_split_groups` 间隙/段长切分基础上按词归属 speaker 变化追加切分点；段 start/end = 首末词时间戳（spec 已定义）
  - [ ] SubTask 2.2: 同人二次聚合——同 speaker 相邻段且间隙 < `speaker_merge_gap` 合并；**短插话保护**：间隙区间内存在其他 speaker turn 覆盖 ≥ 0.3s 不合并；合并后段长超 `max_segment_seconds` 不合并；`merge_gap=0` 不聚合
  - [ ] SubTask 2.3: `build_segment_response(..., speaker_attribution="word", speaker_merge_gap=2.0, coarse_chunks=None)`：word 路径 = 归属→填充→切分→聚合；`segment` 模式代码路径零改动；`coarse_chunks`（失败块 `(text, start, end)` 列表）非空时产出块级粗段（块区间投票，复用段级投票公式）；**混合产出时 segments 按 start 升序全局排列**（粗段插入正确时间位置）；**粗段不参与同人二次聚合**
  - [ ] SubTask 2.4: 段文本游标截取复用 `_extract_segment_text`；粗段文本取块 ASR 原文；`text` 全文不变

- [ ] Task 3: `middleware.py` 逐块对齐容错（真实痛点修复，与模式正交、两种模式均生效）
  - [ ] SubTask 3.1: `_run_asr_align` 对齐循环改造：每个 batch 独立 try/except（仅捕获对齐计算异常；cancel_event 置位时 RuntimeError 照常中止不被吞）；失败 batch 记录该块 `(text, offset, offset+块长)` 进 coarse 列表并 logger.warning（块序号+异常摘要），继续后续块
  - [ ] SubTask 3.2: 返回签名扩展：`(full_text, merged_lang, merged, coarse_chunks)`；align 全空（merged=None 且无成功块）时 coarse 覆盖全部非空文本块；`_run_segment` 将 coarse_chunks 透传给 `build_segment_response`
  - [ ] SubTask 3.3: 回归确认：**segment 模式下**正常路径（无失败块）响应与现状逐字节等价（coarse_chunks=None）；word 模式下无失败块时 coarse 路径不介入（归属/切分变化即方案目的本身）；ASR 分块生成异常不在容错范围（spec 已声明，整请求失败为现状行为）

- [ ] Task 4: `serve.py` 参数与透传
  - [ ] SubTask 4.1: 新增 `--speaker-attribution {word,segment}`（默认 `word`，choices 校验）与 `--speaker-merge-gap`（默认 `2.0`，float，负值启动报错），纳入扩展参数剥离列表
  - [ ] SubTask 4.2: `ExtensionState` 存储两参数并透传至 `build_segment_response` 调用点

- [ ] Task 5: `self_test` 扩展与本地验证（无 GPU 依赖）
  - [ ] SubTask 5.1: word 模式断言组：中点投票（单一/重叠 turn/无覆盖）、洞填充四规则、全 None 序列、快速交锋切分（0.2s 换人 → 两段）、同人停顿聚合（1.2s 停顿 merge_gap=2.0 → 一段）、短插话保护（间隙有 B turn ≥0.3s → 不合并）、merge_gap=0 不聚合、段边界=首末词时间戳
  - [ ] SubTask 5.2: 容错/兜底断言组：coarse_chunks 单块失败 → 正常段+粗段混合、全空 → 全粗段、coarse_chunks=None → segment 模式下与现状等价、混合时 segments 按 start 升序（粗段位置正确）、粗段不参与聚合、segment 模式回归（既有断言全部原样通过）
  - [ ] SubTask 5.3: 本地运行 `python examples/example_segment_api.py --self-test` 全量通过

- [ ] Task 6: 文档更新
  - [ ] SubTask 6.1: `docs/deployment-guide.md`：参数表新增 `--speaker-attribution` / `--speaker-merge-gap`；§8.3 补充 word 模式 `speaker`/`speakers` 口径（**与 segment 模式口径差异明示**）、快速交锋行为变化、精度预期（受 diarization 边界 ±0.5s 限制）；对齐逐块容错与粗段兜底行为说明（含粗段 180s 粒度精度限制、**粗段整段时长计入 dominant speaker 对 totalDuration 的放大失真**、ASR 生成异常不在容错范围）

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2（coarse_chunks 接口先定）
- Task 4 depends on Task 2
- Task 5 depends on Task 1, 2, 3, 4
- Task 6 depends on Task 4
