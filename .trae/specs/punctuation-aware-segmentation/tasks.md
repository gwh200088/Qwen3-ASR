# Tasks

- [x] Task 1: `pipeline.py` 句末标点硬边界计算（纯函数，仅标准库）
  - [x] SubTask 1.1: 新增常量 `_SENTENCE_END_CHARS = set("。！？；.!?;\n")`——**含 ASCII 句点 `.`**（英文句末标点主力；英文缩写 `Mr.` 误切为已接受 trade-off，见 spec）；逗号/顿号/冒号不在内
  - [x] SubTask 1.2: 新增 `_sentence_end_boundaries(items, full_text, coarse_spans=None) -> Tuple[List[bool], List[str]]`——复用 `_extract_segment_text` 的 greedy `find` 游标匹配语义全局匹配每个 item 到 `full_text`；相邻 item 匹配区间之间的 between-span（`full_text[前item匹配终点:后item匹配起点]`）：
    - 含任一句末标点 → `boundaries[i] = True`（硬边界），`puncts[i]` = between-span 中全部句末标点字符按出现顺序拼接（如 `"。"`、`"？！"`；空格/引号等非句末标点字符不收集）
    - 不含 → `boundaries[i] = False`，`puncts[i] = ""`
    - **跨失败块边界 puncts 置空（v3）**：边界时间间隙区间 `[items[i].end_time, items[i+1].start_time]` 与任一 coarse 块时间区间（`coarse_spans`，(start, end) 列表）相交 → `puncts[i] = ""`（boundaries 保持原判定，切分照常——间隙 ≥ 失败块时长 ≥ 2.0s 必然切分）；判断方式与 `_gap_blocked` 同一逻辑（时间域）
    - 两个列表长度均为 `len(items) - 1`；`len(items) <= 1` 时返回 `([], [])`
    - **任一 item 匹配失败 → 全量回退**：返回全 False + 全空串（理由见 spec：失败 item 后游标位置不确定，部分保留边界可能导致后续 between-span 错位），不抛异常
  - [x] SubTask 1.3: helper 级自测断言：句号/问号/叹号/分号/**ASCII 句点**/换行触发、逗号顿号不触发、连续标点 "？！" 完整收集、空格不收集、**跨失败块边界 puncts 置空**（boundaries 不变）、匹配失败回退全 False+空串、单 item 序列返回空

- [x] Task 2: 切分逻辑接入硬边界
  - [x] SubTask 2.1: `_split_groups` 新增可选参数 `hard_boundaries`：切分条件改为 `hard_boundaries[i-1] or gap >= segment_gap_threshold or span > max_segment_seconds`（硬边界处无视间隙恒切分；`hard_boundaries=None` 时全按非硬边界，兼容既有调用）
  - [x] SubTask 2.2: `_split_by_speaker` 计算并透传 `hard_boundaries`（word 模式 4 元组与 3 元组等长同序，索引直接对齐）；说话人变化切分逻辑零改动（无标点处照切）
  - [x] SubTask 2.3: `build_segment_response` 接线：`punctuation_split=True`（默认）时先算 `_sentence_end_boundaries`（传入 coarse 块时间区间列表）再传入切分；`punctuation_split=False` 时跳过计算（boundaries=None，纯间隙行为）；两种归属模式（word/segment）均生效

- [x] Task 3: 聚合硬边界阻断、段文本标点追加与默认值
  - [x] SubTask 3.1: `_merge_same_speaker` 新增硬边界阻断——同人相邻段之间的边界为句末标点硬边界时不合并（分组连续覆盖全部词，按累计词数反查全局边界索引；`hard_boundaries=None` 时无阻断，兼容 off 路径）
  - [x] SubTask 3.2: **段文本句末标点附前段**：段构建循环中（两种归属模式），非末段的段末边界为硬边界时，`text += puncts[边界索引]`（全局边界索引同按累计词数反查；跨失败块边界 puncts 已置空故自然不追加）；空格/引号不追加；`punctuation_split=False` 或匹配失败回退时无追加（回退路径 `_extract_segment_text` 语义不变）
  - [x] SubTask 3.3: **末段尾部句末标点追加（v3）**：全局匹配成功时，full_text 末词匹配终点之后的句末标点字符（按序拼接）追加到末段 `text` 末尾；匹配失败回退 / `punctuation_split=False` 时不追加
  - [x] SubTask 3.4: `build_segment_response` 签名默认 `segment_gap_threshold` 0.8 → 2.0；新增 `punctuation_split: bool = True`

- [x] Task 4: `serve.py` 参数与 `extensions.py`/`middleware.py` 透传
  - [x] SubTask 4.1: `--segment-gap-threshold` 默认 0.8 → 2.0，help 更新为"相邻词之间无句末标点时触发切分的静音间隙阈值（秒）；句末标点处恒切分"
  - [x] SubTask 4.2: 新增 `--punctuation-split {on,off}`（choices 校验，默认 `on`），help 含回退组合说明（完整旧行为 = `off` + `--segment-gap-threshold 0.8`）；纳入扩展参数剥离列表
  - [x] SubTask 4.3: `ExtensionState` 新增 `punctuation_split` 字段，middleware 在 `build_segment_response` 调用点透传（与 `segment_gap_threshold` 同一链路）

- [x] Task 5: self_test 全量更新（无 GPU 依赖）
  - [x] SubTask 5.1: 新增断言组：
    - 句末标点切分：小间隙（0.3s）+ "。" → 切分（"说号就行。啊？"快问快答场景）
    - **英文句点切分**："Nice to meet you. See you." 小间隙 + `.` → 切分两段
    - 逗号不切：小间隙 + "，" → 不切
    - **段文本标点附前段**："说号就行。啊？说吧" → 段 text 为"说号就行。"/"啊？"/"说吧"；`"".join(segments[].text)` 标点无损；连续标点 "？！ " → 前段追加"？！"（空格不追加）
    - **末段尾部标点追加**："说号就行。啊？"（全文以"？"结尾）→ 末段 text 为"啊？"（尾部标点不丢失）
    - **跨失败块边界不追加**：A/B 词之间隔失败块（块文本含多个句末标点）→ 前段无垃圾标点后缀（无"。。？。"）；粗段 text 含自身原文标点、无重复
    - **混合场景拼接无损**：正常段 + 粗段混合产出（含跨失败块边界与全文尾部标点）→ 按 start 排序后 `"".join(segments[].text) == text`
    - 用户场景回归 1：无标点 0.88s 间隙（"想|负责"）→ 不切分
    - 用户场景回归 2：无标点 1.943s 等距单字序列 → 聚合为一段（span > 30s 时强切）
    - 无标点间隙 ≥ 2.0s（含边界值）→ 切分
    - 同人聚合不跨越硬边界："今天不错。明天更好。"同人快速连接（merge_gap 显式调大如 5.0 + gap 阈值调小如 0.5，构造"若无阻断则会合并"的场景）→ 仍两段
    - **默认参数下聚合零触发**：默认 gap 2.0 / merge_gap 2.0 构造同人间隙 2.5s 切分的两段 → 不合并（gap ≥ 2.0 不满足 gap < 2.0）
    - `punctuation_split=False`：有标点小间隙 → 不切分（开关关闭生效）；off + 显式 gap 0.8 + 同人间隙 1.0 → 切分后聚合回一段（完整旧行为回归）
    - 说话人变化无标点处照切（word 模式原断言保持通过）
    - match 失败回退：item 文本不在 full_text → 纯间隙行为（无标点切分、无标点追加）
    - segment 模式标点切分与标点追加同样生效
  - [x] SubTask 5.2: 既有断言适配默认阈值 2.0：
    - 测试 5（"测试/静音" gap 1.0 无标点，原断言两段）：显式传 `segment_gap_threshold=0.8` 保持原断言语义
    - `merge_gap=0` 用例（gap 1.2 原断言两段）：显式传 `segment_gap_threshold=0.8` 或调大间隙至 ≥ 2.0，保持"切分后不聚合"断言意图
    - 短插话保护用例（gap 1.2 原断言两段）：改为 gap ≥ 2.0（切分发生）且 `speaker_merge_gap` 足以合并的场景，验证保护仍阻断合并
    - coarse 混合用例（乙|丙 gap 1.0 原靠间隙隔离两段）：调大间隙至 ≥ 2.0，保持"粗段阻断同人聚合 + start 升序"断言意图
    - 检查其余用例（Hello world/30s 强切/边界阈值显式传参组）在新默认下结果不变；既有用例若涉及段文本断言（如"你好，世界。欢迎光临"），核对硬边界切分后 text 变化（"你好，世界。"附句号）并更新期望值
  - [x] SubTask 5.3: 本地运行 `python examples/example_segment_api.py --self-test` 全量通过

- [x] Task 6: 文档更新
  - [x] SubTask 6.1: `docs/deployment-guide.md`：`--segment-gap-threshold` 新语义与默认值 2.0；`--punctuation-split` 参数与回退组合（仅关标点切分 / 完整旧行为）；句末标点切分规则（含句点 `.`、逗号不切、聚合不跨硬边界、段文本标点附前段**含末段尾部追加与跨失败块不追加、拼接无损口径**）；**默认参数下同人聚合不再触发**及 `--speaker-merge-gap` 生效条件（须大于 gap 阈值）；已知 trade-off（Mr. 缩写误切及 U.S.A./3.14 回退不受影响说明、短插话保护区间收窄、无标点文本段变长）

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3（默认值与 pipeline 签名一致）
- Task 5 depends on Task 1, 2, 3, 4
- Task 6 depends on Task 4
