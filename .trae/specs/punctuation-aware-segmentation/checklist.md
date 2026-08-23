# Checklist

## 句末标点硬边界（pipeline.py）
- [x] `_SENTENCE_END_CHARS` 含 `。！？；.!?;` 与换行——**含 ASCII 句点 `.`**（英文句末标点主力）；逗号/顿号/冒号不在内
- [x] `_sentence_end_boundaries` 游标匹配正确：相邻 item 间 between-span 含句末标点 → True；逗号/顿号 → False
- [x] 返回 `(boundaries, puncts)`：puncts 为 between-span 中句末标点字符按序拼接（连续标点 "？！" 完整收集，空格/引号不收集）
- [x] **跨失败块边界 puncts 置空**（边界时间间隙区间与任一 coarse 块时间区间相交，时间域判断同 `_gap_blocked`）：boundaries 保持原判定、切分照常；前段无垃圾标点后缀、与粗段原文标点无重复
- [x] 任一 item 匹配失败 → 全量回退（全 False + 全空串），不抛异常；spec 已写明全量回退理由（游标错位风险）
- [x] 硬边界处恒切分（无视间隙大小，含间隙 0）；`hard_boundaries=None` 兼容既有调用与 `punctuation_split=False` 路径
- [x] 逗号/顿号不因标点切分（仅按间隙/段长规则）

## 段文本句末标点附前段
- [x] 硬边界切分处：前段 `text` 末尾追加 `puncts[边界]`（"说号就行。"含句号、"啊？"含问号）
- [x] 连续句末标点全量追加（"？！ " → "？！"，空格不追加）
- [x] **末段尾部句末标点追加**：full_text 以句末标点结尾（"说号就行。啊？"）→ 末段 text 为"啊？"，尾部标点不丢失；off/匹配失败回退时不追加
- [x] **跨失败块边界不追加**：A/B 词间隔失败块 → 前段无"。。？。"类垃圾后缀；失败块标点仅出现在粗段 text（块原文）
- [x] `"".join(segments[].text)` 与 `text` 字段一致（拼接无损，**含混合粗段场景**——按 start 排序后正常段标点 + 粗段原文标点恰好还原全文）
- [x] 块内逗号/空格仍由游标截取自然保留；off/匹配失败回退路径无追加、`_extract_segment_text` 语义不变
- [x] segment 模式标点切分与标点追加（含末段尾部）同样生效

## 无句末标点处的间隙阈值
- [x] 无句末标点 + 间隙 < 2.0s 不切分（用户示例回归：0.88s "想|负责" 不切、1.943s 单字序列聚合不逐字成段）
- [x] 无句末标点 + 间隙 ≥ 2.0s 切分（含边界值 2.0）
- [x] 段长 > max_segment_seconds 强切不受影响（含无标点长块，span 33s → 两段）
- [x] `build_segment_response` 签名默认 `segment_gap_threshold` 0.8 → 2.0

## word 模式与同人聚合
- [x] 说话人变化切分保留（无标点处照切，快速交锋原断言保持通过）
- [x] 同人二次聚合不跨越句末标点硬边界（"今天不错。明天更好。"同人 + 构造可合并参数 → 仍两段）
- [x] **默认参数（gap 2.0 / merge_gap 2.0）下聚合零触发**（间隙切分与合并条件互斥）；off + gap 0.8 旧组合下聚合恢复活性（完整旧行为回归断言）
- [x] 短插话保护在仍发生的切分合并处生效（gap ≥ 2.0 切分 + 间隙内他人 turn ≥ 0.3s → 不合并）
- [x] segment 模式归属投票/speakers 口径/speakerSummary 口径零变化
- [x] 粗段（coarse_chunks）不受标点逻辑影响；混合产出仍按 start 升序；粗段不参与聚合

## 标点切分开关
- [x] `--punctuation-split {on,off}` 默认 on；非法值启动报错；`ExtensionState` 存储并透传至 `build_segment_response`
- [x] off：跳过硬边界计算，有标点小间隙不切分（开关生效断言）
- [x] off + `--segment-gap-threshold 0.8` = 完整旧行为（切分 + 聚合活性回归断言）；off 时段文本无标点追加

## 参数与透传
- [x] serve.py `--segment-gap-threshold` 默认 2.0 + help 更新（无句末标点处的间隙阈值；句末标点处恒切分）
- [x] serve.py `--punctuation-split` help 含回退组合说明；纳入扩展参数剥离列表
- [x] middleware 透传链与 `segment_gap_threshold` 同一链路

## 验证
- [x] self_test 新增断言全部通过（快问快答切分/英文句点切分/逗号不切/段文本标点附前段与连续标点/末段尾部追加/跨失败块不追加/混合场景拼接无损/用户两示例场景回归/无标点 ≥2.0 切/聚合不跨硬边界/默认聚合零触发/开关 off 生效与旧行为回归/match 失败回退/segment 模式生效）
- [x] 既有断言适配后全部通过（含段文本期望值更新：硬边界切分后 text 附句号）；`python examples/example_segment_api.py --self-test` 本地全量通过（无 GPU 依赖）
- [x] 响应 JSON 结构/字段名/类型零变化（无新增/删除/改名）

## 文档
- [x] deployment-guide.md 含：新参数语义与默认值 2.0、`--punctuation-split` 与回退组合、句末标点切分规则（含句点、逗号不切、聚合不跨硬边界、段文本标点附前段**含末段尾部追加与跨失败块不追加**、拼接无损口径）
- [x] deployment-guide.md 写明**默认参数下同人聚合不再触发**与 `--speaker-merge-gap` 生效条件（须大于 gap 阈值）
- [x] deployment-guide.md 写明已知 trade-off（Mr. 缩写误切及 U.S.A./3.14 因 token 匹配失败回退不受影响、短插话保护区间收窄、无标点文本段变长）
