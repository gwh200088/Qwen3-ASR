# Checklist

## 词级归属核心（pipeline.py）
- [ ] 词序列直接使用 align_items（无 ASR 词序列映射层，未引入 nagisa/soynlp 等第三方依赖，pipeline.py 仍仅标准库）
- [ ] `_attribute_words` 词中点投票：单一 turn 直接归属；多 turn 重叠取区间重叠最大者（并列 id 字典序最小）；无 turn 覆盖标记为洞
- [ ] `_fill_gaps` 四规则（洞=无 turn 覆盖词）：前后同人继承 / 异人中点插值（仍无覆盖归前词）/ 开头洞后向继承 / 结尾洞前向继承
- [ ] 全序列无已归属词时全部 speaker=None，不抛异常

## word 模式段构建
- [ ] 切分顺序正确：间隙/段长切分 → speaker 变化追加切分 → 同人 merge_gap 二次聚合
- [ ] 段 start/end = 首末词时间戳（spec 定义），3 位小数
- [ ] 快速交锋场景：换人间隙 0.2s < 0.8s 时按说话人切成两段（A 词归 A、B 词归 B）
- [ ] 同人停顿 1.2s 且 merge_gap=2.0 且间隙无他人 turn → 聚合为一段；merge_gap=0 时不聚合
- [ ] 短插话保护：合并间隙区间内存在其他 speaker turn 覆盖 ≥ 0.3s → 不合并
- [ ] 聚合后段长超 max_segment_seconds 时放弃合并
- [ ] `segments[].speaker` / `speakers` 符合 word 模式口径（词归属多数者 / 词归属去重集合按词数降序）
- [ ] `text` 全文与段文本游标截取不受模式影响（保留标点/空格）

## 逐块对齐容错（middleware.py）
- [ ] 单个对齐 batch 异常被捕获：请求 200，失败块走粗段，其余块正常；日志含块序号与异常摘要
- [ ] cancel_event 置位时取消异常照常传播（不被容错吞掉）
- [ ] align 全空时：非空文本块全部产出粗段（块区间投票），不再返回空 segments
- [ ] 无任何有效 ASR 文本块时 segments=[]（与现状一致）
- [ ] 粗段与正常段同构（同字段），粗段粒度 180s 块级，文档已写明精度限制
- [ ] 正常段与粗段混合产出时 segments[] 按 start 升序全局排列（粗段插入正确时间位置，无乱序）
- [ ] 粗段不参与同人二次聚合（不与任何正常段合并）
- [ ] **segment 模式下**正常路径（无失败块）coarse_chunks=None，响应与现状逐字节等价（word 模式正常路径的归属/切分变化即方案目的，不适用逐字节等价）
- [ ] 容错在 word 与 segment 两种模式下均生效
- [ ] ASR 分块生成异常不在容错范围（spec 已声明，整请求失败为现状行为）

## 模式切换与降级
- [ ] `speaker_attribution="segment"` 归属路径代码零改动，既有全部行为等价（除容错/兜底增强）
- [ ] diarization 为空时所有词 speaker=None，间隙切分照常，`speaker=null`/`speakers=[]`
- [ ] 响应 JSON 结构/字段名/类型与升级前完全一致（无新增/删除/改名）

## 参数与透传
- [ ] `--speaker-attribution {word,segment}` 默认 word；非法值启动报错
- [ ] `--speaker-merge-gap` 默认 2.0，float，负值启动报错，0 表示不合并，仅 word 模式生效
- [ ] 两参数经 ExtensionState 透传到 build_segment_response 调用点

## 验证
- [ ] self_test word 模式断言全部通过（投票/洞填充/切分/聚合/短插话保护/段边界定义全覆盖）
- [ ] self_test 容错断言全部通过（单块失败混合/全空粗段/segment 模式下 None 等价/混合 start 升序/粗段不聚合/取消不吞）
- [ ] 既有 self_test 断言（segment 模式回归）全部原样通过
- [ ] `python examples/example_segment_api.py --self-test` 本地全量通过（无 GPU 依赖）

## 文档
- [ ] deployment-guide.md 参数表含两个新参数及默认值
- [ ] deployment-guide.md 写明 `speakers` 字段两模式口径差异、word 模式精度预期（diarization 边界 ±0.5s 限制）
- [ ] deployment-guide.md 写明逐块对齐容错与粗段兜底行为（含 180s 粗段精度限制、粗段对 totalDuration 的放大失真、ASR 生成异常不在容错范围）
