# Tasks

- [x] Task 1: 核查 `_sentence_end_boundaries` / `_extract_segment_text` 全部调用点
  - [x] SubTask 1.1: 确认两函数仅在 `pipeline.py` 内部使用，`__all__` 未导出，无跨模块引用
  - [x] SubTask 1.2: 确认 self_test 中 `_sentence_end_boundaries` 的 7 处直接断言位置

- [x] Task 2: 实现 Layer 1 文本切分
  - [x] SubTask 2.1: 新增 `_TextSpan` dataclass（字符区间 + 粗段下标）
  - [x] SubTask 2.2: 新增 `_resolve_coarse_char_spans()`（精确区间优先，find 兜底，与 coarse 等长对齐）
  - [x] SubTask 2.3: 新增 `_split_text_spans()`（标点连续串 + 空白之后切分；粗段字符区间边界强制切分）
  - [x] SubTask 2.4: 支持 `split_on_punctuation=False`（`punctuation_split=False` 路径）

- [x] Task 3: 实现 Layer 2 时间映射
  - [x] SubTask 3.1: 新增 `_map_items_to_chars()`——**局部回退**（失配记 None、游标不推进）
  - [x] SubTask 3.2: 新增 `_assign_item_buckets()`（bisect 归属，失配继承前一个 item）
  - [x] SubTask 3.3: 新增 `_absorb_orphan_buckets()`（孤儿并入相邻段，**不得跨越粗段**）
  - [x] SubTask 3.4: 新增 `_coarse_span_time()`（块内子区间按字符偏移线性分摊）

- [x] Task 4: 重构 `build_segment_response` 主流程
  - [x] SubTask 4.1: 接入 Layer 1/2，段文本改为直接切片
  - [x] SubTask 4.2: 移除 `puncts` 列表与末段尾部追加（tail）逻辑
  - [x] SubTask 4.3: 粗段改为按文本切分 + 线性分摊时间；未定位粗段退化为整块单段
  - [x] SubTask 4.4: 未定位粗段保留时间域强制切分（遗留 ❶ 兜底）
  - [x] SubTask 4.5: 新增 `_subgroup_char_bounds()` / `_group_text()` 处理 Layer 3 子组划界

- [x] Task 5: 移除退场函数
  - [x] SubTask 5.1: 删除 `_sentence_end_boundaries`（全量回退载体）
  - [x] SubTask 5.2: 删除 `_extract_segment_text`（截取载体）

- [x] Task 6: 更新 self_test
  - [x] SubTask 6.1: 第 12 组改为 Layer 1/2 helper 级断言
  - [x] SubTask 6.2: 第 4/13/14 组：更新因「空白归入前段」变化的期望值
  - [x] SubTask 6.3: 第 14 组：粗段内部按标点切分的新期望（原 3 段 → 4 段）
  - [x] SubTask 6.4: 第 16 组：`punctuation_split=False` 的末段延伸至文末
  - [x] SubTask 6.5: 第 19 组：注释改为「字符域切割」语义
  - [x] SubTask 6.6: **新增第 20 组脱敏回归 fixture**（重复念读 + 拉丁数字串混合，真实证件号不进仓库）

- [x] Task 7: 验证
  - [x] SubTask 7.1: `compileall` 编译通过
  - [x] SubTask 7.2: `self_test()` 通过
  - [x] SubTask 7.3: mock 端到端（真实 `_run_asr_align`）13/13 通过，原失败用例 A1/A7/A9/A10 拼接无损
  - [x] SubTask 7.4: 真实案例文本确认——段数 7 → 75，最长段 29.893s → 15.024s，拼接无损

- [x] Task 8: 文档
  - [x] SubTask 8.1: 更新 `pipeline.py` 模块 docstring 与 `build_segment_response` 参数说明
  - [x] SubTask 8.2: 更新 `middleware.py` 注释（`coarse_char_spans` 语义升级）
  - [x] SubTask 8.3: 新增本 spec 文档

# Task Dependencies

- Task 3 依赖 Task 2（Layer 2 的 bucket 归属需要 Layer 1 的文本段）
- Task 4 依赖 Task 2 + Task 3
- Task 5 依赖 Task 4（生产路径不再引用后才能删除）
- Task 6 依赖 Task 4 + Task 5
- Task 7 依赖 Task 6
- Task 8 依赖 Task 7
