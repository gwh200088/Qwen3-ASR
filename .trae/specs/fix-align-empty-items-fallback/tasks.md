# Tasks

- [x] Task 1: 修改 `_flush_align_batch()` 增加逐块空 items 检测（Change 1）
  - [x] SubTask 1.1: 调整 `_flush_align_batch()` 结构——先 `batch_results = _align_batch(ext, payload)`，再 `aligned.extend(batch_results)`，拿到逐块结果做空 items 检测（重构目的是拿到逐块结果，不是修重复调用——原代码本来就只调用一次）
  - [x] SubTask 1.2: 对 `batch` 与 `batch_results` zip 遍历，检测 `result is None` 或 `not list(result.items)` 的块，调用 `_coarse_interval()` 加入 `coarse_chunks`，块索引 `idx` 加入 `coarse_idx_set`（需在 `_run_asr_align` 作用域声明 `coarse_idx_set = set()` 并在 `_flush_align_batch` 内 nonlocal 引用）。**P0 修复**：`except` 分支的 for 循环内也需 `coarse_idx_set.add(idx)`——否则部分成功 + 整批异常场景下 Change 2 会重复兜底这些块
  - [x] SubTask 1.3: 触发空 items 兜底时输出 WARNING 日志（块序号 + 文本长度 + `[start, end]` 时间区间）

- [x] Task 2: 在 `_run_asr_align()` 收尾处增加时间覆盖率双保险（Change 2）
  - [x] SubTask 2.1: 在 `merged = merge_align_results(...)` 之后、`if merged is None:` 之前，增加覆盖率校验分支（仅当 `merged is not None` 时执行）
  - [x] SubTask 2.2: 遍历 `per_chunk` 中的非空文本块（用 `enumerate` 拿到块索引 `idx`），检查 `merged.items` 中是否有任一 item 的 `start_time` 落在 `[offset, offset+块长)` 区间内（左闭右开：`item.start_time == offset+块长` 算下一块覆盖，不算本块）
  - [x] SubTask 2.3: 对未覆盖的块，检查 `idx in coarse_idx_set`（块索引集合去重，非浮点时间近似相等），已在则跳过；未在则补进 `coarse_chunks` 并输出 WARNING 日志（块序号 + 块区间 + "未被 item 覆盖"）

- [x] Task 3: middleware 层计算并返回精确 coarse 字符区间（Change 4）
  - [x] SubTask 3.1: 在 `_run_asr_align()` 返回前，基于 `per_chunk` 的文本长度精确计算每个 coarse 块的字符区间——块 i 的 `char_start = sum(len(per_chunk[j][1]) for j < i)`，`char_end = char_start + len(per_chunk[i][1])`
  - [x] SubTask 3.2: **coarse 条目 → 块索引映射**（实现关键）——用 `start == per_chunk[i][3]`（offset）反查，不用 `coarse_idx_set`（后者只含 Change 1 的块，会漏整批异常和 `merged is None` 重建的块）
  - [x] SubTask 3.3: `_run_asr_align()` 返回值 4 元组改 5 元组，新增 `coarse_char_spans: List[Tuple[int, int]]`
  - [x] SubTask 3.4: [middleware.py L912 解包点](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L912) 同步改 5 元组解包
  - [x] SubTask 3.5: [middleware.py L930 生产调用点](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L930) 透传 `coarse_char_spans` 到 `build_segment_response`

- [x] Task 4: pipeline tail 逻辑排除 coarse 字符区间（Change 3）
  - [x] SubTask 4.1: `build_segment_response()` 新增可选参数 `coarse_char_spans: Optional[List[Tuple[int, int]]] = None`
  - [x] SubTask 4.2: 在 tail 逻辑前确定 coarse 字符区间——优先用 `coarse_char_spans`；为 `None` 时用 `coarse_text` 在 `full_text` 中游标 `find` 计算（按 `start` 排序避免乱序错位）
  - [x] SubTask 4.3: [tail 逻辑](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py#L766-L778) 改为：取 `full_text[last_end:]` 中的句末标点时，跳过落在任一 coarse 块字符区间 `[char_start, char_end)` 内的字符
  - [x] SubTask 4.4: docstring 补充 `coarse_char_spans` 参数说明

- [x] Task 5: 验证（pipeline self_test 第 19 组新增 P0 场景）
  - [x] SubTask 5.6: 端到端验证 P0 场景拼接无损——第 19 组测试：尾部空 items 块 + coarse_char_spans 精确区间，断言 `"".join(seg["text"]) == full_text`（标点不重复）
  - [x] SubTask 5.7: 回归验证无 coarse 块时 tail 行为不变——第 19 组末尾：无 coarse 块时 tail 取末尾句号（既有行为）
  - [x] SubTask 5.8: 回归验证 `coarse_char_spans=None` 双保险兜底——第 19 组：不传 `coarse_char_spans`，验证 pipeline 自己 `find` 计算字符区间且 tail 逻辑正确排除
  - [ ] SubTask 5.1: 语法编译通过（本地无 Python/Docker，待容器内验证）
  - [ ] SubTask 5.2-5.5: middleware 级单元验证（`_flush_align_batch` 是闭包，需走 `_run_asr_align` 整体 + mock，待容器内验证）

# Task Dependencies

- Task 2 依赖 Task 1（覆盖率校验基于 `coarse_idx_set` 已含逐块兜底结果，防重复需要先有逐块兜底的产出）
- Task 3 依赖 Task 1 + Task 2（`coarse_chunks` 最终内容确定后才能计算对应的 `coarse_char_spans`）
- Task 4 依赖 Task 3（pipeline 侧 `coarse_char_spans` 参数就绪后才能做 tail 逻辑改）
- Task 5 依赖 Task 1 + Task 2 + Task 3 + Task 4
