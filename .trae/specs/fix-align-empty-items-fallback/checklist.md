# Checklist

## Change 1：`_flush_align_batch()` 逐块空 items 检测
- [ ] `_flush_align_batch()` 内对齐返回后逐块检测空 items（`result is None` 或 `not list(result.items)`），空块进 `coarse_chunks`
- [ ] `_flush_align_batch()` 调整结构：先 `batch_results = _align_batch(...)`，再 `aligned.extend(batch_results)`，拿到逐块结果做检测（非"避免重复调用"——原代码本来就只调用一次）
- [ ] 空块兜底触发时块索引 `idx` 加入 `coarse_idx_set`（供 Change 2 去重）
- [ ] 空块兜底触发时输出 WARNING 日志（块序号 + 文本长度 + 时间区间）
- [ ] `_run_asr_align` 作用域声明 `coarse_idx_set = set()`，`_flush_align_batch` 内 nonlocal 引用

## Change 2：`_run_asr_align()` 收尾覆盖率双保险
- [ ] 覆盖率双保险在 `merged is not None` 时执行，`merged is None` 时跳过
- [ ] 覆盖率校验遍历 `per_chunk` 非空文本块（用 `enumerate` 拿块索引 `idx`），检查 `merged.items` 中 `start_time` 是否落入 `[offset, offset+块长)` 区间
- [ ] 区间端点左闭右开：`item.start_time == offset+块长` 算下一块覆盖，不算本块
- [ ] 未覆盖块补进 `coarse_chunks` 前检查 `idx in coarse_idx_set`（块索引集合去重，非浮点时间近似相等）
- [ ] 未覆盖块补进时输出 WARNING 日志（块序号 + 块区间 + "未被 item 覆盖"）
- [ ] spec 记录的假阴性局限（幻觉时间戳跨块）在代码注释中标注

## Change 3：pipeline tail 逻辑排除 coarse 字符区间（P0 修复）
- [ ] [tail 逻辑 L746-749](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/pipeline.py#L746-L749) 改为：取 `full_text[last_end:]` 句末标点时跳过 coarse 块字符区间 `[char_start, char_end)`
- [ ] 优先用 `coarse_char_spans` 参数确定字符区间；为 `None` 时用 `coarse_text` 游标 `find` 计算（双保险兜底）
- [ ] 无 coarse 块时 tail 行为不变（无字符区间需排除）

## Change 4：middleware 层提供精确 coarse 字符区间
- [ ] `_run_asr_align()` 返回值 4 元组改 5 元组，新增 `coarse_char_spans: List[Tuple[int, int]]`（与 `coarse_chunks` 一一对应）
- [ ] 基于 `per_chunk` 的文本长度精确计算字符区间（`char_start = sum(len(per_chunk[j][1]) for j < i)`）
- [ ] **coarse 条目 → 块索引映射**用 `start == per_chunk[i][3]`（offset）反查，**不用 `coarse_idx_set`**（后者只含 Change 1 的块，会漏整批异常和 `merged is None` 重建的块）
- [ ] [_coarse_interval()](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L441-L443) 返回的 `start` 是 `offset` 原值（`min()` 截断只作用于 `end`），三条路径通用
- [ ] `build_segment_response()` 新增可选参数 `coarse_char_spans: Optional[List[Tuple[int, int]]] = None`
- [ ] [middleware.py:857 解包点](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L857) 同步改 5 元组解包——`(full_text, language, merged_align, coarse_chunks, coarse_char_spans), diar_results = results`
- [ ] [middleware.py:861 生产调用点](file:///d:/workplace/TMRI/AI/Qwen-Asr/Qwen3-ASR/qwen_asr/service/middleware.py#L861) 透传 `coarse_char_spans`
- [ ] pipeline.py 的 51 处 self_test 调用点不传 `coarse_char_spans`（默认 None，pipeline 自己 find 兜底；grep 52 处含 1 处 def 定义行）

## 既有行为回归检查
- [ ] 整批异常兜底（`except` 分支）行为不变——批内全部块进 `coarse_chunks`，不重复触发逐块检测
- [ ] 整批异常的块索引加入 `coarse_idx_set`（供 Change 2 覆盖率双保险去重）
- [ ] **部分成功 + 整批异常场景不重复兜底**——batch1 整批异常 + batch2 正常 → merged 非 None → 覆盖率双保险运行时，batch1 的块 `idx in coarse_idx_set` 跳过，不重复补进 `coarse_chunks`
- [ ] `merged is None` 全空分支行为不变——全部非空文本块整体兜底
- [ ] `merge_align_results` 行为不变——仍只跳过 `None`，空 items 由 middleware 层兜底
- [ ] `build_segment_response` 既有参数签名不变（`coarse_char_spans` 是新增可选参数，默认 None）
- [ ] pipeline.py 的 51 处 self_test 调用点无需修改（不传新参数；grep 52 处含 1 处 def 定义行）
- [ ] 取消事件（`cancel_event`）在逐块检测和覆盖率校验路径仍优先抛 RuntimeError，不被容错吞掉

## 验证检查（需走 `_run_asr_align` 整体，非直接调闭包）
- [ ] 语法编译通过 `python3 -m compileall -q qwen_asr`
- [ ] **注意 `_flush_align_batch` 是闭包，无法直接导入调用**——测试需走 `_run_asr_align` 整体，mock `engine_generate` + 真实 event loop + mock `_align_batch` + 构造 `ext`（processor/lock/align_batch_size）
- [ ] 模拟空 items 用例：该块进入 `coarse_chunks`，`coarse_idx_set` 记录其索引
- [ ] 模拟未覆盖用例：覆盖率双保险补进 `coarse_chunks`，不与空 items 兜底重复（`coarse_idx_set` 去重生效）
- [ ] 整批异常用例：既有兜底不受影响，不重复触发逐块检测
- [ ] **部分成功 + 整批异常用例（P0 回归）**：batch1 整批异常 + batch2 正常 → merged 非 None → 覆盖率双保险运行时，batch1 的块 `idx in coarse_idx_set` 跳过，不重复补进 `coarse_chunks`，`coarse_chunks` 数量正确（不翻倍）
- [ ] 全空用例：`merged is None` 分支不受影响，覆盖率双保险跳过
- [ ] **P0 场景拼接无损**：尾部空 items 块时 `"".join(seg["text"] for seg in segments) == full_text`（标点不重复）
- [ ] 无 coarse 块时 tail 行为不变：取 `full_text[last_end:]` 全部句末标点
- [ ] `coarse_char_spans=None` 双保险兜底：pipeline 自己 `find` 计算字符区间且 tail 逻辑正确排除
