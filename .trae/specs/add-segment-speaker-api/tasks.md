# Tasks

- [ ] Task 1: `pyproject.toml` 依赖组 + **依赖兼容性前置验证**（风险闸门）
  - [ ] `[project.optional-dependencies]` 新增 `diarization = ["pyannote.audio==4.0.7"]`（当前最新开源版）
  - [ ] 确认不破坏既有默认依赖与 `vllm` 依赖组（fastapi/uvicorn/python-multipart 由 vllm 带入）
  - [ ] **兼容性实测（闸门，失败则触发降级预案评审）**：在同一虚拟环境 `pip install -e ".[vllm,diarization]"` 后，`import vllm`、`import pyannote.audio`、`import transformers`、`import torchcodec` 全部成功且版本无 pip 冲突告警；系统 ffmpeg 可用（pyannote 4.x 音频解码依赖）；pyannote 最小 pipeline（community-1）加载 + 3 秒合成音频 diarization 冒烟通过；记录实测 torch/torchaudio/torchcodec 版本组合

- [ ] Task 2: 创建 `qwen_asr/inference/qwen3_speaker_diarizer.py`（SDK 层）
  - [ ] SubTask 2.1: `DiarizationSegment` frozen dataclass（`speaker: str`, `start_time: float`, `end_time: float`）
  - [ ] SubTask 2.2: `DiarizationResult` dataclass（`segments: List[DiarizationSegment]`），实现 `__iter__`/`__len__`/`__getitem__`；`speakers` 只读属性返回去重排序说话人列表
  - [ ] SubTask 2.3: `SpeakerDiarizer` 类：`from_pretrained(pretrained_model_name_or_path="pyannote/speaker-diarization-community-1", use_auth_token=None, device=None, **kwargs)`——懒导入 pyannote（未装抛 `ImportError` 提示 `pip install qwen-asr[diarization]`）；**内部以 `token=` 传凭证并同步设置 `HF_TOKEN` 环境变量**（新版 huggingface_hub 已移除 `use_auth_token` 透传）；按 device 调 `pipeline.to(...)`（接受任意 HF 管线 id，含 legacy 3.1 与本地路径）
  - [ ] SubTask 2.4: `diarize(audio, min_speakers=None, max_speakers=None) -> List[DiarizationResult]`：复用 `normalize_audios()` 归一化；构造 `{"waveform": tensor, "sample_rate": 16000}` 调 pipeline；**返回值防御性归一**（有 `speaker_diarization` 属性取之，否则视为 Annotation 直接使用）后统一 `itertracks(yield_label=True)` 收集片段
  - [ ] SubTask 2.5: `qwen_asr/__init__.py` 导出 `SpeakerDiarizer`；顺带修复现存缺陷：`__all__ = ["__version__"]` 引用未定义符号（`from qwen_asr import *` 触发 AttributeError）——`__version__` 改由 `importlib.metadata` 取包版本（回退 `"0.0.0"`），`__all__` 列出实际导出（`Qwen3ASRModel`/`Qwen3ForcedAligner`/`SpeakerDiarizer`/`parse_asr_output`/`__version__`）

- [ ] Task 2b: 对齐结果偏移/合并逻辑公开化（DRY 前置重构）
  - [ ] 将 `qwen_asr/inference/qwen3_asr.py` 私有方法 `_offset_align_result`/`_merge_align_results` 的实现提取到 `qwen_asr/inference/utils.py` 公开函数 `offset_align_result()`/`merge_align_results()`
  - [ ] `Qwen3ASRModel` 私有方法改为薄委托，对外行为不变（跑既有 demo/示例冒烟确认）

- [ ] Task 3: 创建 `qwen_asr/service/pipeline.py` 纯逻辑管道模块
  - [ ] SubTask 3.1: 语言双向映射：`language_to_code()`（内部名→BCP-47 码，覆盖 `SUPPORTED_LANGUAGES` 全部 **30** 项，含 Filipino→fil、Cantonese→yue、Macedonian→mk，未匹配回退小写）与 `resolve_language()`（请求入参 ISO 码/语言名→内部名）
  - [ ] SubTask 3.2: 句级切分 `split_segments(items, text, gap_threshold=0.8, max_segment_seconds=30.0)`：基于 `ForcedAlignItem` 序列的时间间隙与最大段长切分；段文本经游标匹配从原始 ASR 文本截取（保留标点），匹配失败回退 token 拼接；产出 `start/end/text`（3 位小数）
  - [ ] SubTask 3.3: 说话人归属 `attribute_speakers(segments, diarization, min_overlap=0.1)`：计算每段与各说话人重叠时长；dominant 为 `speaker`，重叠 ≥ 0.1s 者按重叠降序列入 `speakers`；无重叠时 `speaker=None, speakers=[]`
  - [ ] SubTask 3.4: 汇总 `build_speaker_summary(segments, diarization_speakers)`：**`speakers[]` 覆盖全部识别说话人（从未 dominant 者为零值项）**，`speakerCount == len(speakers)`；`totalDuration` = dominant 段时长和，`segmentCount` = dominant 段数；按 `totalDuration` 降序
  - [ ] SubTask 3.5: 编排函数 `build_segment_response(...)`：组合（ASR+对齐结果、diarization 结果、时长等）→ 最终响应 dict（language 码 / duration / text / segments / speakerSummary；processTime 由 middleware 填充）

- [ ] Task 4: 创建 `qwen_asr/service/scheduler.py` 多设备 GPU 显存感知调度器
  - [ ] SubTask 4.1: `GpuTaskScheduler(max_concurrent_tasks=2, gpu_reserve_mb=1024, aligner_device="cuda:0", diarizer_device="cuda:0")`：全局 FIFO 等待队列 + `asyncio.Condition`；队首任务一次性检查其涉及的全部设备（无跨设备死锁）
  - [ ] SubTask 4.2: `estimate_task_memory_mb(duration_sec, align_batch_size) -> {device: mb}`：**按设备拆分的瞬态公式**——aligner 设备 `512 + 256×align_batch_size`，diarizer 设备 `256 + 0.0625×duration`（同设备求和）——不含常驻权重（启动加载后已从对应设备空闲扣除，不得重复计算）
  - [ ] SubTask 4.3: `async acquire(needs: dict)`：对每个涉及设备 `torch.cuda.mem_get_info(device)` 实测空闲 ≥ 该设备需求 + gpu_reserve_mb，且运行数 < max_concurrent_tasks 才放行；任一不满足 FIFO 等待、唤醒后重检；CPU 设备退化为纯并发限制
  - [ ] SubTask 4.4: **异常安全协议**：对外暴露 `async with scheduler.slot(needs):` 上下文管理器（内部 try/finally 保证 release），任何异常/取消路径不泄漏许可
  - [ ] SubTask 4.5: `stats()` 返回 `{running, queued, devices: [{device, role, free_mb, total_mb}]}` 供 /health/detail 使用

- [ ] Task 5: 扩展 `qwen_asr/cli/serve.py` + 创建 `qwen_asr/service/middleware.py`（vLLM 进程内扩展，核心）
  - [ ] SubTask 5.1: `qwen_asr/service/__init__.py` 包文件
  - [ ] SubTask 5.2: serve.py 参数解析：新增扩展参数（`--forced-aligner` 默认 `Qwen/Qwen3-ForcedAligner-0.6B`、`--diarizer` 默认 `pyannote/speaker-diarization-community-1`、`--pyannote-token`/环境变量 `PYANNOTE_API_TOKEN`/`HF_TOKEN`、`--aligner-device` 默认跟随 vLLM 主设备、`--diarizer-device` 默认跟随 vLLM 主设备（**三者可指定不同设备自由组合**）、`--max-concurrent-tasks` 2、`--gpu-reserve-mb` 1024（每设备）、`--max-audio-seconds` 3600、`--max-audio-bytes` 500MB、`--segment-gap-threshold` 0.8、`--max-segment-seconds` 30、`--align-batch-size` 4），从 argv 剥离后剩余参数保留 vLLM CLI 语义
  - [ ] SubTask 5.3: **vLLM 0.14.0 app 构建调研（最先执行）**：确认 `vllm.entrypoints.openai` 的 app/engine client 获取方式（`app.state.engine_client` 等），输出简短调研结论；据此构建 app → 挂 ASGI middleware → 启动
  - [ ] SubTask 5.4: **显存预算（按设备）**：仅当 aligner/diarizer 设备与 vLLM 主设备相同且用户未显式指定 `gpu_memory_utilization` 时自动注入 0.70 并打日志（依据：A10 余 7.2GB − 常驻 1.9 − reserve 1 = 4.3GB ≥ 2×1h 任务瞬态 ~4GB，默认双并发自洽；扩展在独立设备时不注入，vLLM 用满 0.9）；扩展模型加载后**对每个扩展设备**做启动校验（`mem_get_info(device)` 空闲 < 该设备最小瞬态预估+reserve 即启动失败，输出该设备预算明细与建议值/独立设备参数示例）；`--diarizer ""`/`--forced-aligner ""` 时不加载扩展、不注入，行为与现状一致
  - [ ] SubTask 5.5: middleware 拦截 `POST /v1/audio/transcriptions`：multipart 解析（file/audio_url 二选一、model、language、prompt、response_format、timestamp_granularities[]、min/max_speakers 且 min>max → 400）；**按参数矩阵路由**（segment+缺省/json → 扩展响应；segment+text/verbose_json → 400；word → 400；非法值 → 400；无 segment → OpenAI 标准）；其余路径零干预
  - [ ] SubTask 5.6: **audio_url 安全**：仅 https、拒绝环回/私网（RFC1918/4193）/链路本地（含 169.254.169.254）/0.0.0.0、DNS 解析后全 IP 校验、下载大小上限 `--max-audio-bytes`、下载超时 60s；违规 400
  - [ ] SubTask 5.7: ASR 调用路径：middleware 自建 `Qwen3ASRProcessor`（CPU 常驻不占显存），chat template 对齐 `_build_text_prompt` 逻辑；经 engine client 生成；180s 分块复用 `utils.split_audio_into_chunks`，偏移/合并复用 Task 2b 公开函数；对齐器按 `--align-batch-size` 批量调用
  - [ ] SubTask 5.8: segment 模式管线：校验扩展已配置（否则 503）→ 解码与时长校验（超限 400）→ `async with scheduler.slot(need_mb)` 排队与许可（**try/finally 语义由上下文管理器保证**）→ **任务内阶段并行：diarization 线程与 ASR 分块循环同时启动**（diarization 不依赖转写结果；对齐依赖 ASR 输出保持串行；两分支各自经线程池执行，**aligner/diarizer 前向各由进程级 threading.Lock 串行化**）→ 两者完成后 pipeline 纯函数组装（时间重叠归并）→ 填充 processTime → 响应；**排队等待期间客户端断连取消任务并释放排队位**
  - [ ] SubTask 5.9: 非 segment 模式：OpenAI 标准响应（`json`→`{"text"}`、`text`→纯文本、`verbose_json`→`{text,duration,language}`），仅 engine ASR
  - [ ] SubTask 5.10: `GET /health/detail`：status/extensionModelsLoaded/runningTasks/queuedTasks/maxConcurrentTasks/gpuReserveMb + **devices[]（device/role/freeVramMb/totalVramMb，按设备）**
  - [ ] SubTask 5.11: 错误统一映射 OpenAI 风格 `{"error": {"message", "type", "code"}}`：400（参数/时长超限/语言不支持/model 不匹配/min>max/参数矩阵违规/URL 校验失败）、415（解码失败）、500（推理异常）、503（扩展未配置/依赖未安装）

- [ ] Task 6: 新增 `examples/example_segment_api.py` + 部署指南说明
  - [ ] 调用示例：multipart 请求（file 上传与 audio_url 两种音频源，`timestamp_granularities[]=segment`），打印美化 JSON；演示 `response_format=json` 标准 OpenAI 模式
  - [ ] `--self-test` 模式：对 Task 3 纯函数（30 项语言映射/切分/归属/汇总新口径）用构造数据断言验证，不依赖 GPU 与网络
  - [ ] 示例文件头注释包含部署要点：nginx `client_max_body_size 500m`/`proxy_read_timeout 900s`、HF_TOKEN 与 HF_HOME 缓存挂载、gpu_memory_utilization 参考配置（单卡 A10 0.70 双并发 / 0.75 单并发；T4 0.55 双并发 / 0.60 单并发；P4 不推荐长音频，仅短音频 0.35+单并发+reserve 512）、**多卡拓扑示例**（两卡 `--aligner-device cuda:1 --diarizer-device cuda:1` + vLLM 用满 0.9；三卡各自独占）、ffmpeg 依赖说明

- [ ] Task 7: 端到端验证
  - [ ] `pip install -e ".[vllm,diarization]"` 安装成功（无依赖冲突）；`python -c "from qwen_asr import SpeakerDiarizer"` 通过
  - [ ] `python examples/example_segment_api.py --self-test` 全部断言通过
  - [ ] **默认参数启动不 OOM**：`qwen-asr-serve`（默认扩展启用）在 A10 上启动成功，日志可见自动注入 gpu_memory_utilization=0.70 与预算明细；`GET /health`、`GET /health/detail` 正常（含 devices[] 按设备信息）；`from qwen_asr import *` 无 AttributeError（验证 __version__ 修复）
  - [ ] **多卡拓扑验证**（如有多卡环境）：`--aligner-device cuda:1 --diarizer-device cuda:1` 启动——不注入 gpu_memory_utilization（vLLM 用满）、扩展模型加载到 GPU1、/health/detail 显示双设备、segment 请求正常返回且 `nvidia-smi` 确认各模型在指定设备
  - [ ] **预算校验快速失败**：人为调高 gpu_memory_utilization 至剩余不足时，启动失败并输出可操作指引
  - [ ] curl 提交多人对话音频（`timestamp_granularities[]=segment`）：响应结构符合 spec（`speakerCount == len(speakers)`、segments 升序、口径自洽）
  - [ ] 参数矩阵全组合验证：无粒度×3 种 response_format、segment+text → 400、word → 400
  - [ ] 不带 timestamp_granularities 的请求返回 OpenAI 标准响应（`{"text": ...}`）
  - [ ] `--diarizer ""`/`--forced-aligner ""` 启动：segment 请求 503，标准转写正常（与现状一致）
  - [ ] 提交 > max-audio-seconds 音频返回 400；畸形音频返回 415；audio_url 传内网地址/非 https 返回 400
  - [ ] 1 小时音频（3600s）完整处理：全局时间戳正确、无 OOM
  - [ ] **diarization 性能压测与调优（40min 典型音频，A10）**：实测 diarization 单独耗时与端到端耗时，验证阶段并行生效（端到端 ≈ max(ASR+对齐, diarize) 而非两者之和）；调优项：分割/embedding 推理 batch_size（对比默认 vs 32/64）、`OMP_NUM_THREADS` 对 VBx 聚类耗时的影响；结果记入部署指南推荐配置
  - [ ] **并发正确性**：并发提交 N 个 segment 任务——排队数在 /health/detail 可见、全部 200、无 OOM；**并发与串行的 diarization 结果一致性抽查**（同音频串行跑 vs 并发跑，说话人片段一致）；构造任务执行中异常（如畸形音频过解码后）后队列仍可继续处理（验证许可不泄漏）
  - [ ] 既有功能回归：`from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner` 正常、`transcribe(return_time_stamps=True)` 冒烟（验证 Task 2b 重构无回归）、`qwen-asr-demo` 不受影响

# Task Dependencies
- [Task 1] 为全局闸门：兼容性实测失败则评审降级预案（独立进程 diarization worker 或回退 pyannote 3.1.1）后才继续
- [Task 2] 依赖 [Task 1]；[Task 2b] 无依赖可并行
- [Task 3] 依赖 [Task 2]（归属逻辑消费 `DiarizationSegment`/`DiarizationResult` 类型）
- [Task 5] 依赖 [Task 2]、[Task 2b]、[Task 3]、[Task 4]；内部 SubTask 5.3（vLLM API 调研）可提前进行
- [Task 6] 依赖 [Task 3]、[Task 5]
- [Task 7] 依赖 [Task 1]-[Task 6]
- [Task 1]、[Task 2b]、[Task 4] 可独立先行；[Task 3] 与 [Task 4] 可并行
