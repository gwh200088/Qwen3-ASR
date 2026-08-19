# Checklist

## 依赖与 SDK 层
- [ ] `pyproject.toml` 新增 `diarization = ["pyannote.audio==4.0.7"]` 可选依赖组，既有依赖与脚本入口不变
- [ ] **依赖兼容性闸门**：同环境 `import vllm + pyannote.audio + transformers + torchcodec`、ffmpeg 可用、community-1 最小推理冒烟通过，torch/torchaudio/torchcodec 版本组合已记录
- [ ] `qwen_asr/inference/qwen3_speaker_diarizer.py` 定义 `DiarizationSegment`（frozen：speaker/start_time/end_time）、`DiarizationResult`（`__iter__`/`__len__`/`__getitem__` + `speakers` 属性）、`SpeakerDiarizer`
- [ ] `SpeakerDiarizer.from_pretrained()` 对外签名含 `use_auth_token`，**内部以 `token=` 传递并设置 `HF_TOKEN` 环境变量**（兼容新版 huggingface_hub）；支持 device；未安装 pyannote 抛含安装提示的 `ImportError`
- [ ] `diarize()` 对 pipeline 返回值做**防御性归一**（`DiarizeOutput.speaker_diarization` 或直接 Annotation）后统一 itertracks；复用 `normalize_audios()`；`min_speakers`/`max_speakers` 透传（4.x 对不支持参数降级为警告，best-effort）；`from_pretrained` 默认 community-1，兼容 legacy 3.1 与本地路径
- [ ] `qwen_asr/__init__.py` 导出 `SpeakerDiarizer`；修复现存 `__all__`/`__version__` 缺陷（`__version__` 由 `importlib.metadata` 取、`__all__` 列实际导出），`from qwen_asr import *` 无 AttributeError
- [ ] Task 2b 重构：`offset_align_result()`/`merge_align_results()` 成为 `utils.py` 公开函数，`Qwen3ASRModel` 私有方法薄委托，`transcribe(return_time_stamps=True)` 冒烟无回归

## 管道纯逻辑
- [ ] `language_to_code()` 覆盖全部 **30** 个 `SUPPORTED_LANGUAGES`（含 Filipino→fil、Cantonese→yue、Macedonian→mk），未匹配回退小写；`resolve_language()` 接受 ISO 码与语言名双形式
- [ ] `split_segments()` 按 token 间隙 ≥ 阈值或段长 ≥ max_segment_seconds 切分；段文本游标匹配自原始 ASR 文本（保留标点），失败回退 token 拼接；start/end 为 3 位小数
- [ ] `attribute_speakers()` 中 dominant 说话人 = 重叠时长最大者；`speakers` 含重叠 ≥ 0.1s 的说话人（按重叠降序）；无重叠时 speaker 为 null
- [ ] `build_speaker_summary()` 的 `speakers[]` 覆盖全部识别说话人（从未 dominant 者为零值项）、`speakerCount == len(speakers)`、按 totalDuration 降序；`Σ totalDuration ≤ duration` 差值语义已在 spec 声明

## 调度器（多设备 + 异常安全 + 瞬态口径）
- [ ] `estimate_task_memory_mb()` 为**按设备拆分的瞬态公式**（aligner 设备 `512 + 256×align_batch_size`；diarizer 设备 `256 + 0.0625×duration`；同设备求和）——不含常驻权重（不重复计算）
- [ ] 对外提供 `async with scheduler.slot(needs)` 上下文管理器，**内部 try/finally**：任务异常/取消必释放许可，队列不因单次异常永久阻塞
- [ ] `acquire()` 对任务涉及的**每个设备**实测 `torch.cuda.mem_get_info(device)` 并叠加 `gpu_reserve_mb`；任一不足或并发满时全局 FIFO 排队（队首一次检查全部设备，无跨设备死锁），release 后唤醒重检；CPU 设备退化为纯并发限制
- [ ] `stats()` 返回 running/queued + devices[]（device/role/free_mb/total_mb）

## 服务扩展（qwen-asr-serve / vLLM 进程内）
- [ ] serve.py 扩展参数从 argv 剥离，剩余参数保留 vLLM CLI 语义；`--aligner-device`/`--diarizer-device` 默认跟随 vLLM 主设备，**可指定不同设备自由组合（单卡/两卡/三卡拓扑）**
- [ ] **显存预算（按设备）**：仅当扩展与 vLLM 同设备且未显式指定 `gpu_memory_utilization` 时自动注入 0.70（A10 默认双并发自洽：7.2−1.9−1=4.3GB ≥ 2×~2GB；独立设备不注入，vLLM 用满）；扩展模型加载后**每个扩展设备**启动校验——空闲 < 该设备最小瞬态预估 + reserve 时启动失败并输出预算明细与建议值（快速失败，不死锁带病运行）
- [ ] **默认参数启动不 OOM**：`--forced-aligner`/`--diarizer` 默认启用 + 自动调整后的 vLLM 配置可正常启动
- [ ] **多卡拓扑可用**：`--aligner-device cuda:1 --diarizer-device cuda:1` 时扩展模型加载到指定设备、vLLM 不降配、/health/detail 显示按设备信息、segment 请求正常
- [ ] middleware 按参数矩阵路由：segment+缺省/json → 扩展响应；segment+text/verbose_json → 400；word → 400；非法值 → 400；无 segment → OpenAI 标准（json/text/verbose_json）
- [ ] **audio_url SSRF 防护**：仅 https、拒绝环回/私网/链路本地（含 169.254.169.254）/0.0.0.0、DNS 解析后全 IP 校验、下载大小上限、超时；违规 400
- [ ] `min_speakers > max_speakers` → 400
- [ ] ASR 复用 vLLM engine client（不重复加载模型）；middleware 自建 processor（CPU 常驻）；分块/偏移/合并复用 utils 公开函数（无第二份实现）
- [ ] segment 模式响应结构与 spec 完全一致（language 码/duration/text/processTime/segments/speakerSummary），`speakerCount == len(speakers)`
- [ ] **任务内阶段并行**：diarization 线程与 ASR 分块循环同时启动，完成后时间重叠归并；端到端 ≈ max(ASR+对齐, diarize)，两卡拓扑零竞争、单卡交错执行
- [ ] **模型前向串行化**：aligner/diarizer 各由进程级 threading.Lock 保护，并发调用安全
- [ ] 排队等待期间客户端断连 → 任务取消并释放排队位
- [ ] `--diarizer ""`/`--forced-aligner ""` 时不加载扩展、不注入 gpu_memory_utilization、标准转写行为与现状完全一致
- [ ] vLLM 其余端点（/v1/chat/completions、/health 等）零干预
- [ ] 错误码完整：400（参数/时长超限/语言不支持/model 不匹配/min>max/矩阵违规/URL 校验）、415（解码失败）、500（推理异常）、503（扩展未配置/依赖未安装）；错误体 OpenAI 风格
- [ ] 1 小时音频（3600s）完整处理：180s 分块对齐合并 + 整段 diarization + 正确全局时间戳、无 OOM
- [ ] `GET /health/detail` 返回 status/extensionModelsLoaded/runningTasks/queuedTasks/maxConcurrentTasks/gpuReserveMb + **devices[]（device/role/freeVramMb/totalVramMb）**，且文档说明 freeVramMb 语义（已扣 vLLM 预分配与常驻，按设备）

## 示例与验证
- [ ] `examples/example_segment_api.py` 支持 file/audio_url 调用演示、`--self-test` 纯逻辑自测（30 项映射与新口径断言）、文件头含部署要点（nginx 体积/超时、HF_TOKEN/HF_HOME、gpu_memory_utilization 参考配置 A10 0.70/T4 0.55/P4 短音频、**多卡拓扑示例**、ffmpeg 依赖）
- [ ] `--self-test` 全部断言通过（不依赖 GPU/网络）
- [ ] **diarization 性能压测（40min 音频，A10）**：实测单独耗时与端到端耗时、阶段并行生效验证、batch_size 与 `OMP_NUM_THREADS` 调优结论记入部署指南
- [ ] **并发正确性**：并发 N 任务排队可见、全部 200、无 OOM；并发 vs 串行 diarization 结果一致性抽查通过；构造执行中异常后队列仍继续处理（许可不泄漏验证）
- [ ] `pip install -e ".[vllm,diarization]"` 安装成功无冲突
- [ ] 既有功能回归：`Qwen3ASRModel`/`Qwen3ForcedAligner` 导入正常、`qwen-asr-demo`、无扩展参数的 `qwen-asr-serve` 不受影响
- [ ] 新代码风格与仓库一致（Apache-2.0 头、类型注解、docstring）
