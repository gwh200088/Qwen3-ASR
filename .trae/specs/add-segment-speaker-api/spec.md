# Segment 级时间戳与说话人识别 API 扩展 Spec（基于 OpenAI 兼容服务）

## Why

当前 `qwen-asr-serve` 是 vLLM serve 的包装：注册 `Qwen3ASRForConditionalGeneration` 后启动，提供 OpenAI 兼容接口，但只覆盖纯转写。SDK 层已具备转写（`Qwen3ASRModel`）与词级强制对齐（`Qwen3ForcedAligner`，Qwen3-ForcedAligner-0.6B）能力，缺少**说话人识别（pyannote）**与 **segment 级（"谁在何时说了什么"）响应**。

会议、访谈、客服等场景需要：一句话一个分段、每段带起止时间与说话人标签、整体说话人统计；且生产环境 GPU 显存有限（A10 24GB / T4 16GB / P4 8GB），高并发时长音频任务必须做显存感知调度，避免 OOM。

**架构决策（用户确认）**：不新建独立服务，**扩展现有 `qwen-asr-serve`**。请求遵循 OpenAI `/v1/audio/transcriptions` 规范（multipart form），通过 `timestamp_granularities` 参数（含 `"segment"`）控制是否启用 segment 时间戳 + 说话人识别。

## What Changes

- **MODIFIED**：`qwen_asr/cli/serve.py` —— 由 vLLM CLI 薄封装改为"组装式"入口：构建 vLLM OpenAI app 后挂载 ASGI middleware，接管 `POST /v1/audio/transcriptions`：
  - 请求带 `timestamp_granularities` 含 `"segment"` → 扩展管线：vLLM 引擎 ASR + 强制对齐 + pyannote 说话人归属 + 切分汇总 → 本 spec 定义的结构化 JSON
  - 不带 → OpenAI 标准行为（`response_format=json|text|verbose_json`），仅 vLLM 引擎 ASR
  - 同进程加载 `Qwen3ForcedAligner` 与 `SpeakerDiarizer`，ASR 复用 vLLM server 已加载的引擎（**不重复加载模型、不二次占显存**）
- **MODIFIED**：`qwen_asr/inference/qwen3_asr.py` —— 将私有方法 `_offset_align_result` / `_merge_align_results` 的实现提取为 `qwen_asr/inference/utils.py` 中的**公开模块级函数** `offset_align_result()` / `merge_align_results()`，私有方法改为薄委托（对外行为不变）。middleware 复用公开函数，避免分块/偏移/合并逻辑出现第二份实现（DRY）
- **NEW**：`qwen_asr/service/` 包（服务扩展逻辑模块，非独立服务）：
  - `middleware.py`：ASGI middleware——拦截 `/v1/audio/transcriptions`，multipart 解析、参数矩阵路由、音频获取与 SSRF 校验、错误统一映射；GPU 推理经调度器在线程池执行
  - `pipeline.py`：Segment 转写管道纯逻辑（对齐 token → 句级切分 → 说话人归属 → speakerSummary 汇总，可单测）
  - `scheduler.py`：GPU 显存感知任务调度器（许可获取/释放异常安全，管控对齐 + diarization 阶段的并发与排队）
- **NEW**：`qwen_asr/inference/qwen3_speaker_diarizer.py`：`SpeakerDiarizer` 类（封装 pyannote.audio 4.0.7，默认管线 `pyannote/speaker-diarization-community-1`，兼容 legacy 3.1），API 风格参照 `Qwen3ForcedAligner`（`from_pretrained()` 工厂 + `diarize()` 方法）
- **NEW**：`examples/example_segment_api.py`：OpenAI 兼容调用示例 + 管道纯逻辑自测（`--self-test`）
- **MODIFIED**：`qwen_asr/__init__.py` 导出 `SpeakerDiarizer`；顺带修复现存缺陷：`__all__ = ["__version__"]` 引用了未定义的 `__version__`（`from qwen_asr import *` 会 AttributeError）——改为以 `importlib.metadata` 取包版本（回退 `"0.0.0"`），`__all__` 列出实际导出符号
- **MODIFIED**：`pyproject.toml` 新增 `diarization = ["pyannote.audio==4.0.7"]` 可选依赖组（精确 pin 当前最新开源版；fastapi/uvicorn/multipart 由 vllm 依赖带入）
- **MODIFIED**：`docker/Dockerfile-qwen3-asr-cu128` 安装 `diarization` 依赖、支持 HF token 传入与模型缓存预下载（含 pyannote community-1）。注：ffmpeg 已在现 Dockerfile 安装（torchcodec 依赖），无需改动

## 关键设计决策

### 版本选型与依赖兼容（含风险验证）
- 对齐：`Qwen/Qwen3-ForcedAligner-0.6B`（项目既有 `Qwen3ForcedAligner`；其 `align()` 不分块、单次输入上限 180s——由 `utils.MAX_FORCE_ALIGN_INPUT_SECONDS` 定义，分块由调用层负责：SDK 侧 `Qwen3ASRModel.transcribe()`、服务侧 middleware 均按 180s 分块后调用）
- 说话人：**`pyannote.audio==4.0.7`**（当前最新开源版，2026-06-30 发布，PyPI 官方确认）+ 默认管线 **`pyannote/speaker-diarization-community-1`**（最新开源管线，VBx 聚类，替代 legacy 3.1）。选型依据（DER%，越低越好，官方 2025-09 benchmark）：AliMeeting（中文会议）24.5→**20.3**、AISHELL-4 12.2→11.7、AMI IHM 18.8→17.0、CALLHOME 28.5→26.7、DIHARD 21.4→20.2——中文场景收益显著。速度（官方自托管数据，H100 80GB）：community-1 约 **31s/小时音频**（AMI 1h 文件）、37s/h（DIHARD 5min 文件）；用户硬件（A10/T4）实际速度需 Task 7 压测确认。HF 门控模型需 token（接受 user conditions 后 `hf.co/settings/tokens` 创建）
  - `--diarizer` 参数可配置：也支持回退 `pyannote/speaker-diarization-3.1`（legacy）或本地路径
  - pyannote 4.0.4 起放宽 torch 版本约束（利于与 vLLM 共存）；4.x 依赖 `torchcodec`（需系统 ffmpeg）
- **依赖兼容为前置验证项（Task 1）**：pyannote 4.0.7 依赖链（torchcodec/torchaudio/speechbrain/asteroid-filterbanks）与 `vllm==0.14.0`、`transformers==4.57.6` 锁定的 torch 版本链，必须在同一虚拟环境实测 `import vllm + pyannote.audio + transformers + torchcodec`、ffmpeg 可用性及最小推理全部通过，方可进入后续开发。若实测不兼容，降级预案：将 diarization 拆为独立进程 worker（通过共享存储交换音频/结果），或回退 legacy `pyannote.audio==3.1.1`，该预案仅在验证失败时启用

### pyannote 集成防御性设计（兼容 3.x/4.x）
- **认证**：`SpeakerDiarizer.from_pretrained(pretrained_model_name_or_path, use_auth_token=None, device=None, **kwargs)` 采用显式常用参数的签名（比 `Qwen3ForcedAligner.from_pretrained(path, **kwargs)` 的纯 kwargs 风格更直白，属有意扩展），内部传给 pyannote 时使用 `token=` 参数（新版 huggingface_hub 已移除 `use_auth_token` 透传），并同时设置 `HF_TOKEN` 环境变量双保险
- **返回类型兼容**：`diarize()` 内部对 pipeline 返回值做防御性归一——若对象有 `speaker_diarization` 属性（pyannote 4.x `DiarizeOutput`）则取之，否则视为 `Annotation`（3.x）直接使用；再统一 `itertracks(yield_label=True)` 收集片段（因此 `--diarizer` 指向 3.1 或 community-1 均可工作）
- **说话人数约束**：`min_speakers`/`max_speakers` 透传 pipeline；pyannote 4.x 对不支持该参数的管线会降级为警告而非报错（4.0.1 changelog），spec 接受此 best-effort 行为
- **并发安全**：aligner 与 diarizer 模型前向调用分别由进程级 `threading.Lock` 串行化（pyannote Pipeline 与 transformers 模型均无并发调用安全保证）；任务级并发收益来自 ASR（vLLM continuous batching）与不同阶段（A 任务 diarization 时 B 任务可对齐）的流水重叠；**波形上传 GPU 在锁内进行**（等待前向的任务仅持 CPU 张量，限制每设备瞬态显存仅一个前向）
- **任务内阶段并行（性能设计）**：diarization 不依赖 ASR/对齐结果，segment 模式下在启动 ASR 分块循环的**同时**并行提交 diarization 线程，两者完成后仅做时间重叠归并——端到端耗时由 `sum(ASR+对齐, diarize)` 降为 `max(两者)+归并`。两卡拓扑零竞争真并行；单卡亦可（vLLM 0.14 引擎核心在独立进程，与 middleware 进程的 pyannote 可交错执行，仅有 SM 竞争）。对齐依赖 ASR 输出，保持串行

### 多 GPU 部署与设备拓扑（自由组合）

ASR（vLLM 引擎）、强制对齐（aligner）、说话人识别（diarizer）三者**不要求同卡**，通过参数自由组合：

| 拓扑 | 配置方式 | 说明 |
|---|---|---|
| 单卡（默认） | 不传 device 参数 | 三者均在 vLLM 主设备（cuda:0）；自动注入 `gpu_memory_utilization=0.70` |
| 两卡（推荐生产） | `--aligner-device cuda:1 --diarizer-device cuda:1` | vLLM 独占 GPU0（可用满 0.9），扩展模型在 GPU1；vLLM 设备无需降配 |
| 三卡 | `--aligner-device cuda:1 --diarizer-device cuda:2` | 各自独占，显存互不竞争 |
| 任意混合 | 如 aligner 与 vLLM 同卡、diarizer 独立 | 按设备分别准入（见调度） |

- **vLLM 引擎设备**由 vLLM 自身参数控制（`CUDA_VISIBLE_DEVICES`/`--device`），serve.py 不干预
- **自动注入规则细化**：仅当 aligner/diarizer 设备与 vLLM 主设备相同（默认情形）时才自动注入 0.70；显式指定不同设备则不注入（vLLM 可用满），并对扩展设备单独做启动校验
- **按设备准入**：调度器对扩展涉及的每个设备分别用 `torch.cuda.mem_get_info(device)` 校验空闲；任务的瞬态需求按设备拆分（见预算公式）
- **无死锁保证**：全局单一 FIFO 队列，队首任务一次性检查其涉及的全部设备，任一不满足则继续等待（队首阻塞换取公平与无死锁，v1 接受并在文档注明）
- CPU 设备（`cpu`）退化为纯并发数限制

### 显存预算方案（按设备，核心，避免死锁与启动 OOM）

显存分四层预算，**按设备**启动校验、运行时准入：

| 层 | 内容 | 量级（估算） |
|---|---|---|
| ① vLLM 引擎池 | `gpu_memory_utilization` 预分配（vLLM 设备） | 由启动参数决定 |
| ② 扩展模型常驻 | aligner 权重（aligner 设备）+ diarizer 权重（diarizer 设备），启动加载常驻 | ~1.2GB + ~0.7GB |
| ③ 任务瞬态 | 对齐激活（aligner 设备）+ diarization 波形/embedding 工作区（diarizer 设备） | 见下方公式 |
| ④ 安全余量 | `--gpu-reserve-mb`（默认 1024，每设备） | 1GB |

- **每任务瞬态显存预估（按设备拆分，不含常驻权重——权重启动后已从对应设备空闲显存扣除，不得重复计算）**：
  - aligner 设备：`align_mb = 512(固定工作余量) + 256×align_batch_size`
  - diarizer 设备：`diar_mb = 256(工作区) + 0.0625MB/s×音频时长`
  - 两设备相同则合并求和。例：1h 音频、align_batch=4 → 对齐侧 ~1.5GB、说话人侧 ~0.5GB；同卡合计 ~2.0GB
- **`gpu_memory_utilization` 自动调整**：仅当 aligner/diarizer 设备与 vLLM 主设备相同（默认单卡情形）且用户未显式指定该参数时，自动注入 **0.70** 并打日志（取 0.70 的依据：A10 24GB 下 vLLM 占 16.8GB、余 7.2GB，扣除常驻 1.9GB + reserve 1GB 后剩 4.3GB，恰好覆盖默认 `max_concurrent_tasks=2` × 1h 任务瞬态 2×~2GB——默认配置自洽）；扩展模型在独立设备时不注入（vLLM 可用满默认 0.9）；用户显式指定则尊重用户值
- **启动时快速失败校验（按设备）**：扩展模型加载完成后，对**每个扩展设备**校验 `mem_get_info(device)` 实测空闲 ≥ 该设备单任务最小瞬态预估（按 `max_concurrent_tasks=1`、`align_batch_size`、30s 音频计算）+ `gpu_reserve_mb`，不满足则**启动失败**并输出可操作指引（该设备各层预算占用明细、建议的 `gpu_memory_utilization` 或改用独立设备的参数示例），绝不带病进入"必然死锁"状态
- **运行时准入（scheduler）**：全局 FIFO，队首任务对其涉及的每个设备检查 `mem_get_info(device)` 空闲 ≥ 该设备需求 + `gpu_reserve_mb`，且运行数 < `--max-concurrent-tasks`，全部满足才放行；任一不满足继续等待，释放时唤醒队首重检（队首阻塞换取公平与无死锁）
- 参考配置（单卡，预算含 reserve 1GB）：
  - A10 24GB → **0.70**（默认）：vLLM 16.8GB，余 7.2GB = 常驻 1.9 + reserve 1.0 + 2 任务瞬态 ~4.1 → **双并发 40min/1h 任务可行**；若仅单并发可显式提到 0.75（余 6GB：2.9 固定 + 单任务 ~2GB ✓，第二任务排队）
  - T4 16GB → 双并发长音频需 **0.55**（余 7.2GB 同上）；单并发 0.60（余 6.4GB：2.9 + 单任务 ~2 ✓）
  - P4 8GB → **不推荐**承载长音频 segment 任务：即便 0.35（余 5.2GB）+ 单并发 + `gpu_reserve_mb=512`，40min 任务（1.9+0.5+1.9=4.3GB）仅勉强放下，无任何裕量；建议 P4 仅做纯转写或短音频（≤10min：瞬态 ~1.6GB）
- 参考配置（两卡）：GPU0 vLLM `gpu_memory_utilization=0.9` + GPU1 扩展（`--aligner-device cuda:1 --diarizer-device cuda:1`），16GB 卡即可舒适承载

### 服务架构（vLLM 进程内扩展）
- `serve.py` 流程：注册模型 → 解析并剥离扩展参数 → 构建 vLLM OpenAI app（适配 vLLM 0.14.0 app 构建 API，Task 5 首先调研验证）→ 挂 ASGI middleware → 启动
- **ASR 调用路径**：middleware 通过 vLLM app 的 engine client（`app.state.engine_client`）发起生成；prompt 构造需要 processor——middleware 启动时自行实例化 `Qwen3ASRProcessor.from_pretrained`（仅 tokenizer/特征提取器，CPU 常驻，不占显存），chat template 构造逻辑对齐 `Qwen3ASRModel._build_text_prompt`
- **分块/偏移/合并复用**：音频 180s 分块复用 `utils.split_audio_into_chunks`（公开函数）；对齐结果的偏移修正与合并复用提取后的 `utils.offset_align_result()` / `utils.merge_align_results()`（见 What Changes）
- middleware 全权接管 `/v1/audio/transcriptions`（含透传分支）；其余路径（`/v1/chat/completions`、`/health` 等）零干预
- 对齐器 + diarizer 启动时同步加载（配置了才加载）；未配置时 segment 模式请求返回 503 并提示启动参数

### 请求参数矩阵（timestamp_granularities × response_format）

| timestamp_granularities | response_format | 行为 |
|---|---|---|
| 缺省 / 不含 `"segment"` | `json`（默认） | OpenAI 标准：`{"text": "..."}` |
| 缺省 / 不含 `"segment"` | `text` | OpenAI 标准：纯文本 |
| 缺省 / 不含 `"segment"` | `verbose_json` | OpenAI 标准：`{"text", "duration", "language"}` |
| 含 `"segment"` | 缺省 或 `json` | **本 spec 扩展响应**（segment + speaker + speakerSummary） |
| 含 `"segment"` | `text` 或 `verbose_json` | 400：`segment` 粒度与该 response_format 不兼容，请移除或改用 json |
| 含 `"word"` | 任意 | 400：v1 不支持 word 粒度（错误信息说明支持 segment） |
| 含其他非法值 | 任意 | 400：非法粒度值 |

**与 OpenAI 标准的已知差异（明示文档化）**：OpenAI 规范中 `segment` 粒度配 `verbose_json` 返回无说话人的标准 segments；本服务将 `segment` 粒度定义为"segment + 说话人扩展"语义。此为用户确认的产品决策，需在 README/示例中明示。

### 长音频（最长 1 小时）
- ASR + 对齐：180s 分块 + 偏移合并（复用公开函数）；1h ≈ 20 块
- pyannote 对整段音频做一次 diarization（1h 波形 float32 约 230MB，计入瞬态预算）
- 服务端 `--max-audio-seconds`（默认 3600）超限返回 400
- **40 分钟音频走查（用户确认的典型用例）**：2400s < 3600s 默认上限 ✓；ASR/对齐 14 块；diarization 波形约 154MB；瞬态显存（同卡）约 512+256×4+256+0.0625×2400 ≈ **1.9GB**。A10 单卡 @0.70 预算验算：余 7.2GB − 常驻 1.9 − reserve 1.0 = 4.3GB 可用瞬态 → **单任务充裕（1.9GB）、双并发可行（3.9GB < 4.3GB，裕量仅 ~0.4GB，估算偏差时第二任务由 FIFO 排队兜底）**；两卡方案扩展侧独享整卡，无此约束。diarization 耗时：官方 H100 31s/h ≈ 21s，A10 现实预期 **1~2.5 分钟**（神经前向受 GPU 算力/带宽限制放大，VBx 聚类为 CPU 迭代不受 GPU 加速；准确数字待 Task 7 压测）；**任务内阶段并行后**端到端 ≈ max(ASR+对齐, diarize)，估计 2.5~5 分钟，在推荐代理超时 900s 内。性能调优项（Task 7 验证）：分割/embedding 推理 batch_size（32~64，A10 显存充裕）、`OMP_NUM_THREADS` 合理化（VBx 聚类吃单核）；实验项：fp16 推理（需验证 DER 无退化）；**不做**音频切块并行 diarization（跨块说话人标签对齐正确性风险高）
- v1 为同步响应；1h 音频端到端估计 3~10 分钟，**部署指南**（见下）给出代理超时与上传体积配置要求；**排队等待期间客户端断连时任务取消**（释放排队位，不占调度许可），执行中断连则尽力取消。异步任务模式（提交→轮询）列为 future work，不在 v1 范围

### audio_url 安全（SSRF 防护）
- 仅允许 `https://` 协议（拒绝 `http`/`file`/`ftp` 等）
- 解析主机名后拒绝：环回地址、私网段（RFC 1918/4193）、链路本地（169.254.0.0/16，含云元数据 169.254.169.254）、0.0.0.0；DNS 解析后对全部 IP 校验（防 DNS rebinding 基础形态）
- 下载大小上限 `--max-audio-bytes`（默认 500MB），超限即中止并 400；下载超时 60s
- `file`（上传）与 `audio_url` 缺失或同时提供 → 400

### Segment 切分与说话人归属
- 句级切分基于对齐 token 序列：相邻 token 时间间隙 ≥ 0.8s（`--segment-gap-threshold`，可配）、或段长达到 30s（`--max-segment-seconds`，可配）强制切分；段文本通过游标匹配从原始 ASR 文本截取（保留标点），匹配失败回退为 token 拼接
- 说话人归属：对每段 `[start, end]` 与各说话人 diarization 片段计算时间重叠总时长；**重叠时长最大者为该段 `speaker`**（用户规则 3）；重叠 ≥ 0.1s 的说话人全部列入 `speakers`（按重叠时长降序）；无任何重叠时 `speaker=null, speakers=[]`
- `min_speakers > max_speakers` → 400 校验错误

### speakerSummary 口径（修正版，内部自洽）
- `speakers[]`：列出 **diarization 识别出的全部说话人**——从未 dominant 的说话人 `totalDuration=0, segmentCount=0`，保证 `speakerCount == len(speakers)`
- `totalDuration`：该说话人作为 dominant 的 segment 时长之和；`segmentCount`：对应 segment 数；按 `totalDuration` 降序
- `speakerCount`：`len(speakers)`（= diarization 识别总数）
- **显式声明差值语义**：`Σ totalDuration ≤ duration`，差值来自 `speaker=null` 的 segment（无人说话归属的段落，如纯音乐/静音上误出的文本段）

### 语言码映射（完整表，30 项）
内部语言名 ↔ BCP-47 风格码，覆盖 `SUPPORTED_LANGUAGES` 全部 **30** 项；请求入参 `language` 同时接受 ISO 码与语言名（双向映射）；未匹配项回退小写全名。

| 内部名 | 码 | 内部名 | 码 | 内部名 | 码 |
|---|---|---|---|---|---|
| Chinese | zh | Korean | ko | Swedish | sv |
| English | en | Russian | ru | Danish | da |
| Cantonese | yue | Thai | th | Finnish | fi |
| Arabic | ar | Vietnamese | vi | Polish | pl |
| German | de | Japanese | ja | Czech | cs |
| French | fr | Turkish | tr | Filipino | fil |
| Spanish | es | Hindi | hi | Persian | fa |
| Portuguese | pt | Malay | ms | Greek | el |
| Indonesian | id | Dutch | nl | Romanian | ro |
| Italian | it | Hungarian | hu | Macedonian | mk |

### 命名风格
响应中 `processTime`/`speakerSummary`/`speakerCount`/`totalDuration`/`segmentCount` 为 camelCase，与 OpenAI 生态 snake_case 不同——此为**用户给定的响应格式，刻意保留**，文档明示即可。

## Impact

- Affected specs: 无（本 spec 为新增能力，不修改其他 spec）
- Affected code:
  - `qwen_asr/cli/serve.py`（扩展为组装式入口 + 显存预算校验）
  - `qwen_asr/service/__init__.py`、`middleware.py`、`pipeline.py`、`scheduler.py`（新文件）
  - `qwen_asr/inference/qwen3_speaker_diarizer.py`（新文件）
  - `qwen_asr/inference/qwen3_asr.py` + `qwen_asr/inference/utils.py`（私有方法提取为公开函数，行为不变）
  - `qwen_asr/__init__.py`（导出 `SpeakerDiarizer`）
  - `pyproject.toml`（diarization 依赖组，pin 4.0.7）
  - `examples/example_segment_api.py`（新文件）
  - `docker/Dockerfile-qwen3-asr-cu128`（依赖、token、模型缓存）
- 对既有功能零破坏：`timestamp_granularities` 不含 `"segment"` 时行为与 OpenAI 标准一致；`--diarizer ""`/`--forced-aligner ""` 显式禁用时 `qwen-asr-serve` 行为与现状完全一致（不加载扩展、不注入 gpu_memory_utilization）；`qwen3_asr.py` 的提取重构对外行为不变；`qwen-asr-demo`、`qwen-asr-demo-streaming` 不受影响

## 部署指南（随 README/示例交付）

- **反向代理**（nginx 示例）：`client_max_body_size 500m`（1h wav ~230MB 上传）；`proxy_read_timeout 900s`（长音频端到端）；`proxy_send_timeout 900s`
- **HF 模型缓存**：pyannote 模型门控，需 `HF_TOKEN`/`PYANNOTE_API_TOKEN`；Docker 挂载 `HF_HOME` 卷缓存模型避免每次启动下载；Dockerfile 支持构建期预下载（ASR/aligner/pyannote）
- **显存配置**：按"参考配置"表设置 `gpu_memory_utilization`；多进程共卡时 `--gpu-reserve-mb` 相应调大

## API 定义

### 请求

`POST /v1/audio/transcriptions`，`multipart/form-data`（OpenAI 规范 + 扩展字段）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | 与 `audio_url` 二选一 | 音频文件上传 |
| `audio_url` | string | 与 `file` 二选一 | 扩展字段：HTTPS URL（经 SSRF 校验与大小限制，大文件推荐） |
| `model` | string | 是 | 须匹配 `--served-model-name` |
| `language` | string | 否 | ISO 码（zh/en）或语言名（Chinese），双向映射 |
| `prompt` | string | 否 | 上下文提示（映射到 ASR context） |
| `response_format` | string | 否 | `json`（默认）/ `text` / `verbose_json`；与 segment 粒度的组合见参数矩阵 |
| `timestamp_granularities[]` | string[] | 否 | OpenAI 规范参数；**含 `"segment"` 时启用 segment 时间戳 + 说话人识别**（核心开关，组合矩阵见上） |
| `min_speakers` / `max_speakers` | int | 否 | 扩展字段：透传 pyannote；`min > max` → 400 |

### 响应（timestamp_granularities 含 "segment"，即 segment 模式）

```json
{
  "language": "zh",
  "duration": 12.345,
  "text": "识别出的完整文本内容...",
  "processTime": 5.678,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "分段文本",
      "speaker": "SPEAKER_00",
      "speakers": ["SPEAKER_00"]
    }
  ],
  "speakerSummary": {
    "speakerCount": 2,
    "speakers": [
      {"id": "SPEAKER_00", "totalDuration": 8.5, "segmentCount": 3}
    ]
  }
}
```

- `duration`：音频时长（秒，3 位小数）
- `processTime`：服务端总耗时（秒，3 位小数，含排队等待）
- `segments` 按 `start` 升序；`start/end` 为 3 位小数
- `speakerSummary.speakers` 含全部识别说话人（含零值项），按 `totalDuration` 降序；`speakerCount == len(speakers)`

### 响应（非 segment 模式，OpenAI 标准）

见参数矩阵前三行：`json`→`{"text"}`、`text`→纯文本、`verbose_json`→`{"text", "duration", "language"}`。

### 错误响应

OpenAI 风格：`{"error": {"message": "...", "type": "invalid_request_error | not_found_error | server_error | service_unavailable", "code": null}}`，配合 400（参数/格式/时长超限/语言不支持/model 不匹配/min>max/不支持的参数组合/audio_url 校验失败）、415（音频解码失败）、500（模型推理异常）、503（扩展未配置/依赖未安装/未授权）。

### 辅助端点

`GET /health/detail`：

```json
{
  "status": "ok",
  "extensionModelsLoaded": true,
  "runningTasks": 1,
  "queuedTasks": 0,
  "maxConcurrentTasks": 2,
  "gpuReserveMb": 1024,
  "devices": [
    {"device": "cuda:0", "role": "vllm+aligner+diarizer", "freeVramMb": 4200, "totalVramMb": 24564},
    {"device": "cuda:1", "role": "diarizer", "freeVramMb": 15200, "totalVramMb": 24564}
  ]
}
```

**`devices[].freeVramMb` 语义说明**：`mem_get_info(device)` 实测的该设备进程外空闲显存，**已扣除 vLLM 预分配与扩展模型常驻**——正常态下该值 ≈ reserve + 剩余任务瞬态预算（如 A10 单卡 0.70 配置下空载约 4.3GB + reserve 1GB ≈ 5.3GB），并非"显卡总空闲"；多进程共卡时该值含其他进程占用影响。`role` 标明该设备承载的角色组合。运维基线：服务空载时每个扩展设备 `freeVramMb` 应 ≥ 该设备单任务最小瞬态预估 + `gpuReserveMb`，否则调度必然排队。

## ADDED Requirements

### Requirement: SpeakerDiarizer SDK 接口

系统 SHALL 提供 `SpeakerDiarizer` 类，封装 pyannote.audio 说话人识别 Pipeline，采用 `from_pretrained()`（显式 `use_auth_token`/`device` 常用参数）加载与 `diarize()` 推理，接口风格参照 `Qwen3ForcedAligner`；认证使用 `token=` 传递并设置 `HF_TOKEN` 环境变量；返回值经防御性归一兼容 pyannote 3.x（Annotation）与 4.x（DiarizeOutput）。

#### Scenario: 单条与批量说话人识别
- **WHEN** 调用 `SpeakerDiarizer.from_pretrained("pyannote/speaker-diarization-community-1", use_auth_token=..., device="cuda:1")` 后调用 `diarize(audio)`
- **THEN** 返回 `List[DiarizationResult]`（长度与输入一致），`segments` 为按时间排序的 `DiarizationSegment(speaker, start_time, end_time)` 列表，`speaker` 形如 `SPEAKER_00`，单位秒

#### Scenario: 兼容 legacy 管线
- **WHEN** `from_pretrained("pyannote/speaker-diarization-3.1", ...)`（legacy）
- **THEN** 同样返回归一化后的 `DiarizationResult`（防御性归一屏蔽 3.x/4.x 返回类型差异）

#### Scenario: 音频归一化与说话人数约束
- **WHEN** 传入路径/URL/base64/(ndarray,sr) 任意组合，及 `min_speakers`/`max_speakers`
- **THEN** 自动复用 `normalize_audios()` 归一化为 16k 单声道 float32 后推理，说话人数量约束 best-effort（pyannote 4.x 对不支持的管线参数降级为警告）

#### Scenario: 未安装 pyannote
- **WHEN** 未安装 `pyannote.audio` 时调用 `from_pretrained`
- **THEN** 抛 `ImportError` 并提示 `pip install qwen-asr[diarization]`

### Requirement: OpenAI 兼容 transcriptions 端点扩展

系统 SHALL 通过扩展 `qwen-asr-serve`（vLLM 进程内 middleware）接管 `POST /v1/audio/transcriptions`：按参数矩阵路由，`timestamp_granularities` 含 `"segment"` 时返回 segment 级时间戳 + 说话人归属 + speakerSummary 的结构化 JSON；否则返回 OpenAI 标准响应。

#### Scenario: segment 模式成功转写
- **WHEN** 客户端以 multipart form 提交多人对话音频，带 `timestamp_granularities[]=segment`
- **THEN** 返回 200，body 符合上述响应结构：`segments` 升序、每段含 `start/end/text/speaker/speakers`、`speakerCount == len(speakers[])`

#### Scenario: 参数矩阵校验
- **WHEN** 请求组合为 `segment+text`、`word`（任意）、或非法粒度值
- **THEN** 分别返回 400，错误信息说明支持的组合

#### Scenario: OpenAI 标准模式（兼容）
- **WHEN** 不带 `timestamp_granularities`（或不含 `"segment"`）
- **THEN** 按 `response_format` 返回 OpenAI 标准响应，不触发对齐与 diarization

#### Scenario: audio_url 安全校验
- **WHEN** `audio_url` 为非 https 协议、或解析到内网/环回/链路本地地址、或下载超过 `--max-audio-bytes`
- **THEN** 返回 400，不做下载或推理

#### Scenario: 长音频支持与超限拒绝
- **WHEN** 提交 ≤ 3600s（`--max-audio-seconds` 可配）音频
- **THEN** 正常完成 180s 分块转写对齐 + 整段 diarization 并返回结果
- **WHEN** 提交超限时长音频
- **THEN** 返回 400 与明确错误信息，不加载入显存

#### Scenario: 扩展未配置时的降级
- **WHEN** 服务未以 `--diarizer`/`--forced-aligner` 启动，收到 segment 模式请求
- **THEN** 返回 503，错误信息指明所需启动参数

### Requirement: GPU 显存预算与多设备调度

系统 SHALL 支持单卡/多卡自由组合的设备拓扑（vLLM 引擎、aligner、diarizer 可分别指定设备），在启动时按设备执行显存预算校验（不满足则快速失败），在执行对齐 + diarization 前按设备评估瞬态显存需求；vLLM 引擎显存为启动时预分配，扩展模型常驻显存不重复计入任务需求。

#### Scenario: 启动预算校验（按设备）
- **WHEN** 任一扩展设备在扩展模型加载后空闲显存 < 该设备单任务最小瞬态预估 + `gpu_reserve_mb`
- **THEN** 启动失败，输出该设备各层预算占用与建议（`gpu_memory_utilization` 或改用独立设备的参数示例）

#### Scenario: gpu_memory_utilization 自动调整
- **WHEN** aligner/diarizer 与 vLLM 同设备（默认单卡）且用户未显式指定 `gpu_memory_utilization`
- **THEN** 自动注入 0.70 并打日志（依据见预算方案）；扩展模型在独立设备时不注入（vLLM 用默认 0.9）；用户显式指定时尊重用户值

#### Scenario: 多设备准入
- **WHEN** 任务涉及的任一设备空闲显存 < 该设备需求 + 安全余量，或并发已满
- **THEN** 该任务在全局 FIFO 中等待（队首一次性检查全部涉及设备）；前序任务完成后被唤醒重检执行，期间服务不 OOM、不拒绝请求、无跨设备死锁

#### Scenario: 许可异常安全
- **WHEN** 已获取调度许可的任务在任意阶段抛出异常（对齐失败/diarization 失败/客户端断连取消）
- **THEN** `try/finally` 保证许可必然释放，队列不因单任务异常永久阻塞

#### Scenario: 模型前向串行化
- **WHEN** 2 个并发任务同时到达 aligner/diarizer 前向
- **THEN** 各自由进程级锁串行执行，无并发调用导致的崩溃或结果错乱；等待前向的任务仅持 CPU 张量

#### Scenario: 调度状态可见性
- **WHEN** 调用 `GET /health/detail`
- **THEN** 返回按设备的 freeVramMb/totalVramMb/role 及全局 running/queued 等字段

### Requirement: 说话人归属与统计口径

#### Scenario: 一段多说话人取占比最大者
- **WHEN** 某 segment 时间范围内 SPEAKER_00 重叠 2.0s、SPEAKER_01 重叠 0.5s
- **THEN** `speaker="SPEAKER_00"`，`speakers=["SPEAKER_00","SPEAKER_01"]`（按重叠降序）

#### Scenario: speakerSummary 口径
- **WHEN** 汇总输出
- **THEN** `speakers[]` 覆盖全部识别说话人（从未 dominant 者为零值项），`speakerCount == len(speakers)`；每说话人 `totalDuration` 等于其 dominant 段时长之和；`Σ totalDuration ≤ duration`（差值为 speaker=null 段）

## MODIFIED Requirements

### Requirement: qwen-asr-serve 入口行为

`qwen-asr-serve` SHALL 在原有 vLLM CLI 语义之上新增扩展参数（先解析剥离再转发 vLLM）：`--forced-aligner`(默认 `Qwen/Qwen3-ForcedAligner-0.6B`)、`--diarizer`(默认 `pyannote/speaker-diarization-community-1`，可配 legacy `pyannote/speaker-diarization-3.1` 或本地路径)、`--pyannote-token`(或环境变量 `PYANNOTE_API_TOKEN`/`HF_TOKEN`)、`--aligner-device`(默认跟随 vLLM 主设备 cuda:0)、`--diarizer-device`(默认跟随 vLLM 主设备 cuda:0；三者可分别指定不同设备自由组合)、`--max-concurrent-tasks`(默认 2)、`--gpu-reserve-mb`(默认 1024，每设备)、`--max-audio-seconds`(默认 3600)、`--max-audio-bytes`(默认 500MB)、`--segment-gap-threshold`(默认 0.8)、`--max-segment-seconds`(默认 30)、`--align-batch-size`(默认 4)。`--diarizer ""`/`--forced-aligner ""` 显式禁用时：不加载扩展、不注入 gpu_memory_utilization、行为与现状完全一致。

### Requirement: 对齐结果偏移/合并逻辑公开化

`qwen_asr/inference/qwen3_asr.py` 的 `_offset_align_result`/`_merge_align_results` 实现 SHALL 提取为 `qwen_asr/inference/utils.py` 公开函数 `offset_align_result()`/`merge_align_results()`，原私有方法薄委托保持对外行为不变；middleware 与后续调用方 SHALL 复用公开函数。

### Requirement: 包依赖与导出

`pyproject.toml` SHALL 新增 `diarization = ["pyannote.audio==4.0.7"]` 可选依赖组；`qwen_asr/__init__.py` SHALL 导出 `SpeakerDiarizer`。既有依赖、脚本入口不变。

## REMOVED Requirements

无。（独立 FastAPI 服务方案已被用户否决；v1 不含异步任务模式与 word 粒度。）
