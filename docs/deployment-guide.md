# Qwen3-ASR segment 时间戳 + 说话人识别服务 · 部署操作手册

> 推荐镜像：`qwen3-asr-offline:cu128-hotfix`（基础镜像 + 3 项实机运行时修复，2026-08-20 A10 实机验证通过）
> 基础镜像：`qwen3-asr-offline:cu128`（基于 `dev` 分支构建，`BUNDLE_FLASH_ATTENTION=false`）
> 部署模式：**完全离线**（模型本地目录挂载，全程零网络）
> 服务能力：vLLM ASR 转写 + segment 级时间戳（强制对齐）+ pyannote 说话人识别，支持最长 1 小时音频

---

## 1. 服务能力概览

| 能力 | 说明 |
|---|---|
| ASR 转写 | Qwen3-ASR-1.7B，30 种语言（中/英/粤/日/韩/法/德 等），vLLM 推理 |
| segment 时间戳 | Qwen3-ForcedAligner-0.6B 强制对齐，输出句级 start/end |
| 说话人识别 | pyannote community-1 管线（VBx 聚类 + PLDA），输出 speaker 标签 |
| 最大音频时长 | 3600 秒（1 小时），由 `--max-audio-seconds` 控制 |
| 并发 | 显存感知调度：显存足够则并发（默认双并发），不足自动排队 |
| 接口 | OpenAI 兼容 `POST /v1/audio/transcriptions`（multipart） |
| 部署形态 | Docker 单容器，三模型同卡或分卡拓扑 |

---

## 2. 镜像版本说明（重要）

| 镜像 | 内容 | 用途 |
|---|---|---|
| `qwen3-asr-offline:cu128` | 基础镜像（Dockerfile-qwen3-asr-cu128 完整构建） | 构建产物基座 |
| `qwen3-asr-offline:cu128-hotfix` | 基础镜像 + 3 个运行时修复（Dockerfile-qwen3-asr-hotfix） | **生产部署用这个** |

### 2.1 hotfix 镜像包含的修复（2026-08-20 GPU 实机首跑发现）

| # | 修复 | 症状（不修复时的表现） |
|---|---|---|
| fix1 | `qwen3_speaker_diarizer.py`：pyannote.audio 4.x `Pipeline.to()` 要求 `torch.device` 实例，统一转换 | 启动即 `TypeError: 'device' must be an instance of 'torch.device', got 'str'` |
| fix2 | `serve.py`：vLLM 0.14.0 SageMaker bootstrap 预建中间件栈，改为手动插入 + 重建栈 | 启动即 `RuntimeError: Cannot add middleware after an application has started` |
| fix3 | `middleware.py`：segment 任务收尾在调度许可释放前 `torch.cuda.empty_cache()` 归还缓存分配器空闲块 | **首个请求 200 OK，之后所有请求永久排队**（详见 §11 故障排查 #3） |

### 2.2 镜像内容判别

```bash
docker run --rm --entrypoint sh qwen3-asr-offline:cu128-hotfix -c \
  "grep -c '_release_gpu_cache' /usr/local/lib/python3.10/dist-packages/qwen_asr/service/middleware.py"
# 输出 2 = 已含 fix3；报错/输出 0 = 旧镜像
```

### 2.3 只有基础镜像时的临时热补丁（不重打镜像，应急用）

```bash
# 把仓库中三个修复文件复制进运行中的容器并重启（容器删除后失效）：
docker cp qwen3_speaker_diarizer.py qwen3-asr:/usr/local/lib/python3.10/dist-packages/qwen_asr/inference/
docker cp serve.py             qwen3-asr:/usr/local/lib/python3.10/dist-packages/qwen_asr/cli/
docker cp middleware.py        qwen3-asr:/usr/local/lib/python3.10/dist-packages/qwen_asr/service/
docker restart qwen3-asr
```

> 正式部署请使用 hotfix 镜像；`docker cp` 的修改随容器删除而丢失，仅用于快速验证。

---

## 3. 部署前提（目标机要求）

| 项 | 要求 | 检查命令 |
|---|---|---|
| NVIDIA 驱动 | ≥ 535（CUDA 12.8 兼容） | `nvidia-smi` |
| GPU 数量确认 | 启动命令的 `--gpus device=N` 序号必须 < 实际卡数 | `nvidia-smi -L` |
| Docker | ≥ 20.10（仅 load/run，不构建） | `docker version` |
| nvidia-container-toolkit | 已安装并配置 | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` |
| 磁盘 | ≥ 40 GB（镜像 ~15GB + 模型 ~6.3GB + 运行余量） | `df -h` |
| 网络 | 无要求（离线运行） | — |

> **GPU 序号注意**：`--gpus '"device=0,1"'` 在单卡机上会直接报 `nvidia-container-cli: device error: 1: unknown device`（容器停留 Created 状态）。务必先 `nvidia-smi -L` 确认卡数再写序号。

---

## 4. 模型清单（必需的三个目录）

| 模型 | 体积 | 角色 | 容器内路径 |
|---|---|---|---|
| `Qwen3-ASR-1.7B` | 4.4 GB | ASR 主模型 | `/models/Qwen3-ASR-1.7B` |
| `Qwen3-ForcedAligner-0.6B` | 1.8 GB | 强制对齐模型 | `/models/Qwen3-ForcedAligner-0.6B` |
| `pyannote-speaker-diarization-community-1` | 33 MB | 说话人识别管线 | `/models/pyannote-speaker-diarization-community-1` |

**不需要的模型**：`pyannote-speaker-diarization-3.1`、`pyannote-segmentation-3.0`、`pyannote-wespeaker-voxceleb-resnet34-LM`、`MOSS-Transcribe-Diarize`。
原因：community-1 管线是**自包含**的——其 `config.yaml` 以相对路径（`$model/segmentation`、`$model/embedding`、`$model/plda`）引用子模型，三个子模型全部随目录自带，不引用任何外部 HF 仓库。

**部署前完整性检查**（community-1 缺任一子目录即加载失败）：

```bash
ls /data/models/pyannote-speaker-diarization-community-1
# 预期输出包含：config.yaml  segmentation/  embedding/  plda/

ls /data/models/Qwen3-ASR-1.7B
# 预期包含：config.json  *.safetensors  preprocessor_config.json  tokenizer_config.json ...

ls /data/models/Qwen3-ForcedAligner-0.6B
# 预期包含：config.json  model.safetensors  ...
```

---

## 5. 安装步骤

### 5.1 导入镜像

```bash
docker load < qwen3-asr-offline-cu128-hotfix.tar.gz
# 预期末尾：Loaded image: qwen3-asr-offline:cu128-hotfix

docker images | grep qwen3-asr-offline   # 确认存在
```

### 5.2 解包模型

```bash
mkdir -p /data/models
tar xzf qwen3-asr-models.tar.gz -C /data/models
# 解包后：
# /data/models/Qwen3-ASR-1.7B
# /data/models/Qwen3-ForcedAligner-0.6B
# /data/models/pyannote-speaker-diarization-community-1
```

### 5.3 冒烟验证（镜像内依赖完整性，可选但推荐）

```bash
docker run --rm qwen3-asr-offline:cu128-hotfix python3 -c "
import warnings; warnings.filterwarnings('ignore')
import vllm, pyannote.audio, transformers, torchcodec, torch
from qwen_asr.service.middleware import TranscriptionsMiddleware
print('smoke OK:', pyannote.audio.__version__, torchcodec.__version__)
"
# 预期输出：smoke OK: 4.0.7 0.13.x
```

---

## 6. 启动服务

### 6.1 标准启动（A10 24GB 单卡，实机验证通过）

```bash
docker run -d --name qwen3-asr --restart unless-stopped \
  --gpus '"device=0"' --shm-size 8g -p 8000:80 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v /data/models:/models:ro \
  qwen3-asr-offline:cu128-hotfix \
  qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner /models/Qwen3-ForcedAligner-0.6B \
    --diarizer /models/pyannote-speaker-diarization-community-1 \
    --gpu-memory-utilization 0.70 \
    --max-num-batched-tokens 8192
```

首次启动加载三个模型约需 **1~3 分钟**，用 `docker logs -f qwen3-asr` 观察，看到 vLLM `Uvicorn running on ...` 即就绪。

> **`--max-num-batched-tokens 8192` 必须加**：vLLM 默认值 2048 会导致长音频请求停滞卡死（详见 §11 #2），该参数是本次实机调试确认的关键修复。

### 6.2 Docker 参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--name` | `qwen3-asr` | 容器名，后续 docker exec/logs 均用它 |
| `--restart unless-stopped` | — | 异常退出自动重启（手动 stop 不重启） |
| `--gpus '"device=0"'` | 按需 | 显式指定 GPU 序号，**序号必须 < 实际卡数**（否则启动即报 unknown device，见 §11 #1）；多卡机上避免暴露全部 GPU |
| `--shm-size 8g` | — | vLLM 共享内存需求（NCCL / tensor 并行通信） |
| `-p 8000:80` | 按需 | 宿主机 8000 → 容器 80；多实例部署时改宿主端口 |
| `-v /data/models:/models:ro` | 按需 | 模型目录**只读挂载**，容器内统一从 `/models` 读取 |
| `-e HF_HUB_OFFLINE=1` | 必须 | 强制 huggingface_hub 离线（防任何组件静默联网） |
| `-e TRANSFORMERS_OFFLINE=1` | 必须 | transformers 侧离线 |
| `-e VLLM_NO_USAGE_STATS=1` | 必须 | 禁用 vLLM 遥测上报 |
| `-e DO_NOT_TRACK=1` | 必须 | 禁用其他工具链统计 |

> `--gpus` 引号写法：Linux bash 与 Windows PowerShell 均为 `--gpus '"device=0"'`（外双内单）。

### 6.3 服务启动参数（qwen-asr-serve 扩展参数）

以下参数由 `qwen-asr-serve` 解析剥离，**不会**透传给 vLLM：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--forced-aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | 对齐模型名/本地路径。**离线部署必传本地路径**；显式传空串（`--forced-aligner ""`）禁用对齐功能 |
| `--diarizer` | `pyannote/speaker-diarization-community-1` | 说话人识别管线名/本地路径。**离线部署必传本地路径**；显式空串禁用 |
| `--pyannote-token` | 无 | HF 访问令牌（联网加载门控模型用）。**离线本地路径加载不需要**，缺省依次取环境变量 `PYANNOTE_API_TOKEN` / `HF_TOKEN` |
| `--aligner-device` | `cuda:0` | 对齐模型设备；分卡部署时如 `cuda:1` |
| `--diarizer-device` | `cuda:0` | 说话人识别设备；分卡部署时如 `cuda:1` / `cuda:2` |
| `--max-concurrent-tasks` | `2` | segment 任务最大并发数（超过即进等待队列） |
| `--gpu-reserve-mb` | `1024` | 每设备显存安全余量（MB），准入控制用 |
| `--max-audio-seconds` | `3600.0` | 单条音频时长上限（秒），超限返回 400 |
| `--max-audio-bytes` | `524288000` | 音频体积上限（字节，默认 500MB） |
| `--segment-gap-threshold` | `0.8` | segment 切分时间间隙阈值（秒） |
| `--max-segment-seconds` | `30.0` | segment 最大段长（秒） |
| `--align-batch-size` | `4` | 对齐批大小（亦为标准模式 ASR 并发上限） |

**位置参数**：`qwen-asr-serve <model_path>` 为 ASR 主模型路径（必须是 argv 中首个非 flag 参数，或用 `--model` 显式指定）。

### 6.4 透传给 vLLM 的常用参数

未在上表中的参数原样转发 vLLM CLI，常用项：

| 参数 | 示例 | 说明 |
|---|---|---|
| `--served-model-name` | `qwen3-asr` | API 请求中 `model` 字段须匹配此名 |
| `--host` / `--port` | `0.0.0.0` / `80` | 监听地址（容器内统一 80，经 `-p` 映射） |
| `--gpu-memory-utilization` | `0.70` | vLLM 显存预分配比例。**单卡默认拓扑且未显式指定时，服务自动注入 0.70**（日志可见 WARNING 提示）；扩展模型在独立 GPU 时可用满默认 0.90 |
| `--max-num-batched-tokens` | `8192` | **必须 ≥ 8192**（详见下方专述） |
| `--max-model-len` | — | 序列长度上限（一般无需调整） |
| `--api-server-count` | 默认 1 | **扩展功能不支持 > 1**，启动即报错拒绝（多进程会导致扩展钩子静默失效） |

**`--max-num-batched-tokens` 专述（实机踩坑，务必阅读）**：

- 服务端把长音频切成 **180 秒/块** 送 ASR，单块 prompt 约 **4500 token**；
- vLLM 启用 chunked prefill 时默认 `max_num_batched_tokens=2048`，4500 > 2048 → 单块 prompt 被迫走"多模态拆分 prefill"路径；
- vLLM 0.14.0 该路径存在停滞缺陷：日志表现为 `Running: N reqs` + `0.0 tokens/s` 持续不动，GPU 利用率 0%，请求永不完成（短音频不受影响，因为一次 prefill 即完成）；
- **解法：启动时显式传 `--max-num-batched-tokens 8192`**，让单块一次 prefill 完成，彻底绕开拆分路径；
- 显存代价：无（KV cache 大小由 vLLM 按剩余显存自算，该参数只影响单步 prefill 的 token 预算）；A10 24GB 实测无压力；
- 生效确认：启动日志应出现 `Chunked prefill is enabled with max_num_batched_tokens=8192`。

### 6.5 显存布局参考（A10 24GB 单卡实测，gmu=0.70）

| 进程 | 显存占用 | 说明 |
|---|---|---|
| vLLM EngineCore | ~16.7 GB | 主模型权重 + KV cache 池 |
| 主进程（aligner + diarizer 常驻） | ~2.4 GB | 对齐/说话人模型权重 |
| **启动后空闲** | **~3.9 GB** | 调度器准入的判据 |

单任务准入阈值 ≈ 任务瞬态需求（默认参数约 1.8 GB）+ `--gpu-reserve-mb`（默认 1024）≈ **2.9 GB** < 3.9 GB 空闲，故默认双并发可正常运行；任务完成后空闲显存应回升（见 §7.2）。

### 6.6 不同 GPU 拓扑的启动示例

**A10 24GB 单卡（默认，见 §6.1）**

**T4 16GB 单卡：**

```bash
  ... qwen-asr-serve /models/Qwen3-ASR-1.7B ... \
    --gpu-memory-utilization 0.55 --max-num-batched-tokens 8192
# 单并发场景可用 0.60
```

**P4 8GB（仅短音频 ≤10min）：**

```bash
  ... qwen-asr-serve /models/Qwen3-ASR-1.7B ... \
    --gpu-memory-utilization 0.35 --gpu-reserve-mb 512 \
    --max-concurrent-tasks 1 --max-num-batched-tokens 8192
```

**两卡（vLLM 独占 GPU0 用满 0.90，扩展在 GPU1）：**

```bash
docker run -d --name qwen3-asr --restart unless-stopped \
  --gpus '"device=0,1"' --shm-size 8g -p 8000:80 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v /data/models:/models:ro \
  qwen3-asr-offline:cu128-hotfix \
  qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner /models/Qwen3-ForcedAligner-0.6B \
    --diarizer /models/pyannote-speaker-diarization-community-1 \
    --aligner-device cuda:1 --diarizer-device cuda:1 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192
```

**三卡（各自独占，显存互不竞争）：**

```bash
  --gpus '"device=0,1,2"' \
  ... --aligner-device cuda:1 --diarizer-device cuda:2 ...
```

### 6.7 关闭扩展功能（纯 vLLM 模式）

```bash
  ... qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner "" --diarizer ""
```

此时不加载扩展模型、不注入 gpu_memory_utilization、不安装钩子，行为与纯 vLLM 一致（vLLM 可用满默认 0.90 显存），仅提供标准转写。

---

## 7. 服务验证

### 7.1 日志检查

```bash
docker logs qwen3-asr 2>&1 | grep -iE "error|401|403|connectionerror" || echo "日志干净"
docker logs qwen3-asr 2>&1 | grep -iE "aligner|diarizer" | tail -5
# 预期：扩展模型加载成功、无任何网络错误（离线部署不应出现 ConnectionError）
docker logs qwen3-asr 2>&1 | grep -i "gpu-memory-utilization\|自动注入"
# 预期（单卡默认拓扑）：可见自动注入 0.70 的 WARNING 日志
docker logs qwen3-asr 2>&1 | grep "max_num_batched_tokens"
# 预期：Chunked prefill is enabled with max_num_batched_tokens=8192
```

> 日志中的 `Repo id must be in the form 'repo_name'...` ERROR 行是 vLLM 对本地路径加载的正常回退日志，**无害可忽略**。

### 7.2 健康检查

```bash
curl http://localhost:8000/health/detail
```

响应示例：

```json
{
  "status": "ok",
  "extensionModelsLoaded": true,
  "runningTasks": 0,
  "queuedTasks": 0,
  "maxConcurrentTasks": 2,
  "gpuReserveMb": 1024,
  "devices": [
    {"device": "cuda:0", "role": "vllm+aligner+diarizer", "freeVramMb": 3890, "totalVramMb": 23028}
  ]
}
```

> - `freeVramMb` 是 vLLM 预分配后的剩余值，**偏低是正常态**（vLLM 启动即按 gmu 比例占住显存），不代表异常。
> - **fix3 验证要点**：segment 任务完成后 `freeVramMb` 应**回升**到 ~3.8GB 左右；若长期停在 ~2.4GB 且 `queuedTasks` 持续 > 0，说明镜像是旧的（未含 fix3），按 §2 处理。

### 7.3 端到端转写验证

**方式一：宿主机 curl（宿主机有 curl 即可）**

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "model=qwen3-asr" \
  -F "file=@test.wav" \
  -F "timestamp_granularities[]=segment" \
  -F "response_format=json"
```

**方式二：容器内一行式请求（目标机宿主机无 Python 环境时，实测可用）**

```bash
# 生成 10 秒测试音频（440Hz 正弦波，返回 text 为空属预期——非语音）
docker exec qwen3-asr python3 -c "
import numpy as np, soundfile as sf
sr = 16000
t = np.arange(int(sr*10)) / sr
sf.write('/tmp/t10s.wav', (0.1*np.sin(2*np.pi*440*t)).astype(np.float32), sr)
"

docker exec qwen3-asr python3 -c "
import requests
with open('/tmp/t10s.wav','rb') as f:
    r = requests.post('http://127.0.0.1:80/v1/audio/transcriptions',
        files={'file': ('t.wav', f)}, data={'model': 'qwen3-asr'}, timeout=120)
print(r.status_code, r.text[:300])
"
# 预期：200 {"text": ""}（正弦波非语音，空文本属正常；重点是 200 且秒回）
```

**方式三：容器内执行示例脚本**

```bash
docker cp test.wav qwen3-asr:/tmp/test.wav
docker exec qwen3-asr python3 /data/shared/Qwen3-ASR/examples/example_segment_api.py \
  --file /tmp/test.wav --base-url http://localhost:80
```

> 注意容器内端口是 **80**（不是宿主机映射的 8000）。

---

## 8. API 接口文档

### 8.1 端点

`POST /v1/audio/transcriptions`（multipart/form-data，OpenAI 兼容扩展）

### 8.2 请求参数

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `model` | string | 是 | 模型名，须匹配 `--served-model-name`（默认 `qwen3-asr`） |
| `file` | file | 二选一 | 音频文件（multipart 文件部分） |
| `audio_url` | string | 二选一 | 音频 HTTPS URL（大文件推荐；服务端流式下载，受 SSRF 校验：仅 https、拒绝内网/环回/链路本地地址） |
| `language` | string | 否 | 语言码（`zh`/`en`）或语言名（`Chinese`）；缺省自动检测 |
| `prompt` | string | 否 | 上下文提示（映射到 ASR context，如专有名词、领域词汇） |
| `timestamp_granularities[]` | string[] | 否 | 含 `segment` 时启用**扩展管线**（segment 时间戳 + 说话人识别）；`word` 粒度 v1 不支持（返回 400） |
| `response_format` | string | 否 | `json`（默认）/ `text` / `verbose_json`；**segment 模式固定 json**，配 text/verbose_json 返回 400 |
| `min_speakers` | int | 否 | 说话人数下限（透传 pyannote，仅 segment 模式有效） |
| `max_speakers` | int | 否 | 说话人数上限（透传 pyannote，仅 segment 模式有效） |

**两种模式**：

| 模式 | 触发条件 | 响应 |
|---|---|---|
| **segment 扩展模式** | `timestamp_granularities[]=segment` + `response_format=json` | 自定义结构（见 §8.3），含时间戳 + 说话人 |
| **标准 OpenAI 模式** | 不带粒度参数 | OpenAI 标准响应（`{"text": ...}`，无 speaker 字段） |

> 注意：OpenAI 规范中 `segment` 粒度配 `verbose_json` 返回无说话人的标准 segments；本服务将 `segment` 粒度定义为"segment + 说话人扩展"语义，属产品决策，与 OpenAI 客户端的默认行为存在差异。

支持的语言（30 种）：Chinese、English、Cantonese、Arabic、German、French、Spanish、Portuguese、Indonesian、Italian、Korean、Russian、Thai、Vietnamese、Japanese、Turkish、Hindi、Malay、Dutch、Swedish、Danish、Finnish、Polish、Czech、Filipino、Persian、Greek、Romanian、Hungarian、Macedonian。

### 8.3 segment 模式响应格式

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

| 字段 | 说明 |
|---|---|
| `language` | 检测/指定的语言码 |
| `duration` | 音频总时长（秒） |
| `text` | 完整转写文本 |
| `processTime` | 服务端处理耗时（秒） |
| `segments[].start/end` | 段级时间戳（秒，强制对齐产出，按 start 升序） |
| `segments[].speaker` | 该段**主导说话人**（一段内多人时取说话时间占比最多者；无法判定时为 `null`） |
| `segments[].speakers` | 该段出现过的全部说话人列表 |
| `speakerSummary.speakerCount` | 识别出的说话人总数 |
| `speakerSummary.speakers[].totalDuration` | 该说话人累计发言时长（秒）；`speaker=null` 的段时长不归属任何人，ΣtotalDuration 可能 < duration |
| `speakerSummary.speakers[].segmentCount` | 该说话人的主导段数 |

### 8.4 错误响应

OpenAI 风格 `{"error": {"message": ..., "type": "invalid_request_error", ...}}`，常见 400 场景：

- `file 与 audio_url 必须提供且只能提供其中之一`
- `segment 时间戳粒度与 response_format=text 不兼容`（segment 模式固定 json）
- `timestamp_granularities 含 word（v1 不支持）`
- `音频时长 XXXs 超过上限 3600.0s`
- `audio_url 仅支持 https 协议` / `解析到内网...已拒绝`（SSRF 防护）

### 8.5 长音频同步响应说明

1 小时音频端到端处理（含排队）可达数分钟，**接口是同步阻塞的**：

- 客户端超时须设足够大（示例脚本默认 900s）；
- 中间若有反向代理，必须按 §10 调整 `client_max_body_size` / `proxy_read_timeout`；
- 客户端断连后服务端会尽力取消排队/处理中的任务（释放显存许可）。

---

## 9. 运维操作

| 操作 | 命令 |
|---|---|
| 查看日志 | `docker logs -f qwen3-asr` |
| 查看最近 N 行日志 | `docker logs -f --tail=200 qwen3-asr`（**`--tail` 是双横线**，`-tail` 会报 `unknown shorthand flag: 'a'`） |
| 停止 | `docker stop qwen3-asr` |
| 启动（已创建） | `docker start qwen3-asr` |
| 重启 | `docker restart qwen3-asr` |
| 删除容器 | `docker rm -f qwen3-asr` |
| 更新代码（新镜像） | 构建机重 build → `docker load` 新 tar → `docker rm -f` 旧容器 → 重新 `docker run`（模型目录不动） |
| 更新模型 | 替换 `/data/models` 下对应目录 → `docker restart qwen3-asr`（镜像不动） |
| 回滚 | 保留旧 tar 包 → `docker load` 旧镜像 → 重建容器 |

**长音频处理中判断是否卡死**（CPU/GPU 都空闲但日志 0 tok/s 时）：

```bash
docker stats --no-stream qwen3-asr        # CPU% 接近 0 = 非计算瓶颈
docker exec qwen3-asr top -H -b -n 1 | head -20
nvidia-smi                                 # GPU-Util 0% = 非计算瓶颈
```

若以上均空闲且引擎日志持续 `Running: N reqs` + `0.0 tokens/s` 超过 2 分钟 → 大概率命中 §11 #2，检查是否漏了 `--max-num-batched-tokens 8192`。

---

## 10. 反向代理配置（前置 nginx 时必改）

```nginx
client_max_body_size 500m;   # 1h wav 约 230MB 上传，默认 1MB 会 413
proxy_read_timeout 900s;     # 长音频端到端耗时（含排队等待）
proxy_send_timeout 900s;
```

---

## 11. 故障排查（实机调试全记录，按严重度排列）

### #1 启动即失败：`nvidia-container-cli: device error: 1: unknown device`

- **现象**：`docker run` 返回容器 ID 但容器停留 `Created` 状态，`docker logs` 为空；
- **原因**：`--gpus '"device=0,1"'` 但机器只有 1 块 GPU——序号 1 不存在；
- **解法**：`nvidia-smi -L` 确认实际卡数，单卡机改 `--gpus '"device=0"'`。

### #2 长音频请求卡死：`Running: N reqs` + `0.0 tokens/s` 持续十几分钟（P0）

- **现象**：上传十几分钟以上的 mp3 后日志吞吐归零、GPU-Util 0%、CPU 空闲，请求永不返回；**短音频正常**（10s 测试请求秒回 200）；
- **原因**：vLLM 默认 `max_num_batched_tokens=2048`，而服务端 180s 音频分块 ≈ 4500 token > 2048，触发多模态拆分 prefill 路径，vLLM 0.14.0 该路径停滞；
- **判别**：容器内发 10s 小请求（§7.3 方式二），正常返回而长音频卡死即为此问题；
- **解法**：启动命令加 `--max-num-batched-tokens 8192`（§6.1 已含）。

### #3 首个请求成功、后续所有请求永久排队（P0，hotfix fix3 已修复）

- **现象**：第一个 segment 请求 200 OK 正常返回；第二个请求起永久挂起，traceback 停在 `scheduler.py ... await ticket.future`；`/health/detail` 显示 `freeVramMb` 从 ~3.9GB 掉到 ~2.4GB 不回升，`queuedTasks` 持续 ≥ 1；
- **原因**：对齐/说话人前向结束后，PyTorch 缓存分配器把空闲显存块留在自己手里不归还驱动 → 调度器用 `mem_get_info` 实测空闲显存做准入判断，误判"显存不足"（阈值 ≈ 2.9GB > 2.4GB）→ 后续任务全部滞留队列。**假死锁**：缓存块实际可复用，并非真占满；
- **解法**：使用 `cu128-hotfix` 镜像（fix3：任务收尾、调度许可释放前 `torch.cuda.empty_cache()` 归还空闲块）；
- **应急绕过**（旧镜像不改代码）：启动加 `--gpu-reserve-mb 256` 降低准入阈值——但余量被压缩后大任务有 OOM 风险，仅限临时；
- **验证**：任务完成后 `/health/detail` 的 `freeVramMb` 应回升 ~3.8GB。

### #4 启动报 `TypeError: 'device' must be an instance of 'torch.device', got 'str'`（hotfix fix1 已修复）

- **原因**：pyannote.audio 4.x 的 `Pipeline.to()` 严格要求 `torch.device` 实例（3.x 两者皆可）；
- **解法**：使用 hotfix 镜像（统一转换 `torch.device(device)`）。

### #5 启动报 `RuntimeError: Cannot add middleware after an application has started`（hotfix fix2 已修复）

- **原因**：vLLM 0.14.0 `build_app` 末尾的 SageMaker bootstrap 预建中间件栈，后续 `add_middleware` 误判服务已启动；
- **解法**：使用 hotfix 镜像（手动插入 `user_middleware` 并重建栈）。

### #6 其他已知问题速查

| 症状 | 原因 | 解法 |
|---|---|---|
| 启动报 diarizer 加载失败 / 找不到子模型 | 模型目录缺 `segmentation/` / `embedding/` / `plda/` 子目录 | §4 完整性检查，重新解包模型 tar |
| 启动报 "无法从 vLLM init_app_state 捕获 engine_client" | 镜像内代码版本旧 | 用包含 `603c8f4` 之后提交的镜像重 build |
| 启动报 `--api-server-count > 1` 不支持 | 多 API server 进程无法继承扩展钩子 | 移除该参数；多实例用负载均衡部署多个单 API server |
| segment 请求一直排队不放行 | 显存准入未通过（空闲显存 < 任务需求 + reserve） | 查 `/health/detail` 的 `freeVramMb`；调低 `--gpu-memory-utilization`、`--max-concurrent-tasks` 或分卡部署 |
| 启动即 OOM | gmu 设置过高，扩展模型无显存可用 | 单卡默认拓扑让服务自动注入 0.70；手动指定时参考 §6.6 拓扑表 |
| 上传 1h 音频 413 / 中途断开 | 前置代理 body/timeout 限制 | §10 nginx 参数 |
| 转写示例连接被拒 | base-url 端口不对 | 容器内是 80，宿主机经 `-p` 映射的是 8000 |
| `import torchcodec` warning（libavutil.so.XX cannot open） | torchcodec 与系统 FFmpeg 版本配对提示 | **可忽略**：pyannote 走内存 waveform，不依赖其解码器 |
| 日志 ERROR `Repo id must be in the form 'repo_name'` | vLLM 对本地路径加载的正常回退日志 | 无害，忽略 |
| tokenizer 加载 warning（incorrect regex pattern） | 上游 tokenizer 配置提示 | 无害，忽略 |
| `docker logs -f -tail=1000` 报 `unknown shorthand flag: 'a'` | `-tail` 应为 `--tail` | `docker logs -f --tail=1000 qwen3-asr` |
| 日志出现 ConnectionError / 401 | 有组件试图联网（离线环境不通） | 确认 `-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1` 已传；确认三个模型参数全部传的本地路径 |

---

## 12. 已知限制

- **vLLM 私有 API 耦合**：engine_client 注入钩子依赖 vLLM 0.14.0 的 `init_app_state` 结构，升级 vLLM 必须重新验证。
- **`--max-num-batched-tokens` 下限约束**：必须 ≥ 8192（覆盖 180s 分块的 ~4500 token 单次 prefill），低于该值长音频会触发 §11 #2 的停滞；若未来调大音频分块时长，需同步上调此参数。
- **同步长响应**：1h 音频高并发时队尾任务等待较久，中间代理超时会切断连接（服务端有断连取消，不白算但客户端拿不到结果）。
- **`speaker=null` 语义**：无法判定主导说话人的段，其时长不计入任何 speaker 的 totalDuration。
- **word 粒度未支持**：`timestamp_granularities[]=word` 返回 400。
