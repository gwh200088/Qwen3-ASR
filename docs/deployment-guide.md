# Qwen3-ASR segment 时间戳 + 说话人识别服务 · 部署文档

> 适用镜像：`qwen3-asr-offline:cu128`（基于 `dev` 分支构建，`BUNDLE_FLASH_ATTENTION=false`）
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

## 2. 部署前提（目标机要求）

| 项 | 要求 | 检查命令 |
|---|---|---|
| NVIDIA 驱动 | ≥ 535（CUDA 12.8 兼容） | `nvidia-smi` |
| Docker | ≥ 20.10（仅 load/run，不构建） | `docker version` |
| nvidia-container-toolkit | 已安装并配置 | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` |
| 磁盘 | ≥ 40 GB（镜像 ~15GB + 模型 ~6.3GB + 运行余量） | `df -h` |
| 网络 | 无要求（离线运行） | — |

---

## 3. 模型清单（必需的三个目录）

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

## 4. 安装步骤

### 4.1 导入镜像

```bash
docker load < qwen3-asr-offline-cu128.tar.gz
# 预期末尾：Loaded image: qwen3-asr-offline:cu128

docker images | grep qwen3-asr-offline   # 确认存在
```

### 4.2 解包模型

```bash
mkdir -p /data/models
tar xzf qwen3-asr-models.tar.gz -C /data/models
# 解包后：
# /data/models/Qwen3-ASR-1.7B
# /data/models/Qwen3-ForcedAligner-0.6B
# /data/models/pyannote-speaker-diarization-community-1
```

### 4.3 冒烟验证（镜像内依赖完整性，可选但推荐）

```bash
docker run --rm qwen3-asr-offline:cu128 python3 -c "
import warnings; warnings.filterwarnings('ignore')
import vllm, pyannote.audio, transformers, torchcodec, torch
from qwen_asr.service.middleware import TranscriptionsMiddleware
print('smoke OK:', pyannote.audio.__version__, torchcodec.__version__)
"
# 预期输出：smoke OK: 4.0.7 0.13.x
```

---

## 5. 启动服务

### 5.1 标准启动（A10 24GB 单卡，推荐）

```bash
docker run -d --name qwen3-asr --restart unless-stopped \
  --gpus '"device=0"' --shm-size 8g -p 8000:80 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v /data/models:/models:ro \
  qwen3-asr-offline:cu128 \
  qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner /models/Qwen3-ForcedAligner-0.6B \
    --diarizer /models/pyannote-speaker-diarization-community-1
```

首次启动加载三个模型约需 **1~3 分钟**，用 `docker logs -f qwen3-asr` 观察，看到 vLLM `Uvicorn running on ...` 即就绪。

### 5.2 Docker 参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--name` | `qwen3-asr` | 容器名，后续 docker exec/logs 均用它 |
| `--restart unless-stopped` | — | 异常退出自动重启（手动 stop 不重启） |
| `--gpus '"device=0"'` | 按需 | 显式指定 GPU 序号，多卡机上避免暴露全部 GPU（注意 PowerShell/Linux 引号写法，见 §5.5） |
| `--shm-size 8g` | — | vLLM 共享内存需求（NCCL / tensor 并行通信） |
| `-p 8000:80` | 按需 | 宿主机 8000 → 容器 80；多实例部署时改宿主端口 |
| `-v /data/models:/models:ro` | 按需 | 模型目录**只读挂载**，容器内统一从 `/models` 读取 |
| `-e HF_HUB_OFFLINE=1` | 必须 | 强制 huggingface_hub 离线（防任何组件静默联网） |
| `-e TRANSFORMERS_OFFLINE=1` | 必须 | transformers 侧离线 |
| `-e VLLM_NO_USAGE_STATS=1` | 必须 | 禁用 vLLM 遥测上报 |
| `-e DO_NOT_TRACK=1` | 必须 | 禁用其他工具链统计 |

### 5.3 服务启动参数（qwen-asr-serve 扩展参数）

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

### 5.4 透传给 vLLM 的常用参数

未在上表中的参数原样转发 vLLM CLI，常用项：

| 参数 | 示例 | 说明 |
|---|---|---|
| `--served-model-name` | `qwen3-asr` | API 请求中 `model` 字段须匹配此名 |
| `--host` / `--port` | `0.0.0.0` / `80` | 监听地址（容器内统一 80，经 `-p` 映射） |
| `--gpu-memory-utilization` | `0.70` | vLLM 显存预分配比例。**单卡默认拓扑且未显式指定时，服务自动注入 0.70**（日志可见 WARNING 提示）；扩展模型在独立 GPU 时可用满默认 0.90 |
| `--max-model-len` | — | 序列长度上限（一般无需调整） |
| `--api-server-count` | 默认 1 | **扩展功能不支持 > 1**，启动即报错拒绝（多进程会导致扩展钩子静默失效） |

### 5.5 不同 GPU 拓扑的启动示例

**A10 24GB 单卡（默认，自动注入 gmu=0.70，双并发 1h 音频）：**
见 §5.1。

**T4 16GB 单卡：**

```bash
  ... qwen-asr-serve /models/Qwen3-ASR-1.7B ... \
    --gpu-memory-utilization 0.55
# 单并发场景可用 0.60
```

**P4 8GB（仅短音频 ≤10min）：**

```bash
  ... qwen-asr-serve /models/Qwen3-ASR-1.7B ... \
    --gpu-memory-utilization 0.35 --gpu-reserve-mb 512 --max-concurrent-tasks 1
```

**两卡（vLLM 独占 GPU0 用满 0.90，扩展在 GPU1）：**

```bash
docker run -d --name qwen3-asr --restart unless-stopped \
  --gpus '"device=0,1"' --shm-size 8g -p 8000:80 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v /data/models:/models:ro \
  qwen3-asr-offline:cu128 \
  qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner /models/Qwen3-ForcedAligner-0.6B \
    --diarizer /models/pyannote-speaker-diarization-community-1 \
    --aligner-device cuda:1 --diarizer-device cuda:1 \
    --gpu-memory-utilization 0.90
```

**三卡（各自独占，显存互不竞争）：**

```bash
  --gpus '"device=0,1,2"' \
  ... --aligner-device cuda:1 --diarizer-device cuda:2 ...
```

> `--gpus` 引号说明：Linux bash 用 `'"device=0,1"'`；Windows PowerShell 用 `--gpus '"device=0,1"'`（外双内单）；若部署机只有一块 GPU 也可直接 `--gpus all`。

### 5.6 关闭扩展功能（纯 vLLM 模式）

```bash
  ... qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner "" --diarizer ""
```

此时不加载扩展模型、不注入 gpu_memory_utilization、不安装钩子，行为与纯 vLLM 一致（vLLM 可用满默认 0.90 显存），仅提供标准转写。

---

## 6. 服务验证

### 6.1 日志检查

```bash
docker logs qwen3-asr 2>&1 | grep -iE "error|401|403|connectionerror" || echo "日志干净"
docker logs qwen3-asr 2>&1 | grep -iE "aligner|diarizer" | tail -5
# 预期：扩展模型加载成功、无任何网络错误（离线部署不应出现 ConnectionError）
docker logs qwen3-asr 2>&1 | grep -i "gpu-memory-utilization\|自动注入"
# 预期（单卡默认拓扑）：可见自动注入 0.70 的 WARNING 日志
```

### 6.2 健康检查

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
    {"device": "cuda:0", "role": "vllm+aligner+diarizer", "freeVramMb": 5230, "totalVramMb": 23028}
  ]
}
```

> `freeVramMb` 是 vLLM 预分配后的剩余值，**偏低是正常态**（vLLM 启动即按 gmu 比例占住显存），不代表异常。

### 6.3 端到端转写验证

**方式一：宿主机 curl（宿主机有 curl 即可）**

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "model=qwen3-asr" \
  -F "file=@test.wav" \
  -F "timestamp_granularities[]=segment" \
  -F "response_format=json"
```

**方式二：容器内执行示例脚本（目标机宿主机无 Python 环境时）**

```bash
docker cp test.wav qwen3-asr:/tmp/test.wav
docker exec qwen3-asr python3 /data/shared/Qwen3-ASR/examples/example_segment_api.py \
  --file /tmp/test.wav --base-url http://localhost:80
```

> 注意容器内端口是 **80**（不是宿主机映射的 8000）。

---

## 7. API 接口文档

### 7.1 端点

`POST /v1/audio/transcriptions`（multipart/form-data，OpenAI 兼容扩展）

### 7.2 请求参数

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
| **segment 扩展模式** | `timestamp_granularities[]=segment` + `response_format=json` | 自定义结构（见 §7.3），含时间戳 + 说话人 |
| **标准 OpenAI 模式** | 不带粒度参数 | OpenAI 标准响应（`{"text": ...}`，无 speaker 字段） |

> 注意：OpenAI 规范中 `segment` 粒度配 `verbose_json` 返回无说话人的标准 segments；本服务将 `segment` 粒度定义为"segment + 说话人扩展"语义，属产品决策，与 OpenAI 客户端的默认行为存在差异。

支持的语言（30 种）：Chinese、English、Cantonese、Arabic、German、French、Spanish、Portuguese、Indonesian、Italian、Korean、Russian、Thai、Vietnamese、Japanese、Turkish、Hindi、Malay、Dutch、Swedish、Danish、Finnish、Polish、Czech、Filipino、Persian、Greek、Romanian、Hungarian、Macedonian。

### 7.3 segment 模式响应格式

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

### 7.4 错误响应

OpenAI 风格 `{"error": {"message": ..., "type": "invalid_request_error", ...}}`，常见 400 场景：

- `file 与 audio_url 必须提供且只能提供其中之一`
- `segment 时间戳粒度与 response_format=text 不兼容`（segment 模式固定 json）
- `timestamp_granularities 含 word（v1 不支持）`
- `音频时长 XXXs 超过上限 3600.0s`
- `audio_url 仅支持 https 协议` / `解析到内网...已拒绝`（SSRF 防护）

### 7.5 长音频同步响应说明

1 小时音频端到端处理（含排队）可达数分钟，**接口是同步阻塞的**：

- 客户端超时须设足够大（示例脚本默认 900s）；
- 中间若有反向代理，必须按 §9 调整 `client_max_body_size` / `proxy_read_timeout`；
- 客户端断连后服务端会尽力取消排队/处理中的任务（释放显存许可）。

---

## 8. 运维操作

| 操作 | 命令 |
|---|---|
| 查看日志 | `docker logs -f qwen3-asr` |
| 停止 | `docker stop qwen3-asr` |
| 启动（已创建） | `docker start qwen3-asr` |
| 重启 | `docker restart qwen3-asr` |
| 删除容器 | `docker rm -f qwen3-asr` |
| 更新代码（新镜像） | 构建机重 build → `docker load` 新 tar → `docker rm -f` 旧容器 → 重新 `docker run`（模型目录不动） |
| 更新模型 | 替换 `/data/models` 下对应目录 → `docker restart qwen3-asr`（镜像不动） |
| 回滚 | 保留旧 tar 包 → `docker load` 旧镜像 → 重建容器 |

---

## 9. 反向代理配置（前置 nginx 时必改）

```nginx
client_max_body_size 500m;   # 1h wav 约 230MB 上传，默认 1MB 会 413
proxy_read_timeout 900s;     # 长音频端到端耗时（含排队等待）
proxy_send_timeout 900s;
```

---

## 10. 故障排查

| 症状 | 原因 | 解法 |
|---|---|---|
| 启动报 diarizer 加载失败 / 找不到子模型 | 模型目录缺 `segmentation/` / `embedding/` / `plda/` 子目录 | §3 完整性检查，重新解包模型 tar |
| 启动报 "无法从 vLLM init_app_state 捕获 engine_client" | 镜像内代码版本旧 | 用包含 `603c8f4` 之后提交的镜像重 build |
| 启动报 `--api-server-count > 1` 不支持 | 多 API server 进程无法继承扩展钩子 | 移除该参数；多实例用负载均衡部署多个单 API server |
| segment 请求一直排队不放行 | 显存准入未通过（空闲显存 < 任务需求 + reserve） | 查 `/health/detail` 的 `freeVramMb`；调低 `--gpu-memory-utilization`、`--max-concurrent-tasks` 或分卡部署 |
| 启动即 OOM | gmu 设置过高，扩展模型无显存可用 | 单卡默认拓扑让服务自动注入 0.70；手动指定时参考 §5.5 拓扑表 |
| 上传 1h 音频 413 / 中途断开 | 前置代理 body/timeout 限制 | §9 nginx 参数 |
| 转写示例连接被拒 | base-url 端口不对 | 容器内是 80，宿主机经 `-p` 映射的是 8000 |
| `import torchcodec` warning（libnvrtc/FFmpeg） | torchcodec 二进制配对提示 | 功能不受影响（pyannote 走内存 waveform，不依赖其解码器），可忽略 |
| 日志出现 ConnectionError / 401 | 有组件试图联网（离线环境不通） | 确认 `-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1` 已传；确认三个模型参数全部传的本地路径 |

---

## 11. 已知限制

- **vLLM 私有 API 耦合**：engine_client 注入钩子依赖 vLLM 0.14.0 的 `init_app_state` 结构，升级 vLLM 必须重新验证。
- **同步长响应**：1h 音频高并发时队尾任务等待较久，中间代理超时会切断连接（服务端有断连取消，不白算但客户端拿不到结果）。
- **`speaker=null` 语义**：无法判定主导说话人的段，其时长不计入任何 speaker 的 totalDuration。
- **word 粒度未支持**：`timestamp_granularities[]=word` 返回 400。
