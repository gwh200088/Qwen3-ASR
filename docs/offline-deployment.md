# Qwen3-ASR segment + 说话人识别服务 · 完全离线部署方案

> 适用分支：`dev`（`f73ed66` 功能实现 + `03b8349` 离线构建改造 + `4a7f92b` torchcodec 修正 + `603c8f4` CMake 条件化）
> 目标：目标机**零网络**运行，服务全功能可用（vLLM ASR + 强制对齐 + pyannote 说话人识别）。
>
> **模型来源**：`D:\workplace\TMRI\AI\body_Camera\model\models`（已存在，无需任何下载）。

> ⚠️ **本文档定位**：记录**首次从零构建基础镜像**（`qwen3-asr-offline:cu128`）的完整流程与踩坑实证，
> 适用于需要重建基座或排查基础镜像问题的场景。
>
> **日常部署与升级请优先看 `docs/deployment-guide.md`** —— 当前生产镜像为
> `qwen3-asr-offline:cu128-align-fallback`（在 `cu128-punct` 之上全量覆盖 `qwen_asr/`，
> 采用"只传源码、目标机轻量构建"的叠加方式，见该文档 §2.5 与 §9）。
> 基础镜像已存在时**不需要**重跑本文档的 §3.1~§3.3。

---

## 0. 本地模型盘点与实测结论（本会话容器内验证）

| 模型 | 目录 | 状态 | 离线方案中的角色 |
|---|---|---|---|
| Qwen3-ASR-1.7B | `models/Qwen3-ASR-1.7B`（4.4 GB） | ✅ config 实测可读（`model_type=qwen3_asr`） | ASR 主模型，本地路径直载 |
| Qwen3-ForcedAligner-0.6B | `models/Qwen3-ForcedAligner-0.6B`（1.8 GB） | ✅ config 实测可读（装本仓库包后） | 对齐模型，本地路径直载 |
| pyannote-speaker-diarization-community-1 | `models/pyannote-speaker-diarization-community-1`（33 MB） | ✅ **管线完整加载实测通过** | 说话人识别管线，本地路径直载 |
| pyannote-segmentation-3.0 / wespeaker-voxceleb / diarization-3.1 | `models/` 下同名目录 | 不需要 | 仅旧 3.1 管线使用，community-1 不引用，**无需打包** |
| MOSS-Transcribe-Diarize | `models/MOSS-Transcribe-Diarize` | 不需要 | 与本项目无关，无需打包 |

**关键实测**（pyannote 4.x 容器内，挂载本地目录）：

```
Pipeline.from_pretrained("<本地目录>/pyannote-speaker-diarization-community-1")
→ PIPELINE OK: SpeakerDiarization（segmentation/embedding 组件真实加载，零网络调用）
```

原因：community-1 的 `config.yaml` 以 `$model/segmentation`、`$model/embedding`、`$model/plda` **相对路径**引用子模型——子模型（`segmentation/`、`embedding/`、`plda/`）全部随仓库目录自带，**不引用任何外部 HF repo**。因此：

- ❌ 不需要 HF 缓存格式（`models--org--name/snapshots/...`）
- ❌ 不需要 HF_TOKEN、不需要接受门控条款、不需要 hf-mirror
- ❌ 不需要"真实加载管线拉全子模型"的预下载步骤
- ✅ 三个模型目录原样打包 → 目标机挂载 → 传**本地路径**启动，即完即用

---

## 1. 总体思路

### 1.1 产物两件套（联网构建机准备）

| 产物 | 内容 | 体积 |
|---|---|---|
| `qwen3-asr-offline-cu128.tar.gz` | Docker 镜像（功能代码 + 全部 Python 依赖 + ffmpeg） | ~15 GB |
| `qwen3-asr-models.tar.gz` | 三个模型目录（ASR 4.4G + aligner 1.8G + community-1 33M） | ~6.3 GB |

（模型已存在本地，直接 tar 打包即可；原"HF 缓存预下载"流程整体作废。）

### 1.2 离线运行原理

```bash
HF_HUB_OFFLINE=1        # 保险：即使某组件试图访问 HF 也直接走本地/报错，绝不静默联网
TRANSFORMERS_OFFLINE=1  # 同上（transformers 侧）
VLLM_NO_USAGE_STATS=1   # 禁用 vLLM 遥测上报
DO_NOT_TRACK=1          # 禁用其他工具链统计
```

- 三个模型全部传**本地目录路径**（挂载进容器），加载链路不经过 huggingface_hub 的 repo 解析；
- 实测依据：pyannote `Pipeline.from_pretrained` 对目录走 `isdir` 本地分支；transformers/vLLM 对本地路径是原生支持；
- `HF_HUB_OFFLINE=1` 保留作为防触网双保险（pyannote 4.x 源码级核查：全部加载经 `hf_hub_download`，尊重该变量）。

### 1.3 构建机要求（注意：与目标机要求不同）

| 项 | 要求 | 原因 |
|---|---|---|
| Docker | **≥ 23.0** | Dockerfile 使用 heredoc（`RUN <<EOF`）与 `--mount=type=cache`，需 BuildKit 内置 dockerfile 前端 ≥ 1.4；Docker 20.10 构建会在 heredoc 处直接解析失败。仅 `docker load/run` 的目标机 20.10 即可 |
| 网络 | 见下表 | 构建期拉取基础镜像、apt 包、pip 包 |

构建期网络直连点：

| 直连点 | 何时需要 | 国内不通时的对策 |
|---|---|---|
| `docker.io` | 拉取 `nvidia/cuda:12.8.0-devel-ubuntu22.04` | mirror 前缀拉取后 retag（§3.2）；备选 `docker.m.daocloud.io` / `docker.1ms.run` / `dockerproxy.net`，或 `--build-arg HTTP(S)_PROXY` 走代理（apt/wget/pip 均尊重） |
| `archive.ubuntu.com` | apt 安装系统包 | 同上走代理；或构建前临时改 sources.list 为国内源（自行斟酌） |
| PyPI | pip 安装全部 Python 依赖 | `--build-arg HTTP(S)_PROXY`；注意 Dockerfile 未内置 `PIP_INDEX_URL`，如需固定镜像源需自行改造 |
| `github.com` | **仅** `BUNDLE_FLASH_ATTENTION=true` 时（CMake + flash-attn 源码） | **离线部署推荐 `false`，完全不触 github**（CMake 下载已条件化） |
| ~~`hf.co`~~ | ~~模型下载~~ | **不再需要**（模型全部来自本地目录） |

### 1.4 目标机要求（OS 层，与镜像无关）

- NVIDIA 驱动（`nvidia-smi` 可用，建议 ≥ 535）
- Docker ≥ 20.10（仅 load/run，不构建）+ `nvidia-container-toolkit`

---

## 2. 关键坑位与实证状态

**实证分级说明**：区分"已实测 / 已修复（代码级）/ 待构建机执行"。本机（无 GPU、Docker Hub 被墙）能做的验证已全部做完，其余以强制预演闸门覆盖。

| # | 坑 | 措施 | 状态 |
|---|---|---|---|
| 1 | PyPI 的 `qwen-asr` 不含 segment/说话人功能（未发布） | Dockerfile 改为 `COPY .` 本地源码安装 | ✅ 已提交 `03b8349` |
| 2 | ~~pyannote 子模型需联网拉全~~ → **不适用**：community-1 子模型随仓库自带（`$model/` 相对路径），本地目录自包含 | 本地目录直载 | ✅ **已实测**（管线完整加载，零网络） |
| 3 | torchcodec ≥0.14 按 CUDA 13 构建（链接 `libnvrtc.so.13`），CUDA 12.8 镜像内 import 即炸 | Dockerfile pin `torchcodec==0.13.*`（torch 2.9 官方配对） | ✅ 已提交 `4a7f92b`；镜像内复验见 §3.3 |
| 4 | ~~pyannote 门控模型 401/403~~ → **不再需要 token**：本地目录加载不鉴权 | — | ✅ 已随本地目录方案消除 |
| 5 | 国内直连 Docker Hub 超时 | mirror 前缀拉取后 retag（§3.2） | ⚠️ 本会话实测 mirror 可用性波动（daocloud 长时间卡死、1ms 慢），建议多备选 + 代理兜底 |
| 6 | HF_TOKEN 打进镜像层有泄露风险 | ~~模型外挂缓存卷~~ → 模型外挂目录，全程无 token | ✅ 已消除 |
| 7 | pyannote 离线加载路径是否含在线校验调用 | 源码级核查通过（见下） | ✅ 已实测（源码级）；运行时由 §3.5 预演最终把关 |
| 8 | 离线完整启动（三模型加载 + 服务就绪）零实证 | §3.5 **断网预演**设为打包前置强制闸门 | ⏳ 待构建机执行（本机无 GPU 无法完成） |

### 已实证的依赖链（CUDA 12.8 / Python 3.10 容器内验证）

```
vllm 0.14.0 + pyannote.audio 4.0.7 + transformers 4.57.6 + torch 2.9.1+cu128
→ pip 解析安装成功，全功能模块 import 通过（serve / middleware / diarizer / extensions / scheduler）
```

### pyannote 离线安全性 · 源码级核查结论（本会话实测）

对 pyannote.audio 4.x 安装源码逐项 grep 核查：

- 管线 config 与子模型权重加载**全部**经 `huggingface_hub.hf_hub_download`（`pyannote/audio/utils/hf_hub.py` + `core/model.py:577`）——该函数天然尊重 `HF_HUB_OFFLINE=1`；
- **无任何** `HfApi` / `whoami` 在线校验调用（不存在"revision 校验触网"路径）；
- **无任何** `torch.hub` / `TORCH_HOME` 写入（不存在 `~/.cache/torch` 残留风险）；
- `Pipeline.from_pretrained` 对本地目录走 `isdir` 分支（`config.yaml` 直接读），实测通过。

`BUNDLE_FLASH_ATTENTION=false` 安全：serve 走 vLLM 路径（自带 flash attention 内核），且可大幅缩短构建时间并消除 github 依赖。

---

## 3. 阶段一：联网构建机

### 3.1 分支确认（构建前必做）

方案严格绑定 `dev` 分支——其他分支（如 `main` / `dev-diarize-1`）**不含** segment/说话人功能，构建出的镜像无该功能：

```bash
git checkout dev
# 确认 HEAD 包含全部关键提交（输出 ok 才继续）
git merge-base --is-ancestor 603c8f4 HEAD && echo ok || echo "分支不对，停止构建"
```

### 3.2 构建镜像

```bash
# 国内网络：先从 mirror 拉基础镜像并 retag（直连 docker.io 会超时）
docker pull docker.m.daocloud.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
docker tag  docker.m.daocloud.io/nvidia/cuda:12.8.0-devel-ubuntu22.04 \
            nvidia/cuda:12.8.0-devel-ubuntu22.04
# mirror 卡死可换：docker.1ms.run / dockerproxy.net 等

docker build -f docker/Dockerfile-qwen3-asr-cu128 \
  --build-arg BUNDLE_FLASH_ATTENTION=false \
  -t qwen3-asr-offline:cu128 .
# 有代理的环境可追加：--build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...
```

说明：
- `.dockerignore` 已排除 `.git`、本地模型目录、`*.tar.gz` 等，构建上下文精简；
- 构建上下文即仓库根目录（Dockerfile 内 `COPY . /data/shared/Qwen3-ASR` 后本地安装）。

### 3.3 构建后立即冒烟（镜像内验证）

```bash
docker run --rm qwen3-asr-offline:cu128 python3 -c "
import warnings; warnings.filterwarnings('ignore')
import vllm, pyannote.audio, transformers, torchcodec, torch
from qwen_asr.service.middleware import TranscriptionsMiddleware
print('smoke OK:', pyannote.audio.__version__, torchcodec.__version__)
"
```

预期输出包含 `smoke OK: 4.0.7 0.13.x`。
**torchcodec 若仅打 warning 而功能 import 通过，可接受**——pyannote 走内存 waveform 输入，不依赖其解码器。

### 3.4 打包模型（本地目录原样打包，无任何下载）

```bash
# Windows 构建机（PowerShell，模型在 D:\workplace\TMRI\AI\body_Camera\model\models）
tar -czf qwen3-asr-models.tar.gz `
  -C "D:\workplace\TMRI\AI\body_Camera\model\models" `
  Qwen3-ASR-1.7B Qwen3-ForcedAligner-0.6B pyannote-speaker-diarization-community-1

# Linux 构建机等价：
# tar czf qwen3-asr-models.tar.gz -C /path/to/models \
#   Qwen3-ASR-1.7B Qwen3-ForcedAligner-0.6B pyannote-speaker-diarization-community-1
```

只打包三个目录（~6.3 GB）；`pyannote-segmentation-3.0` / `wespeaker-*` / `diarization-3.1` / `MOSS-*` 均不需要（community-1 不引用，实测证明）。

**打包前完整性检查**（三个目录的关键文件齐全）：

```bash
# community-1 必须含子模型目录（segmentation/embedding/plda 任一缺失即加载失败）
ls "D:\workplace\TMRI\AI\body_Camera\model\models\pyannote-speaker-diarization-community-1"
#   预期：config.yaml + segmentation/ + embedding/ + plda/
```

### 3.5 断网预演（强制闸门：此步不过，禁止打包镜像传输）

用 `--network none` 完全隔离网络，模拟目标机真实离线条件，完整启动一次服务。**本机（无 GPU）无法代做，此步是"离线运行时链路"的最终实证**：

```bash
# 先解包模型目录到 ./models（或直接用原始路径挂载）
mkdir -p rehearsal-models && tar xzf qwen3-asr-models.tar.gz -C rehearsal-models

docker run -d --name offline-rehearsal --network none \
  --gpus all --shm-size 8g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v "$PWD/rehearsal-models:/models:ro" \
  qwen3-asr-offline:cu128 \
  qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner /models/Qwen3-ForcedAligner-0.6B \
    --diarizer /models/pyannote-speaker-diarization-community-1

sleep 180   # 首次加载三模型约 1-3 分钟，按机器调整

# 验证 1：日志无网络错误 / 模型加载成功
docker logs offline-rehearsal 2>&1 | grep -iE "error|401|403|connectionerror|offline" || echo "日志干净"
docker logs offline-rehearsal 2>&1 | grep -iE "aligner|diarizer" | tail -5

# 验证 2：健康检查（容器内 wget，镜像未装 curl）
docker exec offline-rehearsal wget -qO- http://localhost:80/health/detail

# 验证 3（可选但推荐）：断网下发一条真实转写请求
docker cp test.wav offline-rehearsal:/tmp/test.wav
docker exec offline-rehearsal python3 /data/shared/Qwen3-ASR/examples/example_segment_api.py \
  --file /tmp/test.wav --base-url http://localhost:80

# 全部通过后清理
docker rm -f offline-rehearsal
```

任一验证失败 → 按 §8 排查修复后重跑，**禁止带病交付**。

### 3.6 构建热修复镜像（生产交付镜像，必须执行）

GPU 实机首跑（2026-08-20，A10）发现 3 个运行时问题，已修复在仓库源码并通过
`docker/Dockerfile-qwen3-asr-hotfix` 叠加到基础镜像之上：

| # | 修复文件 | 问题 |
|---|---|---|
| fix1 | `qwen_asr/inference/qwen3_speaker_diarizer.py` | pyannote 4.x `Pipeline.to()` 要求 `torch.device` 实例，传 str 抛 TypeError |
| fix2 | `qwen_asr/cli/serve.py` | vLLM 0.14.0 SageMaker bootstrap 预建中间件栈，`add_middleware` 误报已启动 |
| fix3 | `qwen_asr/service/middleware.py` | PyTorch 缓存分配器不归还空闲显存块 → 调度器准入误判，首任务成功后其余请求永久排队 |

```bash
# 仓库根目录执行（基于已存在的 qwen3-asr-offline:cu128，仅追加 3 个源文件层，秒级完成）
docker build -f docker/Dockerfile-qwen3-asr-hotfix -t qwen3-asr-offline:cu128-hotfix .
```

### 3.7 打包镜像

```bash
docker save qwen3-asr-offline:cu128-hotfix | gzip > qwen3-asr-offline-cu128-hotfix.tar.gz

# 当前生产镜像为叠加层，打包方式相同（见 deployment-guide.md §2.5）：
# docker save qwen3-asr-offline:cu128-align-fallback \
#   | gzip > qwen3-asr-offline-cu128-align-fallback.tar.gz
```

最终交付物：镜像 tar（~15 GB）+ `qwen3-asr-models.tar.gz`（~6.3 GB），合计 ~21 GB。

> 目标机部署、启动参数（含必加的 `--max-num-batched-tokens 8192`）与故障排查
> 见 `docs/deployment-guide.md`（部署操作手册）。
> 说话人识别调优见 `docs/diarization-tuning-guide.md`。

## 4. 阶段二：传输

scp / rsync / 移动硬盘均可：

```bash
scp qwen3-asr-offline-cu128.tar.gz qwen3-asr-models.tar.gz user@target:/data/offline/
```

---

## 5. 阶段三：离线目标机部署

### 5.1 解包

```bash
mkdir -p /data/models
cd /data/offline

docker load < qwen3-asr-offline-cu128.tar.gz
tar xzf qwen3-asr-models.tar.gz -C /data/models
# → /data/models/Qwen3-ASR-1.7B
# → /data/models/Qwen3-ForcedAligner-0.6B
# → /data/models/pyannote-speaker-diarization-community-1
```

### 5.2 启动（A10 24GB 单卡，默认参数）

单卡拓扑显式指定 `device=0`，避免多卡目标机暴露全部 GPU。**三个模型全部传容器内本地路径**：

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

单卡默认拓扑下服务自动注入 `gpu_memory_utilization=0.70`（日志可见），支持双并发 1h 音频。

### 5.3 其他拓扑

**两卡（vLLM 独占 GPU0 用满 0.9，扩展在 GPU1）：**

```bash
  --gpus '"device=0,1"' \
  ... qwen-asr-serve /models/Qwen3-ASR-1.7B ... \
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

**T4 16GB 单卡：** 追加 `--gpu-memory-utilization 0.55`（双并发）/ `0.60`（单并发）。

**P4 8GB：** 不推荐承载长音频；仅短音频（≤10 min）可用 `--gpu-memory-utilization 0.35 --gpu-reserve-mb 512 --max-concurrent-tasks 1`。

### 5.4 长音频经反向代理时（如前置 nginx，必改）

```nginx
client_max_body_size 500m;   # 1h wav 约 230MB 上传
proxy_read_timeout 900s;     # 长音频端到端耗时（含排队等待）
proxy_send_timeout 900s;
```

---

## 6. 离线验证清单

| # | 验证项 | 命令 / 方法 | 通过标准 |
|---|---|---|---|
| 1 | 依赖冒烟 | §3.3 的 python -c import（构建机） | `smoke OK` 输出 |
| 2 | 断网预演 | §3.5（构建机，交付前） | 三项验证全过 |
| 3 | 扩展模型加载 | `docker logs qwen3-asr` | aligner / diarizer 加载成功日志，无网络错误 |
| 4 | 显存预算注入 | `docker logs qwen3-asr` | 单卡拓扑可见"自动注入 0.70"日志 |
| 5 | 健康检查 | `curl http://localhost:8000/health/detail` | 200，devices 空闲显存符合预算 |
| 6 | 端到端转写（**容器内执行**，目标机宿主机无需 Python 环境） | 见下方命令 | 返回 segments + speakerSummary |
| 7 | 离线干净性 | `docker logs qwen3-asr` 全程无网络请求报错 | 无 `ConnectionError` / `requests` 相关错误 |

验证 6 的执行方式（镜像内已含 Python 与仓库源码）：

```bash
docker cp test.wav qwen3-asr:/tmp/test.wav
docker exec qwen3-asr python3 /data/shared/Qwen3-ASR/examples/example_segment_api.py \
  --file /tmp/test.wav --base-url http://localhost:80
```

---

## 7. 更新与回滚

| 场景 | 操作 | 影响范围 |
|---|---|---|
| 只更新代码 | 构建机重 build 镜像 → `docker load` → 重建容器 | 模型目录不动 |
| 只更新模型 | 替换 `/data/models` 下对应目录后重启容器 | 镜像不动 |
| 回滚 | 保留旧 tar 包，`docker load` 旧镜像重建容器 | 分钟级恢复 |

---

## 8. 故障排查

| 症状 | 原因 | 解法 |
|---|---|---|
| 构建机拉基础镜像超时 | 国内直连 docker.io 被墙 | §3.2 mirror 前缀拉取 + retag；mirror 卡死换备选或代理 |
| 构建在 `RUN <<EOF` 处语法报错 | 构建机 Docker < 23（heredoc 不支持） | 升级 Docker，或临时在 Dockerfile 首行加 `# syntax=docker/dockerfile:1`（需能拉取 frontend 镜像） |
| 构建在 wget CMake 处超时 | `BUNDLE_FLASH_ATTENTION=true` 触发 github 直连 | 离线部署用 `--build-arg BUNDLE_FLASH_ATTENTION=false`（CMake 已条件化跳过） |
| 启动报 diarizer 加载失败 / 找不到子模型 | models tar 缺 community-1 的 `segmentation/` / `embedding/` / `plda/` 子目录 | §3.4 完整性检查；重新打包（三目录齐全） |
| 容器内 `import torchcodec` 报 libnvrtc/FFmpeg 错误 | torchcodec 与 torch/CUDA/FFmpeg 二进制配对问题 | 已 pin 0.13.x；若仍 warning 可忽略（功能走内存 waveform，不依赖其解码器） |
| 启动报 "无法从 vLLM init_app_state 捕获 engine_client" | 不应出现（已修 P0）；若出现说明镜像内代码版本旧 | §3.1 分支检查（HEAD ≥ `603c8f4`） |
| segment 请求一直排队不放行 | 显存准入未过 | 查 `/health/detail` 空闲显存；调低 `--gpu-memory-utilization` 或降低并发 |
| 上传 1h 音频 413 / 中途断开 | 前置代理 body/timeout 限制 | §5.4 nginx 参数 |
| 目标机 `docker exec` 转写示例失败 | test.wav 未拷入容器 / base-url 端口不对 | §6 验证 6：先 `docker cp`，容器内端口是 80 非 8000 |

---

## 9. 已知限制与遗留事项

### 已知限制（接受并知悉）

- **构建可重复性**：`apt upgrade -y`（系统包随时间漂移）、`torchcodec==0.13.*` 通配（0.13.x 内小版本漂移）、pip 无 lock/constraints——不同时间构建的镜像可能与"已实证依赖链"有偏差。缓解：首次成功构建后 `docker run <image> pip3 freeze > build-manifest.txt` 随产物归档，供后续比对/复现。
- **vLLM 私有 API 耦合**：engine_client 注入钩子依赖 vLLM 0.14 的 `init_app_state` 私有结构。版本已 pin（`vllm==0.14.0`）可控；**未来升级 vLLM 必须重新验证该钩子**（多 API server 进程已被显式拒绝，`--api-server-count > 1` 启动即报错）。
- **镜像体积**：`BUNDLE_FLASH_ATTENTION=false` 时 git-lfs/vim 等仍打入镜像（~200MB 级，相对 15GB 占比小，维持上游行为未裁剪）；CMake 已条件化跳过。
- **模型版本**：本地 `Qwen3-ASR-1.7B` / `Qwen3-ForcedAligner-0.6B` 快照版本未与 HF 最新对齐检查；功能实测以 §3.5 断网预演为准。

### 遗留事项（需 GPU 实机环境，非离线部署阻断项）

- 断网预演（§3.5，需构建机 GPU）
- 默认参数启动不 OOM 实测（A10 / T4）
- 多卡拓扑实测（cuda:1 / cuda:2）
- 1h 音频端到端 + 40min 压测
- 并发 diarization 结果正确性

以上项 checklist 中已标注〔需 GPU 实机环境〕，建议离线部署完成后在目标机上一并执行。
