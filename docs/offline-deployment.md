# Qwen3-ASR segment + 说话人识别服务 · 完全离线部署方案

> 适用分支：`dev`（`f73ed66` 功能实现 + `03b8349` 离线构建改造 + `4a7f92b` torchcodec 修正 + `603c8f4` CMake 条件化）
> 目标：目标机**零网络**运行，所有模型提前下载打包，服务全功能可用（vLLM ASR + 强制对齐 + pyannote 说话人识别）。

---

## 1. 总体思路

### 1.1 产物三件套（联网构建机准备）

| 产物 | 内容 | 体积参考 |
|---|---|---|
| `qwen3-asr-offline-cu128.tar.gz` | Docker 镜像（功能代码 + 全部 Python 依赖 + ffmpeg） | ~15 GB |
| `hf-cache.tar.gz` | HF 缓存（对齐模型 + pyannote 管线 + **全部子模型**） | ~3 GB |
| `qwen3-asr-1.7b.tar.gz` | ASR 主模型目录 | ~4 GB |

### 1.2 离线运行原理

```bash
HF_HUB_OFFLINE=1        # huggingface_hub 全部走本地缓存，不发任何网络请求
TRANSFORMERS_OFFLINE=1  # transformers 同上
VLLM_NO_USAGE_STATS=1   # 禁用 vLLM 遥测上报
DO_NOT_TRACK=1          # 禁用其他工具链统计
```

- 扩展模型传 **repo id**（如 `Qwen/Qwen3-ForcedAligner-0.6B`），离线模式下自动从 `HF_HOME` 本地缓存解析，启动参数形态与联网部署完全一致；
- **运行时不需要 HF_TOKEN**——离线缓存命中不鉴权。Token 仅构建机下载阶段需要。

### 1.3 构建机要求（注意：与目标机要求不同）

| 项 | 要求 | 原因 |
|---|---|---|
| Docker | **≥ 23.0** | Dockerfile 使用 heredoc（`RUN <<EOF`）与 `--mount=type=cache`，需 BuildKit 内置 dockerfile 前端 ≥ 1.4；Docker 20.10 构建会在 heredoc 处直接解析失败。仅 `docker load/run` 的目标机 20.10 即可 |
| 网络 | 见下表 4 个直连点 | 构建期需拉取基础镜像、apt 包、pip 包、（可选）github 资源 |

构建期网络直连点：

| 直连点 | 何时需要 | 国内不通时的对策 |
|---|---|---|
| `docker.io` | 拉取 `nvidia/cuda:12.8.0-devel-ubuntu22.04` | mirror 前缀拉取后 retag（§3.2）；mirror 可用性随时间波动，备选 `docker.m.daocloud.io` / `docker.1ms.run` / `dockerproxy.net`，或 `--build-arg HTTP(S)_PROXY` 走代理（apt/wget/pip 均尊重） |
| `archive.ubuntu.com` | apt 安装系统包 | 同上走代理；或构建前临时改 sources.list 为国内源（自行斟酌） |
| PyPI | pip 安装全部 Python 依赖 | `--build-arg HTTP(S)_PROXY`；注意 Dockerfile 未内置 `PIP_INDEX_URL`，如需固定镜像源需自行改造 |
| `github.com` | **仅** `BUNDLE_FLASH_ATTENTION=true` 时（CMake + flash-attn 源码） | **离线部署推荐 `false`，完全不触 github**（CMake 下载已条件化） |
| `hf.co` | §3.4 预下载模型 | `HF_ENDPOINT=https://hf-mirror.com` |

### 1.4 目标机要求（OS 层，与镜像无关）

- NVIDIA 驱动（`nvidia-smi` 可用，建议 ≥ 535）
- Docker ≥ 20.10（仅 load/run，不构建）+ `nvidia-container-toolkit`

---

## 2. 关键坑位与实证状态

**实证分级说明**：下表区分"已实测 / 已修复（代码级）/ 待构建机执行"，避免过度声明。本机（无 GPU、无 HF token、Docker Hub 被墙）能做的验证已全部做完，其余以强制预演闸门覆盖。

| # | 坑 | 措施 | 状态 |
|---|---|---|---|
| 1 | PyPI 的 `qwen-asr` 不含 segment/说话人功能（未发布） | Dockerfile 改为 `COPY .` 本地源码安装 | ✅ 已提交 `03b8349` |
| 2 | pyannote `community-1` 管线 config 内部引用分割/声纹/VBx 子模型，只 snapshot 管线仓库**不够**，离线加载必失败 | 构建期**真实加载一次管线**拉全子模型（§3.4）+ 缓存完整性检查 | ⏳ 待构建机执行（机制属 huggingface_hub 标准行为） |
| 3 | torchcodec ≥0.14 按 CUDA 13 构建（链接 `libnvrtc.so.13`），CUDA 12.8 镜像内 import 即炸 | Dockerfile pin `torchcodec==0.13.*`（torch 2.9 官方配对） | ✅ 已提交 `4a7f92b`；镜像内复验见 §3.3 |
| 4 | pyannote 门控模型 401/403 | 构建机下载前先到模型页接受访问条款（§3.4） | ⏳ 待构建机执行 |
| 5 | 国内直连 Docker Hub 超时 | mirror 前缀拉取后 retag（§3.2） | ⚠️ 本会话实测 mirror 可用性波动（daocloud 长时间卡死、1ms 慢），建议多备选 + 代理兜底 |
| 6 | HF_TOKEN 打进镜像层有泄露风险 | 模型外挂缓存卷提供，token 不进镜像 | ✅ 已移除构建参数 |
| 7 | pyannote 离线加载路径是否含在线校验调用 | 源码级核查通过（见下） | ✅ 已实测（源码级）；运行时由 §3.6 预演最终把关 |
| 8 | 离线完整启动（三模型加载 + 服务就绪）零实证 | §3.6 **断网预演**设为打包前置强制闸门 | ⏳ 待构建机执行（本机无 GPU 无法完成） |

### 已实证的依赖链（CUDA 12.8 / Python 3.10 容器内验证）

```
vllm 0.14.0 + pyannote.audio 4.0.7 + transformers 4.57.6 + torch 2.9.1+cu128
→ pip 解析安装成功，全功能模块 import 通过（serve / middleware / diarizer / extensions / scheduler）
```

### pyannote 离线安全性 · 源码级核查结论（本会话实测）

对 pyannote.audio 4.x 安装源码逐项 grep 核查：

- 管线 config 与子模型权重加载**全部**经 `huggingface_hub.hf_hub_download`（`pyannote/audio/utils/hf_hub.py` + `core/model.py:577`）——该函数天然尊重 `HF_HUB_OFFLINE=1`；
- **无任何** `HfApi` / `whoami` 在线校验调用（不存在"revision 校验触网"路径）；
- **无任何** `torch.hub` / `TORCH_HOME` 写入（不存在 `~/.cache/torch` 残留风险）。

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

### 3.4 预填充 HF 缓存（关键步骤：子模型必须拉全）

```bash
mkdir -p hf-cache

# 避免明文 token 进 shell 历史：
read -s HF_TOKEN && export HF_TOKEN   # 粘贴 token 后回车，输入不可见

docker run --rm -v "$PWD/hf-cache:/root/.cache/huggingface" \
  -e HF_TOKEN \
  -e HF_ENDPOINT=https://hf-mirror.com \
  qwen3-asr-offline:cu128 \
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B')
snapshot_download('pyannote/speaker-diarization-community-1')
from pyannote.audio import Pipeline
Pipeline.from_pretrained('pyannote/speaker-diarization-community-1')  # 关键：拉全子模型
print('offline cache ready')
"
```

- 遇 401/403：用同一 HF 账号到 `pyannote/speaker-diarization-community-1` 及报错的子模型页（如 `pyannote/segmentation-3.0`、wespeaker 声纹模型页）接受访问条款后重跑；
- `HF_ENDPOINT` 可按需去掉（直连 hf.co 可达时）。

**缓存完整性检查（打包前必做）**：

```bash
# 1) 子模型已拉全：除两个主仓库外，还应出现若干子模型目录（分割/声纹/VBx 聚类等）
ls hf-cache/hub/ | sort
#    预期至少包含：
#      models--Qwen--Qwen3-ForcedAligner-0.6B
#      models--pyannote--speaker-diarization-community-1
#      models--pyannote--*（若干子模型，具体 repo 以 community-1 config 为准）
#    缺任何一个 → 目标机离线加载必失败 → 重跑上面的加载命令

# 2) 无 HF 缓存以外的残留写入（torch.hub 等）
docker run --rm -v "$PWD/hf-cache:/root/.cache/huggingface" \
  qwen3-asr-offline:cu128 \
  bash -c "find /root/.cache -maxdepth 2 -not -path '*/huggingface*' | head -20"
#    预期仅剩 pip/ccache 等构建缓存目录；若出现 torch/ 目录，需一并打包并设 TORCH_HOME
#    （源码核查表明 pyannote 不写 torch.hub，此步为廉价保险）
```

### 3.5 下载 ASR 主模型

```bash
hf download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
# 旧版 huggingface_hub 无 hf 命令时等价：huggingface-cli download ...（会有 deprecation 警告）

# 国内备选（ModelScope）：
# modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B
```

### 3.6 断网预演（强制闸门：此步不过，禁止打包）

用 `--network none` 完全隔离网络，模拟目标机真实离线条件（无 HF、无 token、无代理），完整启动一次服务。**本机（无 GPU）无法代做，此步是"离线运行时链路"的最终实证**：

```bash
docker run -d --name offline-rehearsal --network none \
  --gpus all --shm-size 8g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v "$PWD/hf-cache:/root/.cache/huggingface" \
  -v "$PWD/Qwen3-ASR-1.7B:/models/asr:ro" \
  qwen3-asr-offline:cu128 \
  qwen-asr-serve /models/asr --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B \
    --diarizer pyannote/speaker-diarization-community-1

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

任一验证失败 → 按 §8 排查修复后重跑，**禁止带病打包**。

### 3.7 打包三件套

```bash
docker save qwen3-asr-offline:cu128 | gzip > qwen3-asr-offline-cu128.tar.gz
tar czf hf-cache.tar.gz -C hf-cache hub
tar czf qwen3-asr-1.7b.tar.gz Qwen3-ASR-1.7B
```

---

## 4. 阶段二：传输

scp / rsync / 移动硬盘均可，合计 ~22 GB：

```bash
scp qwen3-asr-offline-cu128.tar.gz hf-cache.tar.gz qwen3-asr-1.7b.tar.gz user@target:/data/offline/
```

---

## 5. 阶段三：离线目标机部署

### 5.1 解包

```bash
mkdir -p /data/models/hf /data/models
cd /data/offline

docker load < qwen3-asr-offline-cu128.tar.gz
tar xzf hf-cache.tar.gz -C /data/models/hf        # → /data/models/hf/hub/models--...
tar xzf qwen3-asr-1.7b.tar.gz -C /data/models     # → /data/models/Qwen3-ASR-1.7B
```

### 5.2 启动（A10 24GB 单卡，默认参数）

单卡拓扑显式指定 `device=0`，避免多卡目标机暴露全部 GPU：

```bash
docker run -d --name qwen3-asr --restart unless-stopped \
  --gpus '"device=0"' --shm-size 8g -p 8000:80 \
  -e HF_HOME=/data/models/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v /data/models/hf:/data/models/hf \
  -v /data/models/Qwen3-ASR-1.7B:/models/asr:ro \
  qwen3-asr-offline:cu128 \
  qwen-asr-serve /models/asr --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B \
    --diarizer pyannote/speaker-diarization-community-1
```

单卡默认拓扑下服务自动注入 `gpu_memory_utilization=0.70`（日志可见），支持双并发 1h 音频。

### 5.3 其他拓扑

**两卡（vLLM 独占 GPU0 用满 0.9，扩展在 GPU1）：**

```bash
  --gpus '"device=0,1"' \
  ... qwen-asr-serve /models/asr ... \
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
| 2 | 断网预演 | §3.6（构建机，打包前） | 三项验证全过 |
| 3 | 扩展模型加载 | `docker logs qwen3-asr` | aligner / diarizer 加载成功日志，无 401/网络错误 |
| 4 | 显存预算注入 | `docker logs qwen3-asr` | 单卡拓扑可见"自动注入 0.70"日志 |
| 5 | 健康检查 | `curl http://localhost:8000/health/detail` | 200，devices 空闲显存符合预算 |
| 6 | 端到端转写（**容器内执行**，目标机宿主机无需 Python 环境） | 见下方命令 | 返回 segments + speakerSummary |
| 7 | 离线干净性 | `docker logs qwen3-asr` 全程无 HF 网络请求报错 | 无 `ConnectionError` / `requests` 相关错误 |

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
| 只更新代码 | 构建机重 build 镜像 → `docker load` → 重建容器 | 模型缓存/主模型不动 |
| 只更新 ASR 主模型 | 替换 `/data/models/Qwen3-ASR-1.7B` 后重启容器 | 镜像/扩展缓存不动 |
| 回滚 | 保留旧 tar 包，`docker load` 旧镜像重建容器 | 分钟级恢复 |

---

## 8. 故障排查

| 症状 | 原因 | 解法 |
|---|---|---|
| 构建机拉基础镜像超时 | 国内直连 docker.io 被墙 | §3.2 mirror 前缀拉取 + retag；mirror 卡死换备选或代理 |
| 构建在 `RUN <<EOF` 处语法报错 | 构建机 Docker < 23（heredoc 不支持） | 升级 Docker，或临时在 Dockerfile 首行加 `# syntax=docker/dockerfile:1`（需能拉取 frontend 镜像） |
| 构建在 wget CMake 处超时 | `BUNDLE_FLASH_ATTENTION=true` 触发 github 直连 | 离线部署用 `--build-arg BUNDLE_FLASH_ATTENTION=false`（CMake 已条件化跳过） |
| 下载 pyannote 模型 401/403 | 门控模型未接受条款 | HF 账号到对应模型页 Accept 后重跑 §3.4 |
| §3.6 预演卡在模型解析/报缓存缺失 | HF 缓存缺子模型 | 重跑 §3.4（真实加载管线拉全子模型）+ 完整性检查 |
| 容器内 `import torchcodec` 报 libnvrtc/FFmpeg 错误 | torchcodec 与 torch/CUDA/FFmpeg 二进制配对问题 | 已 pin 0.13.x；若仍 warning 可忽略（功能走内存 waveform，不依赖其解码器） |
| 启动报 "无法从 vLLM init_app_state 捕获 engine_client" | 不应出现（已修 P0）；若出现说明镜像内代码版本旧 | §3.1 分支检查（HEAD ≥ `4a7f92b`） |
| segment 请求一直排队不放行 | 显存准入未过 | 查 `/health/detail` 空闲显存；调低 `--gpu-memory-utilization` 或降低并发 |
| 上传 1h 音频 413 / 中途断开 | 前置代理 body/timeout 限制 | §5.4 nginx 参数 |
| 目标机 `docker exec` 转写示例失败 | test.wav 未拷入容器 / base-url 端口不对 | §6 验证 6：先 `docker cp`，容器内端口是 80 非 8000 |

---

## 9. 已知限制与遗留事项

### 已知限制（接受并知悉）

- **构建可重复性**：`apt upgrade -y`（系统包随时间漂移）、`torchcodec==0.13.*` 通配（0.13.x 内小版本漂移）、pip 无 lock/constraints——不同时间构建的镜像可能与"已实证依赖链"有偏差。缓解：首次成功构建后 `docker run <image> pip3 freeze > build-manifest.txt` 随产物归档，供后续比对/复现。
- **vLLM 私有 API 耦合**：engine_client 注入钩子依赖 vLLM 0.14 的 `init_app_state` 私有结构。版本已 pin（`vllm==0.14.0`）可控；**未来升级 vLLM 必须重新验证该钩子**（多 API server 进程已被显式拒绝，`--api-server-count > 1` 启动即报错）。
- **镜像体积**：`BUNDLE_FLASH_ATTENTION=false` 时 git-lfs/vim 等仍打入镜像（~200MB 级，相对 15GB 占比小，维持上游行为未裁剪）；CMake 已条件化跳过。
- **token 安全**：§3.4 使用 `read -s` 避免明文进 shell 历史；token 只在构建机下载阶段存在，不进镜像层。

### 遗留事项（需 GPU 实机环境，非离线部署阻断项）

- 断网预演（§3.6，需构建机 GPU）
- 默认参数启动不 OOM 实测（A10 / T4）
- 多卡拓扑实测（cuda:1 / cuda:2）
- 1h 音频端到端 + 40min 压测
- 并发 diarization 结果正确性

以上项 checklist 中已标注〔需 GPU 实机环境〕，建议离线部署完成后在目标机上一并执行。
