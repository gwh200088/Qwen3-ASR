# 内网增量构建操作手册（diarize-tune 镜像）

面向"**内网目标机已有 `qwen3-asr-offline:cu128-align-fallback`，只需把新代码送进去重新构建**"的场景。

本手册解决一个问题：镜像约 15GB，每次改代码都 `docker save` + 传输 + `docker load` 代价太高。改用**叠加层镜像**——`FROM` 指向内网已有的 `align-fallback`，只覆盖 `qwen_asr/` 源码目录，构建上下文**约 780KB（tar.gz 193KB）**，内网构建秒级完成。

相关文档：`docs/deployment-guide.md`（完整部署手册）、`docs/diarization-tuning-guide.md`（说话人调参）。

---

## 1. 镜像层级关系

三个镜像是**逐层叠加**关系，每层都在上一层基础上全量覆盖 `qwen_asr/` 源码（依赖不变，不重装）：

| 镜像 tag | FROM | 相对上一层的增量 |
|---|---|---|
| `cu128-punct2` | `cu128-punct` | 标点感知分段 + CAM++ 声纹 wrapper + 切分模式参数 |
| `cu128-align-fallback` | `cu128-punct` | 对齐逐块空 items 兜底 + **文本优先切分解耦** |
| `cu128-diarize-tune` | `cu128-align-fallback` | **AHC `min_cluster_size` 参数暴露**（本手册） |

> 每一层都用"先删旧包代码目录、再整体复制"的全量覆盖方式，因此**只依赖 FROM 指定的那一层**，中间缺失不影响。例如 `diarize-tune` 的 Dockerfile 只要求目标机有 `align-fallback`。

**本层新增能力（`--diarization-min-cluster-size`）**

解决"对话中发言很少的说话人被吞掉、误并入他人"：

- pyannote AHC 聚类**结束后**，会把样本数少于 `min_cluster_size` 的簇**整个合并到最近簇**。该值硬编码为 **12**，此前无法调整；
- 执法记录仪场景常见"一方只说了几句话"，其簇远小于 12 → 被整体吞掉并并入另一方，表现为"两人被识别成一个 speaker"；
- 关键点：**这个问题调 `--diarization-clustering-threshold` 救不回来**。阈值只决定聚类阶段怎么切，管不了事后的小簇合并。两者是不同阶段的不同超参；
- 现在该值可作为启动参数调整（正整数，建议 2~4；设为 1 等于关闭小簇合并，但需配合阈值防止过分割）。

---

## 2. 前置检查（内网目标机）

```bash
# 1) 基础镜像必须存在（本层 FROM 依赖）
docker images | grep align-fallback
# 预期：qwen3-asr-offline   cu128-align-fallback   ...   约 15GB

# 2) 磁盘：本层约 780KB 源码 + 镜像层增量，1GB 余量足够
df -h /var/lib/docker

# 3) CAM++ 模型目录（本层参数要生效就必须有它）
ls /data/models/speech_campplus_sv_zh-cn_16k-common
# 预期：campplus_cn_common.bin  config.yaml   （约 27MB）
```

若第 1 项为空，先把基础镜像 load 进去：

```bash
docker load < qwen3-asr-offline-cu128-align-fallback.tar.gz
```

若第 3 项缺失：本层参数会被忽略并打 WARNING（服务仍能启动，但说话人问题不会改善）。需先从外网机导出 modelscope 的 `speech_campplus_sv_zh-cn_16k-common` 目录再拷入。

---

## 3. 步骤一：本机生成补丁包

> `dist/` 目录已被 `.gitignore` 忽略（不进仓库），因此**每次都要重新生成**。以下步骤在仓库根目录执行。

### 3.1 Windows（PowerShell）

```powershell
# 1) 组装补丁目录
Remove-Item -Recurse -Force dist\intranet-patch -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path dist\intranet-patch -Force | Out-Null
Copy-Item docker\Dockerfile-qwen3-asr-diarize-tune dist\intranet-patch\Dockerfile
Copy-Item qwen_asr dist\intranet-patch\qwen_asr -Recurse

# 2) 清理字节码缓存
#    注意用 -Filter 而不是 -Include：Get-ChildItem -Recurse -Include 在 PowerShell 5.1
#    下会静默失效（不报错、也不删）。实测该写法会把 64 个 .pyc 打进包里，
#    包体积从 193KB 涨到 710KB，且不易察觉。
Get-ChildItem dist\intranet-patch -Recurse -Directory -Filter __pycache__ |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem dist\intranet-patch -Recurse -File -Filter *.pyc |
  Remove-Item -Force -ErrorAction SilentlyContinue

# 3)【关键】Dockerfile 必须转 LF（且必须保留 UTF-8 编码，见下方说明）
$p = "dist\intranet-patch\Dockerfile"
[System.IO.File]::WriteAllText($p,
  ([System.IO.File]::ReadAllText($p) -replace "`r`n", "`n"),
  [System.Text.UTF8Encoding]::new($false))

# 4) 打包
tar -czf dist\qwen3-asr-diarize-tune-patch.tar.gz -C dist\intranet-patch Dockerfile qwen_asr
```

### 3.2 Linux / macOS

```bash
rm -rf dist/intranet-patch && mkdir -p dist/intranet-patch
cp docker/Dockerfile-qwen3-asr-diarize-tune dist/intranet-patch/Dockerfile
cp -r qwen_asr dist/intranet-patch/
find dist/intranet-patch -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
find dist/intranet-patch -name '*.pyc' -delete 2>/dev/null

# 【关键】Dockerfile 转 LF
sed -i 's/\r$//' dist/intranet-patch/Dockerfile

tar czf dist/qwen3-asr-diarize-tune-patch.tar.gz -C dist/intranet-patch Dockerfile qwen_asr
```

### 3.3 为什么 Dockerfile 必须转 LF

Windows 工作区取出的 Dockerfile 是 CRLF。Docker 构建时，`RUN` 指令用 `\` 续行，**行尾残留的 `\r` 会被 shell 当作命令名的一部分**：

```
/bin/sh: 1: python3 -m compileall -q /usr/.../qwen_asr\r: not found
```

表现是构建在校验步骤报出一串带 `^M` 的诡异报错，而非直接的语法错误——排查成本很高。**Python 源文件是 CRLF 没关系**（Python 在 Linux 上能正常处理），只需转 Dockerfile。

> **不要用 `Set-Content -Encoding ascii` 做转换。** 实测同一个文件：正确写法 52 字节，
> `-Encoding ascii` 后只剩 30 字节——中文注释被整段替换成 `?`。镜像照样能构建成功，
> 只是注释全成问号，极难察觉。上面的 .NET 写法保留 UTF-8 且不带 BOM（PowerShell 5.1
> 的 `Set-Content -Encoding utf8` 会加 BOM，同样应避开）。

### 3.4 产物结构

```
dist/intranet-patch/
├── Dockerfile          # 3.9 KB，已改名为 Dockerfile，构建时不用 -f
└── qwen_asr/           # 778 KB：21 个 .py + inference/assets/korean_dict_jieba.dict (296KB)
```

打包后 `tar.gz` 约 **193 KB**。若明显偏大（例如 700KB 量级），说明上一步的 `.pyc`
清理没生效，按 §3.1 注释改用 `-Filter` 重做。

生成后自检（推荐，能同时抓出 CRLF 与 `.pyc` 两类污染）：

```bash
# 解包回读（跨平台，Windows 同样可跑）
mkdir -p /tmp/v && tar xzf dist/qwen3-asr-diarize-tune-patch.tar.gz -C /tmp/v

python3 -c "print('CRLF:', open('/tmp/v/Dockerfile','rb').read().count(b'\r\n'))"
# 预期 CRLF: 0

find /tmp/v/qwen_asr -type f | wc -l          # 预期 22
find /tmp/v/qwen_asr -name '*.pyc' | wc -l    # 预期 0
ls /tmp/v                                     # 预期 Dockerfile qwen_asr
```

三个预期值任一不符都不要上传——`CRLF` 非 0 会导致构建报 `\r: not found`（§3.3），
`.pyc` 非 0 说明打进了 Windows 平台字节码（虽会被容器忽略，但掩盖了真实文件数）。

---

## 4. 步骤二：传到内网

单文件传输（推荐，193KB）：

```bash
scp dist/qwen3-asr-diarize-tune-patch.tar.gz user@<target>:/data/
```

或 U 盘拷贝 `dist/intranet-patch/` 整个文件夹。

---

## 5. 步骤三：内网构建

```bash
mkdir -p /data/patch && tar xzf /data/qwen3-asr-diarize-tune-patch.tar.gz -C /data/patch
cd /data/patch && docker build -t qwen3-asr-offline:cu128-diarize-tune .
```

> **用新 tag，不要覆盖 `cu128-align-fallback`**——保留回退点，出问题时能一键切回（见 §8）。

### 5.1 构建期自动校验（任一不过则构建失败）

Dockerfile 内置三条，无需手工执行：

| # | 校验 | 拦截的问题 |
|---|---|---|
| 1 | `compileall` 全量语法编译 | 文件传输截断、编码损坏 |
| 2 | `pipeline.self_test()` | 分段逻辑回归（含文本优先切分 + 脱敏回归 fixture） |
| 3 | `grep` 新参数与 `apply_clustering_hparams` | **源码是旧的/漏拷**——这是最容易发生且最难察觉的失效 |

第 3 条是故意加的：叠加层构建太快，如果源码漏拷或用了旧快照，**构建会成功产出一个"看起来是新版、实际是旧版"的镜像**，到线上排查时极难定位。这条 grep 让它在构建期就失败。

预期构建输出末尾包含：

```
pipeline self_test ok
Successfully tagged qwen3-asr-offline:cu128-diarize-tune
```

---

## 6. 步骤四：启动服务

### 6.1 双卡环境（GPU 0,1；对齐器与 diarizer 走 cuda:1）

```bash
docker rm -f qwen3-asr
docker run --security-opt seccomp=unconfined -d --name qwen3-asr --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 \
  --shm-size 8g -p 8000:80 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v /data/models:/models:ro \
  qwen3-asr-offline:cu128-diarize-tune \
  qwen-asr-serve /models/Qwen3-ASR-1.7B --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 80 \
    --forced-aligner /models/Qwen3-ForcedAligner-0.6B \
    --diarizer /models/pyannote-speaker-diarization-community-1 \
    --diarization-min-speakers 2 \
    --diarization-clustering-threshold 0.5 \
    --diarizer-embedding campplus \
    --diarizer-embedding-model /models/speech_campplus_sv_zh-cn_16k-common \
    --diarization-min-cluster-size 3 \
    --aligner-device cuda:1 --diarizer-device cuda:1 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192
```

### 6.2 相比你当前命令改了三处

| 改动 | 原值 | 新值 | 必要性 |
|---|---|---|---|
| 镜像 tag | `cu128-align-fallback` | `cu128-diarize-tune` | 必需 |
| `--diarizer-embedding` + `--diarizer-embedding-model` | 无 | `campplus` + 模型路径 | **`min_cluster_size` 生效的前提** |
| `--diarization-min-cluster-size` | 无 | `3` | 本层目的 |

### 6.3 单卡环境改法

```bash
  --gpus '"device=0"' \                    # 替换 --runtime/--gpus 那两行
  --gpu-memory-utilization 0.70 \          # 单卡显存更紧
  # 去掉 --aligner-device / --diarizer-device 两行（单卡自动同源）
```

其余参数不变。

---

## 7. 步骤五：验证

### 7.1 确认镜像内是新代码

```bash
docker run --rm qwen3-asr-offline:cu128-diarize-tune \
  grep -c 'diarization-min-cluster-size' \
  /usr/local/lib/python3.10/dist-packages/qwen_asr/cli/serve.py
# 预期 ≥ 1
```

### 7.2 启动日志（关键两条，缺一即未生效）

```bash
docker logs qwen3-asr 2>&1 | grep -E "CAM\+\+|聚类超参"
```

预期输出：

```
CAM++ 注入完成（生效机制: 加载后组件替换）: embedding=CAM++(192 维, 16000) → clustering=AgglomerativeClustering(method=centroid, threshold=0.515771, min_cluster_size=12, metric=cosine)；说话人聚类切换为 3.1 式 AHC 余弦路径。
说话人聚类超参已覆写为 {'threshold': 0.5, 'min_cluster_size': 3}（生效机制: instantiate；threshold 调低更倾向拆分说话人...
```

- 第一条缺失 → CAM++ 未注入，检查模型目录（§9 #4）；
- 第二条缺失 → 参数未生效，检查是否漏加 `--diarizer-embedding campplus`（§9 #3）；
- 第二条里 `min_cluster_size` 仍是 12 → 参数被忽略，见 §9 #3。

### 7.3 端到端冒烟

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@/tmp/test.wav" -F "response_format=segment" | head -c 400
```

随后按 `docs/deployment-guide.md` §7.4 做三项切分质量校验（拼接无损 / 段数与最长段 / 兜底块日志）。

### 7.4 说话人数量 A/B（本层效果验证）

拿一段已知是两人对话、且其中一方发言很少的音频，分别用三档 `min_cluster_size` 跑，对比 `speakers` 数量与段归属：

| `min_cluster_size` | 语义 | 适用 |
|---|---|---|
| 12（默认） | 小簇（<12 样本）被合并 | 双方发言均衡 |
| 3（建议起点） | 保留发言极少的一方 | **执法/问询场景** |
| 1 | 完全关闭小簇合并 | 需同时收紧 `--diarization-clustering-threshold`，否则易把一人切成多个 |

判据：`segments[].speaker` 的取值集合大小是否等于实际人数；发言少的一方是否有独立 `speaker` 值。

---

## 8. 回滚

镜像是叠加层，回滚即换回上一层镜像 + 去掉新参数：

```bash
docker rm -f qwen3-asr
# 用回 align-fallback 镜像，启动命令去掉 --diarization-min-cluster-size
# （以及可选去掉 campplus 两行）
```

无需重新 load 镜像——`cu128-align-fallback` 一直在本地。

---

## 9. 故障排查

### #1 构建报 `pull access denied for qwen3-asr-offline:cu128-align-fallback`

基础镜像不存在。执行 `docker load < qwen3-asr-offline-cu128-align-fallback.tar.gz`，或临时补别名：

```bash
docker tag <现有镜像:tag> qwen3-asr-offline:cu128-align-fallback
```

### #2 构建在校验步骤报 `...\r: not found`

Dockerfile 是 CRLF。按 §3.3 转换后重新打包（193KB 的包整体重传比重传单个文件更省事）。

### #3 启动日志没有"聚类超参已覆写"，或 `min_cluster_size` 仍是 12

参数被忽略了，三种可能：

- **没加 `--diarizer-embedding campplus`**：日志里会有 `仅 --diarizer-embedding campplus 生效...已忽略` 的 WARNING —— `grep WARNING` 确认；
- **模型目录不存在**：CAM++ 注入失败，第一条日志也不会出现；
- **参数值非法**：`_positive_int` 只接受正整数，`0` 或负数会被 argparse 直接拒绝（构建期不会拦，启动期报错）。

### #4 CAM++ 模型目录缺失

```
ls /data/models/speech_campplus_sv_zh-cn_16k-common
```

缺失时服务仍能启动（wespeaker 路径），但 `min_cluster_size` 无效。需从外网机导出后拷入：

```bash
# 外网机
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common --local_dir ./campplus
tar czf campplus.tar.gz campplus
# 拷到内网 /data/models/ 下解包
```

### #5 构建成功但服务行为与旧版一致

先执行 §7.1 确认镜像内源码版本。若为 0，说明补丁包里的 `qwen_asr/` 不是最新——重新执行 §3 生成。

---

## 10. 参数速查

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--diarization-min-cluster-size` | 正整数 | 12（管线值） | 聚类后样本数少于该值的簇被合并到最近簇。仅 `campplus` 生效。建议 2~4 |
| `--diarization-clustering-threshold` | (0, 2) | 模型 config.yaml | 调低更倾向拆分说话人。**与小簇合并是不同阶段**，不能互相替代 |
| `--diarizer-embedding` | `wespeaker`\|`campplus` | `wespeaker` | `min_cluster_size` 需设为 `campplus` |
| `--diarizer-embedding-model` | 路径 | None | `campplus` 时必填 |
| `--diarization-min-speakers` | 整数 | None | 已有参数，本次未改动 |
| `--diarization-max-speakers` | 整数 | None | 已有参数，本次未改动 |

---

## 11. 本次改动清单

代码改动 3 个文件（commit `15bef31`）：

- `qwen_asr/cli/serve.py` — 新增 `--diarization-min-cluster-size`（`_positive_int` 校验）
- `qwen_asr/inference/qwen3_speaker_diarizer.py` — 新增 `apply_clustering_hparams()`，多个 AHC 超参一次性写入
- `qwen_asr/service/extensions.py` — 参数接线 + 非 campplus 场景的 WARNING 与置空

构建产物 1 个文件（commit `768852a`）：

- `docker/Dockerfile-qwen3-asr-diarize-tune` — 本手册对应的叠加层 Dockerfile

> 为什么多超参要一次性写入：pyannote 的 `instantiate` 只覆盖 dict 里给出的键，分次调用一旦中途失败会留下"部分生效"的中间态，而日志无法区分实际配置。因此实现上统一走 `apply_clustering_hparams()`；仅传 threshold 时仍沿用既有的 `apply_clustering_threshold`，保证 wespeaker 路径行为零变化。
