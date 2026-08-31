# 说话人识别（diarization）调优指南

> 适用范围：`qwen3-asr-offline:cu128-punct2` 及之后镜像，segment 模式（`timestamp_granularities[]=segment`）。
> 本文仅涉及**启动参数调整**，不含代码改动。

## 1. 问题定位：说话人错归类的三个层面

| 层 | 性质 | 是否可配参解决 |
|---|---|---|
| **① 声纹模型域不匹配** | WeSpeaker 由英文 VoxCeleb 训练，中文男声在其向量空间距离过小 → 聚类欠分割、两人判成一人 | ✅ 换 CAM++ |
| **② 聚类约束不当** | 只约束了下限未约束上限 → 单人独白被强行劈成两类，或人数不定时刻意合并 | ✅ 补参数 |
| **③ punctuation 模式 dominant 投票** | 默认 `punctuation` 模式**刻意不在说话人变化处切分**，段 `speaker` 取段内词归属票数多者 | ⚠️ 设计取舍，需 A/B |

**排查顺序必须是 ① → ② → ③**：若 diarizer 本身把 A/B 判成同一人，换任何切分模式都救不回来。

## 2. 当前配置的问题

以生产启动命令为例：

```
--diarizer /models/pyannote-speaker-diarization-community-1 \
--diarization-min-speakers 2 \
--diarization-clustering-threshold 0.5
```

| 项 | 状态 | 说明 |
|---|---|---|
| `--diarizer-embedding` | **未传 → 默认 `wespeaker`** | CAM++ 根治方案**未启用**，仍在用英文域声纹 |
| `--diarization-max-speakers` | **未传 → None** | 只约束了下限，存在过分割风险 |
| `--diarization-clustering-threshold 0.5` | **需确认是否生效** | community-1 走 VBx 聚类，`instantiate` 探测大概率失败 |

### 2.1 CAM++ 未启用（影响最大）

`add-punct-split-mode-and-diarization-tuning` spec 已定位：WeSpeaker 的中文男声区分能力不足是「两男被识别成一个 speaker」的根因，并为此集成了 CAM++（`speech_campplus_sv_zh-cn_16k-common`，约 200k 中文说话人训练，CN-Celeb EER 4.32%）。**不使用该参数等于该 spec 的核心修复未生效。**

### 2.2 只约束 `min-speakers` 的过分割风险

`--diarization-min-speakers 2` 强制聚类至少输出 2 类。执法记录仪音频**并非每段都严格 2 人对话**——民警单方陈述、长时间独白、只有当事人说话的片段都会被**强行劈成 2 个 speaker**。

### 2.3 聚类阈值需先验证生效

阈值应用为 best-effort 三候选探测（`instantiate` 超参 → `clustering_threshold` 属性 → 嵌套 hparams 覆写）。community-1 使用 VBx 聚类，`instantiate` 大概率抛异常降级。

**必须先确认日志**，否则该参数是无效的：

```bash
docker logs qwen3-asr 2>&1 | grep -E "聚类阈值|embedding="
```

- 出现 `说话人聚类阈值已覆写为 0.5（生效机制: xxx）` → **生效**
- 出现 `聚类阈值 0.5 应用失败：当前管线不支持任一探测机制` → **未生效**，正按管线默认阈值运行

> 即使生效也要注意方向：spec 的「默认值纪律」明确不在离线环境猜测管线默认阈值。若默认值本就低于 0.5，则该设定反而**更倾向合并**，适得其反，需实测确认。

## 3. 改动清单（按优先级）

### 步骤 1：补齐人数上界（零风险，先做）

场景固定 2 人对话：

```bash
--diarization-min-speakers 2 --diarization-max-speakers 2
```

人数不固定：

```bash
# 去掉 --diarization-min-speakers，改由请求级 min_speakers/max_speakers 逐请求指定
```

> 请求级 form 参数 `min_speakers` / `max_speakers` 逐参数覆盖服务级默认；`min > max` 返回 400（错误消息会标明值来源）。

### 步骤 2：启用 CAM++（收益最大）

前置：`/models` 下挂载 CAM++ 模型目录（约 30MB，ModelScope `iic/speech_campplus_sv_zh-cn_16k-common` 产物，须含 `campplus_cn_common.bin` 与 `config.yaml`）。

```bash
--diarizer-embedding campplus \
--diarizer-embedding-model /models/speech_campplus_sv_zh-cn_16k-common
```

**fail fast 语义**：目录缺失/权重损坏/注入失败 → 启动即 `RuntimeError`（中文消息含期望文件清单与 `--diarizer-embedding wespeaker` 回退提示），**不静默回退**——静默回退会让问题看起来已修而实际未修。

**一键回退**：改回 `--diarizer-embedding wespeaker`（或删除该参数）。

### 步骤 3：切分模式 A/B（最后评估，属取舍）

默认 `punctuation` 模式下句中说话人变化**不触发切分**，段 `speaker` 为段内词归属 dominant。快问快答场景下一方的发言会被整体归入另一方。

对照：

```bash
--segment-split-mode hybrid
```

hybrid 会在说话人变化处切分（无标点处照切），但会退回「一句话被拆两段」的旧问题。

**判定建议**：若业务更看重说话人准确而非整句完整，用 hybrid；否则保持 punctuation，并由下游按 `speakers` 字段（段内出现过的说话人去重集合，按词数降序）辅助判断，而非只看 `speaker`。

## 4. A/B 验证步骤

### 4.1 准备样本

- 5~10 段真实执法音频，覆盖：两人对话、单人独白、三人以上、快问快答
- 人工标注每段的真实说话人数与主要说话人

### 4.2 基线采集（改参数前）

对每段音频记录：

| 指标 | 采集方式 |
|---|---|
| `speakerCount` | 响应 `speakerSummary.speakerCount` |
| 段内混合度 | 统计 `segments[].speakers` 长度 ≥ 2 的段占比 |
| dominant 票数占比 | 段内 dominant 词数 / 总词数（反映 dominant 投票的可信度） |
| 说话人是否串台 | 人工核对 `segments[].speaker` 与标注 |

### 4.3 逐项变更、逐项复测

严格**一次只改一个变量**：

1. 基线（当前配置）
2. 加 `--diarization-max-speakers 2`
3. 加 `--diarizer-embedding campplus --diarizer-embedding-model <dir>`
4. 加 `--segment-split-mode hybrid`（最后评估）

每次变更后重启容器，重跑同批样本，记录上表指标。

### 4.4 判定标准

| 指标 | 达标方向 |
|---|---|
| 单人独白音频的 `speakerCount` | 应等于 1（验证过分割是否消除） |
| 两人对话音频的 `speakerCount` | 应等于 2（验证欠分割是否消除） |
| `speakers` 长度 ≥ 2 的段占比 | 越低越好（说明段内不再混人） |
| dominant 票数占比 | 越接近 1 越好（说明 dominant 可信） |

若步骤 3 后「两人判成一人」的比例**未显著下降**，按 spec 约定：保持 `wespeaker` 默认，记录结论，触发 VBx + PLDA 重估升级路径的立项评估（需 CN-Celeb 类标注语料，非当前范围）。

## 5. 参数速查

| 参数 | 默认 | 作用 | 备注 |
|---|---|---|---|
| `--diarizer-embedding` | `wespeaker` | 声纹模型：`wespeaker`（英文域）/ `campplus`（中文域） | `campplus` 需配 `--diarizer-embedding-model` |
| `--diarizer-embedding-model` | None | CAM++ 模型目录 | 仅 `campplus` 生效 |
| `--diarization-min-speakers` | None | 聚类类数下限 | 服务级默认，请求级可覆盖 |
| `--diarization-max-speakers` | None | 聚类类数上限 | 服务级默认，请求级可覆盖 |
| `--diarization-clustering-threshold` | None（管线默认） | 聚类阈值，合法区间 `(0, 2)` | **调低更倾向拆分**；过度调低会过分割一人成多。须先在日志确认生效机制 |
| `--segment-split-mode` | `punctuation` | `punctuation` 不按说话人切分 / `hybrid` 按说话人切分 | 取舍项，需 A/B |

> 注意：`--segment-gap-threshold` 与 `--speaker-merge-gap` **仅在 `hybrid` 模式生效**，punctuation 模式下为无操作参数。

## 6. 与切分逻辑的交互

`text-first-segmentation` 重构后，粗粒度兜底段（对齐失败块）由「整块 180s 单段投票」改为「块内按标点切分后逐段归属」，说话人归属粒度相应细化——兜底块越多，该改善越明显。

未定位的粗段（字符区间反查失败）仍退化为整块单段投票，属兜底路径，发生概率低。
