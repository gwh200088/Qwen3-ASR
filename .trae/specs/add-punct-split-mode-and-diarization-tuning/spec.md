# 纯标点切分模式与说话人识别精度调优（含 CAM++ 中文声纹集成）Spec

> 实施分支：`feat/punct-split-diarization-tuning`（基于 dev @ c2f0355 新建，便于整体回滚）
> 前置：`punctuation-aware-segmentation` spec（标点感知切分 v3，已完成并随 `cu128-punct` 镜像上线）

## Why

`cu128-punct` 镜像实测暴露两个问题：

1. **一句话仍被拆成两段**：punct 版切分是三维混合（句末标点 + 无标点静音间隙 ≥ 2.0s + word 模式说话人变化）。实测中句中长停顿（≥ 2.0s）与 diarizer 边界抖动导致的说话人变化仍会把一句话从中间裂开。用户要求：**不看间隔，只按标点符号划分 segment**。
2. **两个男性对话（东北话）被识别成一个 speaker**：根因是声纹向量化模型域不匹配——现用 WeSpeaker ResNet34 由英文 VoxCeleb 训练，中文男声（同性别 + 方言语调接近）在其向量空间距离过小，聚类阶段欠分割（合成一类）。这不是噪声问题，也不是聚类参数能根治的问题。

## What Changes

**问题 1（切分）**：

- 新增 `--segment-split-mode {punctuation, hybrid}` 服务参数，**默认 `punctuation`**：只按句末标点硬边界 + 段长上限（30s 兜底）切分；静音间隙与句中说话人变化**不再触发切分**；同人二次聚合在该模式下跳过。`hybrid` 为上一代行为，作为一键回退路径。

**问题 2（说话人识别）——双层方案：CAM++ 根治 + 参数立即缓解**：

- **根治层：CAM++ 中文声纹 embedding 集成**。新增 `--diarizer-embedding {wespeaker, campplus}`（默认 `wespeaker`，A/B 验证通过后切默认）与 `--diarizer-embedding-model <dir>`：campplus 模式下声纹向量化组件替换为 CAM++（`speech_campplus_sv_zh-cn_16k-common`，约 200k 中文说话人训练，Apache-2.0，CN-Celeb EER 4.32%），聚类切换为 **3.1 式 AHC + 余弦**（无 PLDA 依赖——PLDA 重估仅在保留 community-1 VBx 聚类链时必须，改 AHC 路径后完全绕开，无需标注语料）。运行时一键回退 = `--diarizer-embedding wespeaker`。
- **缓解层：聚类约束参数**。新增 `--diarization-min-speakers` / `--diarization-max-speakers` 服务级默认（人数已知场景确定性修复，CAM++ 模式下同样生效）与 `--diarization-clustering-threshold` 聚类阈值覆写（人数不定场景调参缓解；AHC 路径下为管线原生超参，机制更直接）。
- `build_segment_response` 新增 `segment_split_mode` 参数（默认 `"punctuation"`）——**默认行为变更**：直接调用方默认不再按间隙/说话人切分。
- **顺手修复遗留 ❶（跨失败块段文本重复）**：`_sentence_end_boundaries` 中 `_gap_blocked(...)` 为 True 的边界强制 `boundaries[i] = True`（约 3 行，与 v3 puncts 置空逻辑互补）。punctuation 模式下该问题触发面从"无标点且间隙 < 2.0s"扩大为"无标点"（间隙维度关闭），故必须一并修复；hybrid 模式同样受益。
- 新增 `docker/Dockerfile-qwen3-asr-punct2`（FROM `qwen3-asr-offline:cu128-punct` 全量覆盖代码 + 构建期双重校验），产出新生产镜像 `qwen3-asr-offline:cu128-punct2`（CAM++ 模型为外部挂载资产，不打进镜像）。
- 更新 `docs/deployment-guide.md`。

### 非目标（明确不做）

- 遗留 ❷（音频以失败块结尾时，末段尾部句末标点追加与粗段尾部标点重复）：触发面极窄，不受本 spec 影响，保持遗留记录。
- 遗留 ❸（交界标点丢失）：跨失败块边界带句末标点时（成功块尾部标点位于失败块文本之前），v3 的 puncts 整体置空使该标点既不附前段、也不在粗段原文中——"严格拼接无损"在该场景不成立。v3 遗留行为，触发面窄，本 spec 仅记录不修复（遗留 ❶ 回归断言的 fixture 须避开交界标点）。

- **community-1 + VBx + CAM++ 的 PLDA 重估升级路径**：需申请 CN-Celeb 类标注语料 + 重估计算。**触发条件**：本 spec AHC 路径 A/B 实测说话人合并率改善不达标时再立项（届时 wrapper 与集成代码可复用）。
- 聚类 `min_cluster_size` 等其他 pyannote 超参数暴露：暂不扩展。
- 响应 JSON 结构变更：`segments[]` 字段名/类型零变化。

## Impact

- Affected specs: `punctuation-aware-segmentation`（切分维度需求被模式参数细化）、`add-segment-speaker-api`（响应语义不变）
- Affected code:
  - `qwen_asr/service/pipeline.py`：`build_segment_response` 新增参数 + 两模式接线 + self_test
  - `qwen_asr/cli/serve.py`：6 个新启动参数（split-mode / min·max-speakers / clustering-threshold / embedding / embedding-model）
  - `qwen_asr/service/extensions.py`：`ExtensionState` 新字段 + 加载/校验/日志 + `_load_diarizer` 扩展
  - `qwen_asr/service/middleware.py`：说话人数请求级→服务级回退 + `segment_split_mode` 透传
  - `qwen_asr/inference/qwen3_speaker_diarizer.py`：聚类阈值应用 + campplus 管线构建与 embedding 注入
  - `qwen_asr/inference/campplus_speaker_embedding.py`：**新增**（vendor 3D-Speaker CAMPPlus 模型定义 + fbank80 特征 + 批量推理封装）
  - `docker/Dockerfile-qwen3-asr-punct2`（新增）、`docs/deployment-guide.md`
- 部署资产：目标机新增 CAM++ 模型目录挂载（约 30MB）；segmentation-3.0 权重可得性需部署机确认（community-1 目录内复用或补充下载）

---

## ADDED Requirements

### Requirement: segment 切分维度模式（segment-split-mode）

系统 SHALL 提供 `--segment-split-mode {punctuation, hybrid}` 启动参数，默认 `punctuation`，控制 segment 切分的维度集合：

- **punctuation 模式（新默认）**：切分仅由①句末标点硬边界（`。！？；.!?;` 及换行，沿用既有规则与标点附前段/末段尾部追加逻辑）与②段长上限（`max_segment_seconds`，默认 30s 强切）驱动。静音间隙阈值视为无穷大（完全不切）；word 模式下词级说话人变化不触发切分；同人二次聚合（`_merge_same_speaker`）跳过。
- **hybrid 模式**：上一代行为——句末标点 + 无标点间隙 ≥ `segment_gap_threshold`（默认 2.0s）+ word 模式说话人变化切分 + 同人二次聚合。
- 模式仅在 `--punctuation-split on` 时生效；`off` 时 mode 被忽略（纯间隙/段长/说话人变化行为），启动时输出告警日志。

`build_segment_response` 新增 `segment_split_mode: str = "punctuation"` 参数，`punctuation_only = punctuation_split and segment_split_mode == "punctuation"`：

- word 模式：`punctuation_only` 时 `groups = _split_groups(pairs, gap_threshold_eff, max_segment_seconds, hard_boundaries)`（`gap_threshold_eff = float("inf")`），**跳过** `_split_by_speaker` 与 `_merge_same_speaker`；词级归属仍照常计算——段 `speaker` 为段内词归属投票（`_word_vote`）的 dominant，`speakers` 为段内出现过的说话人去重集合。
- segment 模式：`_split_groups` 传 `gap_threshold_eff`。
- 段文本游标截取、标点附前段、末段尾部追加、粗段兜底与混合排序：两模式共用，逻辑零改动；跨失败块边界处理见 MODIFIED「跨失败块边界处理」（puncts 置空 + 强制切分）。

#### Scenario: 一句话中间长停顿不再拆段（用户问题 1 回归）

- **WHEN** 相邻词无句末标点且静音间隙 3.0s（> 2.0s 阈值），默认模式（punctuation）
- **THEN** 不切分，两词同属一段（hybrid 模式下会切为两段）

#### Scenario: 句末标点切分保留

- **WHEN** 相邻词 between-span 含句末标点（快问快答"说号就行。啊？"）
- **THEN** 恒切分两段，句号/问号附前段与末段尾部（沿用 punct 版既有规则，两种模式行为一致）

#### Scenario: 句中说话人变化不拆段

- **WHEN** word 模式下段内词归属从 SPEAKER_00 变为 SPEAKER_01 且无句末标点，默认模式
- **THEN** 不切分；该段 `speaker` = 词归属票数多者（dominant），`speakers` = ["SPEAKER_00", "SPEAKER_01"]（按票数降序）

#### Scenario: 段长上限兜底强切

- **WHEN** 无标点连续语音 span > 30s
- **THEN** 按 `max_segment_seconds` 强切（punctuation 模式下唯一的非标点切分来源）

#### Scenario: 整体匹配失败回退退化（trade-off，文档必须写明）

- **WHEN** `_sentence_end_boundaries` 整体匹配失败（如英文缩写规范化不一致 U.S.A.→USA、Mr. 等），punctuation 模式
- **THEN** 边界全 False + 间隙维度关闭 + 说话人切分跳过 → 仅剩 30s 段长强切，产出约 30s 均匀粗块；hybrid 模式下仍有间隙 2.0s 与说话人变化两维度兜底。部署手册须写明：英文/缩写规范化差异密集的内容建议显式 `--segment-split-mode hybrid`

#### Scenario: hybrid 一键回退上一代行为

- **WHEN** 启动传 `--segment-split-mode hybrid`
- **THEN** 切分行为与 `cu128-punct` 镜像完全一致

#### Scenario: punctuation-split off 组合告警

- **WHEN** 启动传 `--punctuation-split off`（mode 保持默认 punctuation）
- **THEN** mode 被忽略，行为为纯间隙/段长/说话人变化；启动日志输出 WARNING

#### Scenario: segment 归属模式同样生效

- **WHEN** `speaker_attribution=segment` + 默认 punctuation 模式，无标点间隙 3.0s
- **THEN** 不因间隙切分；段级重叠投票归属照常

### Requirement: CAM++ 中文声纹 embedding 集成（--diarizer-embedding）

系统 SHALL 提供 `--diarizer-embedding {wespeaker, campplus}`（默认 `wespeaker`，A/B 实测通过后将默认切为 `campplus`）与 `--diarizer-embedding-model <dir>`（CAM++ 模型目录，campplus 模式必填）启动参数：

- **campplus 模式**：diarization 管线的声纹向量化组件替换为 CAM++（`speech_campplus_sv_zh-cn_16k-common`，192 维输出，约 200k 中文说话人训练）；聚类切换为 **pyannote 3.1 式 AHC + 余弦相似度**（项目已兼容 legacy 3.1 配置），**不涉及 PLDA**（community-1 的 VBx+PLDA 链整体不用于该模式）。segmentation（VAD/重叠检测）沿用 segmentation-3.0 权重。
- **wrapper 实现**：新增 `qwen_asr/inference/campplus_speaker_embedding.py`——vendor 3D-Speaker CAMPPlus 模型定义（Apache-2.0，文件头保留原作者署名与来源链接）+ `torchaudio` fbank80 特征（对齐 3D-Speaker 官方推理配方）+ CMVN + 批量推理；接口对齐 pyannote embedding 组件（批量波形窗口 `[B, T]` → `[B, 192]`），支持 `.to(device)`。
- **注入机制**（pyannote 4.0.7 实测确认后固化，按优先级尝试）：a) 程序化构建 3.1 式 `SpeakerDiarization` 管线并注入自定义 embedding 组件；b) 管线加载后替换 embedding 属性；c) CAM++ 导出 ONNX + `wespeaker` 文件名路由（pyannote 按路径名推断 wrapper 的既有机制）。**三条机制的适用语境均为 3.1 式 AHC 管线内的 embedding 加载子机制**——AHC + 余弦聚类不关心 embedding 维度；**不得**理解为"在 community-1（VBx+PLDA）管线内原位替换 embedding 文件"：PLDA 变换绑定 WeSpeaker 256 维向量空间，CAM++ 输出 192 维，维度不匹配必然失败（fail fast 会兜住，但部署机验证会走弯路），该路径明确排除在尝试范围外。部署机验证任务确认实际可用机制，不得静默假设。
- **fail fast 语义**：campplus 模式下模型文件缺失、权重加载失败或 embedding 注入失败 → 启动 RuntimeError（中文消息含模型目录、期望文件清单与回退参数 `--diarizer-embedding wespeaker` 提示）。**不静默回退**——静默回退会让"两人合一"问题看起来已修而实际未修。
- **wespeaker 模式**（默认，现状）：行为与 `cu128-punct` 完全一致（community-1 + WeSpeaker + VBx），零变化。
- **参数组合语义**（比照 `--punctuation-split off` 的告警先例，不阻断启动）：`--diarizer-embedding campplus` + diarizer 显式禁用（`--diarizer ""`）→ embedding 相关参数无效果，启动 WARNING；`--diarizer-embedding wespeaker`（或默认）+ 传了 `--diarizer-embedding-model` → 模型目录被忽略，启动 WARNING。
- 说话人数约束（min/max_speakers）与聚类阈值参数在 campplus 模式下同样生效（3.1 式管线原生支持约束透传与超参设置）。

#### Scenario: 两男东北话合并率下降（核心验证，部署机 A/B）

- **WHEN** campplus 模式 + 真实东北话两男对话音频，与 wespeaker 模式（community-1）A/B 对比
- **THEN** 两说话人被合并为一个 speaker 的比例显著下降；A/B 结论记入部署手册。**若改善不达标**：保持 wespeaker 默认，记录结论并触发 VBx+PLDA 升级路径立项评估（非本 spec 范围）

#### Scenario: 一键回退现状

- **WHEN** 启动传 `--diarizer-embedding wespeaker`（或不传任何 embedding 参数）
- **THEN** diarization 行为与 `cu128-punct` 镜像完全一致

#### Scenario: 模型缺失启动失败（fail fast）

- **WHEN** `--diarizer-embedding campplus` 且模型目录缺失/权重损坏/注入失败
- **THEN** 启动即 RuntimeError（不进入服务循环，不静默回退）

#### Scenario: 约束与阈值参数兼容

- **WHEN** campplus 模式 + `--diarization-min-speakers 2` / `--diarization-clustering-threshold 0.6`
- **THEN** 说话人数约束透传 AHC 聚类；阈值覆写生效（3.1 式管线超参机制）

#### Scenario: diarizer 禁用时 embedding 参数告警

- **WHEN** `--diarizer-embedding campplus --diarizer ""`（diarizer 显式禁用）
- **THEN** diarization 整体关闭（现状语义不变），embedding 相关参数无效果；启动日志输出 WARNING 说明该组合下 embedding 参数被忽略

#### Scenario: wespeaker 模式下 embedding-model 参数告警

- **WHEN** `--diarizer-embedding wespeaker --diarizer-embedding-model /path/to/campplus`
- **THEN** 按 wespeaker（community-1）管线正常运行，模型目录参数被忽略；启动日志输出 WARNING 说明该参数仅 campplus 模式生效

### Requirement: 说话人数约束服务级默认

系统 SHALL 提供 `--diarization-min-speakers` / `--diarization-max-speakers`（int，默认 None 不约束）服务级默认，请求级 form 参数 `min_speakers` / `max_speakers` **逐参数优先**：请求级未传时回退服务级默认。合并后的生效值执行既有校验（min > max → 400）。启动时校验服务级自身组合（min > max 或 < 1 → 启动失败）。两种 embedding 模式下均生效。

#### Scenario: 固定人数对话的确定性修复（东北话两男场景）

- **WHEN** 启动传 `--diarization-min-speakers 2 --diarization-max-speakers 2`，请求不传 form 参数
- **THEN** 每次请求聚类强制输出 2 类（叠加 CAM++ 后为"向量区分 + 数量硬约束"双保险）

#### Scenario: 请求级覆盖服务级

- **WHEN** 服务级 min=2/max=2，请求显式传 `min_speakers=3`
- **THEN** 生效 (3, 2)，min > max → 400 错误（消息提示值来源）

#### Scenario: 不设置时行为不变

- **WHEN** 两参数均未设置
- **THEN** 透传 None，聚类自动决定类数（现状零变化）

### Requirement: 聚类阈值服务级覆写

系统 SHALL 提供 `--diarization-clustering-threshold`（float，默认 None = 管线默认阈值；argparse 校验 0 < t < 2），在 diarizer 管线加载后 best-effort 应用（防御式多候选探测：`instantiate` 超参 / parameters 覆写等；AHC 路径下 `instantiate` 为预期主机制）。全部机制不可用 → WARNING 后正常启动。方向：**调低 → 更倾向拆分说话人**（缓解音色接近合并；过度调低会过分割一人成多），help 与文档写明。

**默认值纪律**：管线默认阈值的具体数值**本机不可验证**（venv 无 pyannote，模型在部署机）——help 文本与文档不得断言未经验证的具体数值（如 0.6），须表述为"模型默认（具体值以部署机 config.yaml 为准）"；部署机验证任务确认两条聚类路径（community-1/VBx 与 AHC）的实际默认值后回填文档。

#### Scenario: 人数不定场景调低阈值缓解合并

- **WHEN** `--diarization-clustering-threshold` 取值低于该管线实际默认值（默认值因聚类路径而异，由部署机验证任务确认），音色接近两人对话
- **THEN** 阈值生效（日志确认），合并概率下降；文档写明 trade-off 与调参阶梯

#### Scenario: 应用机制不可用时降级

- **WHEN** 部署机管线不支持任一探测机制
- **THEN** WARNING + 正常启动，用默认阈值；结论记录

#### Scenario: 非法值启动失败

- **WHEN** 传 0 或 2.5
- **THEN** argparse 中文报错（含合法区间）

### Requirement: punct2 镜像与部署物

新增 `docker/Dockerfile-qwen3-asr-punct2`：`FROM qwen3-asr-offline:cu128-punct`，删旧包目录后全量复制本地 `qwen_asr/`（含 campplus wrapper），构建期双重校验（`compileall` + `pipeline.self_test()`）。CAM++ 模型与 segmentation-3.0 权重为**外部挂载资产**（离线部署模式一致），不打进镜像。

#### Scenario: 轻量构建与构建期防错

- **WHEN** `docker build -f docker/Dockerfile-qwen3-asr-punct2 -t qwen3-asr-offline:cu128-punct2 .`
- **THEN** 秒级完成；语法/分段逻辑异常在 build 阶段失败

#### Scenario: 容器内验证

- **WHEN** 容器内运行 self_test 与 CLI `--help`
- **THEN** self_test ok；6 个新参数全部可见

#### Scenario: 离线部署资产

- **WHEN** 目标机以 campplus 模式启动
- **THEN** 仅依赖本地挂载目录（CAM++ 模型目录 + 既有模型），`HF_HUB_OFFLINE=1` 下零网络请求

---

## MODIFIED Requirements

### Requirement: segment 切分默认行为（源自 punctuation-aware-segmentation spec）

原需求：切分维度 = 句末标点硬边界 + 无标点间隙 ≥ 2.0s（`segment_gap_threshold`）+ word 模式说话人变化，`--punctuation-split on/off` 控制。

修改后：切分维度由 `--segment-split-mode` 控制——默认 `punctuation`（仅标点 + 段长兜底）；`hybrid` 保持原三维混合行为（原需求语义整体移入 hybrid）。**`--segment-gap-threshold` 与 `--speaker-merge-gap` 均仅在 hybrid 模式下生效**——punctuation 模式下前者因间隙维度关闭、后者因 `_merge_same_speaker` 整体跳过而成为**无操作参数**（部署手册参数表须两者一并标注"仅 hybrid 生效"，避免用户调参无效产生困惑）。`--punctuation-split off` 语义不变（此时 mode 被忽略并告警）。

**行为变更提示**：`build_segment_response` 签名默认 `segment_split_mode="punctuation"`，依赖间隙/说话人切分的调用方需显式传 `hybrid`（self_test 既有断言按此适配）。

### Requirement: 跨失败块边界处理（源自 punctuation-aware-segmentation spec v3）

原需求：跨失败块边界（边界时间间隙区间与任一 coarse 块区间相交，`_gap_blocked` 判定）→ `puncts[i]` 置空（避免垃圾后缀），`boundaries[i]` 保持原判定（是否切分取决于 between-span 是否含句末标点）。

修改后：`_gap_blocked(...)` 为 True 的边界**强制 `boundaries[i] = True`（跨失败块边界恒切分）**，`puncts[i]` 置空逻辑不变。修复遗留 ❶：无句末标点 + 跨失败块时，正常段横跨失败块，`_extract_segment_text` 游标截取 `full_text[首匹配起点:末匹配终点]` 会把 between-span 中的失败块 ASR 文本截入段文本——与粗段 `text` 重复且 `"".join(segments[].text)` 拼接有损。该问题在 hybrid 模式下触发条件为"无标点且间隙 < 2.0s"（间隙 ≥ 2.0s 时切分可避免横跨）；punctuation 模式下间隙维度关闭，触发面扩大为"无标点"即触发，故随本 spec 一并修复。生效范围：`punctuation_split=on` 时的两种模式（punctuation / hybrid）；`off` 路径不经过 `_sentence_end_boundaries`，保持旧行为不变。

#### Scenario: 无标点跨失败块恒切分（遗留 ❶ 修复）

- **WHEN** 相邻词 between-span 无句末标点（回归 fixture 须按此设计，避开遗留 ❸ 的交界标点丢失场景），词间隙区间与 coarse 块区间相交（含间隙 < 2.0s 的情形），punctuation 或 hybrid 模式（`punctuation_split=on`）
- **THEN** 恒切分：正常段不横跨失败块，段文本不含失败块 ASR 文本（**失败块文本仅计入粗段一次，不重复计入**）；交界处无句末标点时 `"".join(segments[].text)` 与 `text` 一致

#### Scenario: 含标点跨失败块行为与 v3 一致

- **WHEN** between-span 含句末标点且跨失败块
- **THEN** 切分照常（原已 True），`puncts` 置空不追加（v3 行为，回归不变）

### Requirement: diarization 声纹模型（隐含自 add-segment-speaker-api）

原需求（隐含）：diarization 固定使用 pyannote community-1 管线（WeSpeaker 声纹 + VBx 聚类）。

修改后：声纹模型由 `--diarizer-embedding` 选择——`wespeaker`（默认，原行为）/ `campplus`（中文域 CAM++ + AHC 聚类）。diarization 输出接口（`DiarizationResult` / segments 结构 / 中间件消费方式）零变化。

## REMOVED Requirements

无。
