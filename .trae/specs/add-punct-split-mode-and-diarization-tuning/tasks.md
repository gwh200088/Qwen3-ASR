# Tasks

> 实施分支：`feat/punct-split-diarization-tuning`（已基于 dev @ c2f0355 创建）

- [ ] Task 1: `pipeline.py` 纯标点切分模式（segment_split_mode）
  - [ ] SubTask 1.1: `build_segment_response` 新增 `segment_split_mode: str = "punctuation"` 参数；`punctuation_only = bool(punctuation_split) and segment_split_mode == "punctuation"`；`gap_threshold_eff = float("inf") if punctuation_only else float(segment_gap_threshold)`
  - [ ] SubTask 1.2: word 模式接线：`punctuation_only` 时 `groups = _split_groups(pairs, gap_threshold_eff, max_segment_seconds, hard_boundaries)`，**跳过** `_split_by_speaker` 与 `_merge_same_speaker`；词级归属 `_attribute_words` + `_fill_gaps` 照常，段 `speaker`/`speakers` 由 `_word_vote` 产出
  - [ ] SubTask 1.3: segment 模式接线：`_split_groups(items, gap_threshold_eff, max_segment_seconds, hard_boundaries)`
  - [ ] SubTask 1.4: docstring 更新：模式语义、`segment_gap_threshold` 与 `speaker_merge_gap` 均仅 hybrid 生效、punctuation 模式下 `speaker` 为段内 dominant
  - [ ] SubTask 1.5: `_sentence_end_boundaries` 跨失败块边界强制切分：`_gap_blocked(...)` 为 True 时强制 `boundaries[i] = True`（puncts 置空逻辑不变，约 3 行）——修复遗留 ❶（无标点跨失败块时段文本 `full_text[首:末]` 截入失败块文本、与粗段重复且拼接有损；punctuation 模式触发面扩大为"无标点"即触发，hybrid 的"无标点且间隙 < 2.0s"场景同样受益）；函数 docstring 同步更新；确认既有 helper 级断言（跨失败块置空两用例，boundaries 原判定已为 True）与 build 级粗段用例（切分原已由间隙/标点触发）不受影响
- [ ] Task 2: `serve.py` 参数与 `extensions.py` 状态/校验/日志（切分与约束部分）
  - [ ] SubTask 2.1: 新增 `--segment-split-mode {punctuation, hybrid}`（默认 `punctuation`）
  - [ ] SubTask 2.2: 新增 `--diarization-min-speakers` / `--diarization-max-speakers`（int，默认 None）
  - [ ] SubTask 2.3: 新增 `--diarization-clustering-threshold`（float，默认 None，argparse 校验 `0 < t < 2`，中文报错；help 写明调参方向）
  - [ ] SubTask 2.4: `ExtensionState` 新增 `segment_split_mode` / `diarization_min_speakers` / `diarization_max_speakers` / `diarization_clustering_threshold` 字段与 docstring
  - [ ] SubTask 2.5: `load_extensions`：说话人数交叉校验（min > max 或 < 1 → RuntimeError）；`punctuation_split=off` + mode=punctuation 时 WARNING；调优参数 INFO 汇总日志
- [ ] Task 3: `qwen3_speaker_diarizer.py` 聚类阈值防御式应用
  - [ ] SubTask 3.1: `from_pretrained` 新增 `clustering_threshold: Optional[float] = None`（管线加载后应用）
  - [ ] SubTask 3.2: `_apply_clustering_threshold` 防御式多候选探测（`instantiate` 超参 / parameters 覆写等；首个成功生效 + INFO 日志，全部失败 WARNING 放行）；docstring 记录机制清单与部署机验证要求
- [ ] Task 4: `middleware.py` 说话人数回退与模式透传
  - [ ] SubTask 4.1: 请求级 `min_speakers`/`max_speakers` 为 None 时逐参数回退服务级默认；合并后 min > max 校验（400 错误消息区分值来源）
  - [ ] SubTask 4.2: `build_segment_response` 调用点透传 `segment_split_mode`
- [ ] Task 5: `pipeline.py` self_test 全量更新（无 GPU 依赖）
  - [ ] SubTask 5.1: 新增 punctuation 模式断言组（长停顿不拆 / 说话人变化不拆+dominant / 段长兜底 / match 失败仅段长 / 粗段不变 / off 时 mode 无效 / segment 模式 / hybrid 对照各一条）+ 跨失败块强制切分断言：无标点（fixture 交界处无句末标点，避开遗留 ❸）、间隙 < 2.0s、跨 coarse 块 → punctuation 与 hybrid 两模式均切分、段文本不含失败块文本（失败块文本仅计入粗段一次）、交界无标点下拼接一致（遗留 ❶ 回归）
  - [ ] SubTask 5.2: 既有断言适配（依赖间隙/说话人切分的用例显式传 `segment_split_mode="hybrid"`：组 2 阈值用例、组 5 gap 0.8 用例、组 10 全部、组 11 coarse 用例、组 15 长静音切分、组 16 聚合用例、组 17 match 失败与说话人变化切分用例）
  - [ ] SubTask 5.3: 本地全量运行通过（importlib 按路径加载执行 `self_test()`）
- [ ] Task 6: CAM++ 模型资产获取（构建机，联网）
  - [ ] SubTask 6.1: 从 ModelScope 下载 `speech_campplus_sv_zh-cn_16k-common`（PyTorch 权重 + 配置 + 示例音频 `speaker1_a/b`、`speaker2_a`），记录实际文件清单与 embedding 维度（预期 192，以配置为准）
  - [ ] SubTask 6.2: 确认 3D-Speaker CAMPPlus 模型定义源文件（Apache-2.0，`speakerlab/models/campplus/DTDNN.py`）与推理配方（fbank80 参数：帧长/帧移/CMVN），作为 vendor 与特征实现依据
- [ ] Task 7: CAM++ embedding wrapper（`qwen_asr/inference/campplus_speaker_embedding.py` 新增）
  - [ ] SubTask 7.1: vendor CAMPPlus 模型定义（文件头保留 Apache-2.0 声明、原作者与 3D-Speaker/ModelScope 来源链接）+ `from_pretrained(model_dir)` 加载权重
  - [ ] SubTask 7.2: `CampplusSpeakerEmbedding` 封装：`torchaudio` kaldi fbank80（对齐 3D-Speaker 配方）+ CMVN + 批量推理 `[B, T] → [B, 192]`；`.to(device)` 兼容管线设备搬移；CPU/GPU 均可运行
  - [ ] SubTask 7.3: 模块级 `self_test()`：随机张量前向形状断言 + 示例音频同人/异人余弦相似度断言（同人 > 异人，参考阈值 0.31）；纯离线可执行
- [ ] Task 8: diarizer CAM++ 集成（`qwen3_speaker_diarizer.py` + `serve.py` + `extensions.py`）
  - [ ] SubTask 8.1: `serve.py` 新增 `--diarizer-embedding {wespeaker, campplus}`（默认 `wespeaker`）与 `--diarizer-embedding-model <dir>`（campplus 模式必填，缺失即启动报错）
  - [ ] SubTask 8.2: `SpeakerDiarizer.from_pretrained` 新增 embedding 模式参数：campplus 时构建 3.1 式 AHC 管线（segmentation-3.0 + 自定义 embedding + AgglomerativeClustering）；注入机制按优先级实现：a) 程序化构建注入自定义组件 b) 加载后属性替换 c) ONNX + `wespeaker` 文件名路由——代码留部署机确认日志（实际生效机制 INFO 输出）
  - [ ] SubTask 8.3: fail fast：模型缺失/权重加载失败/注入失败 → 中文 RuntimeError（含目录、期望文件、`--diarizer-embedding wespeaker` 回退提示），不静默回退
  - [ ] SubTask 8.4: `extensions._load_diarizer` 透传 embedding 参数；`ExtensionState` 新字段；启动日志输出 embedding 模型/维度/聚类路径
  - [ ] SubTask 8.5: 说话人数约束与聚类阈值在 campplus 管线（3.1 式）下透传验证代码路径打通（`diarize` 签名过滤复用既有逻辑）
  - [ ] SubTask 8.6: 参数组合 WARNING（`load_extensions` 内，比照 punctuation-split off 先例，不阻断启动）：`--diarizer-embedding campplus` + diarizer 禁用（`--diarizer ""`）→ WARNING（embedding 参数无效果）；wespeaker 模式 + 传 `--diarizer-embedding-model` → 忽略 + WARNING（仅 campplus 模式生效）
- [ ] Task 9: punct2 镜像构建与本地验证
  - [ ] SubTask 9.1: 新增 `docker/Dockerfile-qwen3-asr-punct2`：FROM `qwen3-asr-offline:cu128-punct` + 删旧包目录全量 COPY `qwen_asr/`（含 campplus wrapper）+ `compileall` + `pipeline.self_test()` 构建期校验
  - [ ] SubTask 9.2: 执行构建；容器内验证 self_test + CLI `--help` 可见全部新参数（脚本文件挂载方式执行，规避 PowerShell 引号转义）
  - [ ] SubTask 9.3: 容器内 wrapper 离线自测（campplus self_test，若构建机含模型目录则挂载验证）
- [ ] Task 10: 部署机集成验证与阈值校准（需部署机，用户协助）
  - [ ] SubTask 10.1: pyannote 4.0.7 下确认 embedding 注入机制实际生效（按优先级 a/b/c 逐一验证，选定固化）；确认 segmentation-3.0 权重可得性（community-1 挂载目录内复用 or 补充下载挂载）
  - [ ] SubTask 10.2: 聚类阈值机制确认与初始校准：两条聚类路径（wespeaker/community-1 VBx 与 campplus/AHC）的**实际默认阈值**从部署机 config.yaml 确认并记录（help 与文档以实测值回填，不预先断言 0.6——本机无 pyannote 不可验证）；示例音频同人/异人余弦分布参考；`--diarization-clustering-threshold` 覆写生效验证
  - [ ] SubTask 10.3: min/max_speakers 在 campplus 管线下透传验证（min=max=2 实测强制 2 类）
- [ ] Task 11: A/B 实测与默认值决策（需部署机 + 真实音频）
  - [ ] SubTask 11.1: 东北话两男真实音频 A/B：wespeaker（community-1+VBx）vs campplus（AHC），对比说话人合并率与段边界质量；结论记录
  - [ ] SubTask 11.2: 改善达标 → `--diarizer-embedding` 默认值切 `campplus` 重建镜像；不达标 → 保持 `wespeaker` 默认 + 记录结论 + VBx+PLDA 升级路径立项评估建议
- [ ] Task 12: `docs/deployment-guide.md` 更新
  - [ ] SubTask 12.1: 推荐镜像 → `cu128-punct2`（punct 降为上一代）；镜像版本表 + punct2 改动章节（含构建命令与目标机轻量构建）
  - [ ] SubTask 12.2: 新参数文档：split-mode 两模式与 hybrid 回退组合（含 match 失败回退 trade-off：英文/缩写规范化差异密集内容建议显式 hybrid，症状为约 30s 均匀粗块）、说话人数约束（固定人数 min=max / 请求级优先）、聚类阈值方向与调参阶梯（默认值以部署机 config.yaml 实测为准回填）、`--diarizer-embedding`/`--diarizer-embedding-model`（CAM++ 原理、模型下载与挂载、一键回退、参数组合告警语义：diarizer 禁用时无效 / wespeaker 下 model 目录忽略）；参数表将 `--segment-gap-threshold` 与 `--speaker-merge-gap` **一并**标注"仅 hybrid 模式生效"（punctuation 模式下均为无操作参数）
  - [ ] SubTask 12.3: §5.1 导入镜像 / §5.3 冒烟验证 / §6.1 标准启动 / 运维"更新代码"等章节镜像名与启动示例更新（含固定双人 + campplus 启动示例）
  - [ ] SubTask 12.4: 故障排查更新（说话人合一步骤 → embedding 模式 + 约束参数；分段问题 → punct2 默认纯标点切分）；回滚说明（参数级回退 hybrid/wespeaker；镜像级回退 punct）；CAM++ A/B 结论章节（Task 11 结论回填）

# Task Dependencies

- Task 2 depends on Task 1；Task 3 depends on Task 2；Task 4 depends on Task 1, 2
- Task 5 depends on Task 1–4
- Task 6 独立（可与 Task 1–5 并行，需联网）
- Task 7 depends on Task 6
- Task 8 depends on Task 3, 7（diarizer 同时承载阈值应用与 embedding 注入）
- Task 9 depends on Task 5, 8（镜像含全部代码）
- Task 10 depends on Task 9（需部署机）
- Task 11 depends on Task 10
- Task 12 depends on Task 2, 9（A/B 结论章节回填依赖 Task 11，其余可先行）
