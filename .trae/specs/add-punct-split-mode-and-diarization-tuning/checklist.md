# Checklist

## 纯标点切分模式（pipeline.py）

- [ ] `build_segment_response` 新增 `segment_split_mode` 参数，默认 `"punctuation"`；`punctuation_only = punctuation_split and mode == "punctuation"`
- [ ] punctuation 模式：无标点间隙不切分（3.0s 间隙回归用例通过，hybrid 下对照切分）
- [ ] punctuation 模式：句末标点硬边界切分照常（快问快答/英文句点/标点附前段/末段尾部追加全部沿用）
- [ ] punctuation 模式：word 模式说话人变化不切分；段 `speaker` = `_word_vote` dominant，`speakers` = 段内去重集合
- [ ] punctuation 模式：`_merge_same_speaker` 跳过
- [ ] punctuation 模式：段长 > `max_segment_seconds`（30s）仍强切
- [ ] punctuation 模式：match 失败回退后仅段长切分
- [ ] punctuation 模式：粗段兜底、混合产出按 start 升序、拼接无损均不受影响
- [ ] segment 归属模式：punctuation 模式同样生效
- [ ] hybrid 模式：与 `cu128-punct` 行为完全一致
- [ ] 跨失败块边界强制切分实现（`_gap_blocked` → `boundaries[i]=True`，puncts 置空不变；`punctuation_split=on` 两种模式生效，`off` 路径不变）
- [ ] 无标点跨失败块（间隙 < 2.0s，交界无句末标点 fixture，避开遗留 ❸）回归用例通过：两模式均切分、段文本不含失败块文本（失败块文本仅计入粗段一次）、交界无标点下拼接一致（遗留 ❶ 修复）
- [ ] helper 级既有断言（跨失败块置空两用例）与 build 级粗段用例不受强制边界影响（复核通过）
- [ ] `punctuation_split=False` 时 mode 被忽略，启动 WARNING 日志输出

## 参数与透传（serve.py / extensions.py / middleware.py）

- [ ] `--segment-split-mode {punctuation, hybrid}` 默认 `punctuation`；非法值启动报错
- [ ] `--diarization-min-speakers` / `--diarization-max-speakers` 默认 None；服务级 min > max 或 < 1 → 启动失败（中文 RuntimeError）
- [ ] `--diarization-clustering-threshold` 默认 None；`0 < t < 2` 否则 argparse 中文报错
- [ ] `ExtensionState` 新字段存储；`load_extensions` 透传 + 交叉校验 + off/mode 组合 WARNING + 调优参数 INFO 汇总
- [ ] middleware：请求级说话人数未传时逐参数回退服务级默认；合并后 min > max → 400（消息区分值来源）
- [ ] middleware：`segment_split_mode` 透传至 `build_segment_response`
- [ ] 未设置任何新参数时（全默认），行为 = punctuation 切分 + wespeaker/community-1 diarization（除切分默认外零变化）

## 聚类阈值应用（qwen3_speaker_diarizer.py）

- [ ] `from_pretrained` 新增 `clustering_threshold`，加载后应用
- [ ] 防御式多候选探测：首个可用机制生效 + INFO 日志；全部失败 WARNING 后正常返回
- [ ] `extensions._load_diarizer` 透传 threshold；None 时零额外行为
- [ ] 部署机验证：覆写机制确认生效（config.yaml/日志/实测佐证）；两条聚类路径（community-1/VBx 与 AHC）实际默认阈值确认并回填 help/文档（不预先断言 0.6）；失败如实记录并建后续任务

## CAM++ 中文声纹集成（campplus_speaker_embedding.py / qwen3_speaker_diarizer.py）

- [ ] ModelScope `speech_campplus_sv_zh-cn_16k-common` 模型资产下载完成（权重 + 配置 + 示例音频），文件清单与 embedding 维度记录
- [ ] `campplus_speaker_embedding.py`：vendor CAMPPlus 定义（Apache-2.0 署名保留）+ fbank80/CMVN 对齐 3D-Speaker 推理配方 + `[B,T]→[B,192]` 批量推理 + `.to(device)`
- [ ] wrapper `self_test()` 通过：随机张量形状断言 + 示例音频同人余弦 > 异人（参考 0.31）
- [ ] `--diarizer-embedding {wespeaker, campplus}`（默认 wespeaker）+ `--diarizer-embedding-model <dir>` 参数；campplus 模式模型目录必填
- [ ] campplus 模式：3.1 式 AHC 管线构建 + CAM++ embedding 注入；实际生效机制 INFO 日志
- [ ] fail fast：模型缺失/加载失败/注入失败 → 中文 RuntimeError（含目录、期望文件、回退参数提示），不静默回退
- [ ] wespeaker 模式（默认）：行为与 `cu128-punct` 完全一致（零变化）
- [ ] min/max_speakers 与聚类阈值在 campplus 管线下生效（代码路径 + 部署机验证）
- [ ] 注入机制 (c) 语境约束：仅在 3.1 式 AHC 管线 embedding 加载子机制内实现/验证；不尝试 community-1（VBx+PLDA）管线内原位替换 embedding 文件（256 维 PLDA vs 192 维 CAM++，维度不匹配必败，明确排除）
- [ ] 参数组合 WARNING：campplus + diarizer 禁用（`--diarizer ""`）→ WARNING（embedding 参数无效果）；wespeaker 模式 + 传 `--diarizer-embedding-model` → 忽略 + WARNING（不阻断启动）
- [ ] 部署机验证：注入机制按 a/b/c 优先级确认固化；segmentation-3.0 权重可得性确认（复用 or 补充挂载）
- [ ] 部署机 A/B 实测：东北话两男音频 wespeaker vs campplus 说话人合并率对比，结论记录
- [ ] 默认值决策执行：达标切 campplus 默认并重建镜像 / 不达标保持 wespeaker 并记录 + 升级路径建议

## self_test（pipeline.py）

- [ ] 新增 punctuation 模式断言全部通过
- [ ] 既有断言适配后全部通过（依赖间隙/说话人切分的用例显式传 `hybrid`）
- [ ] 本地全量运行通过（importlib 按路径加载 + `self_test()`）
- [ ] 响应 JSON 结构/字段名/类型零变化

## 镜像与容器验证

- [ ] `docker/Dockerfile-qwen3-asr-punct2` 存在：FROM `cu128-punct` + 全量覆盖 `qwen_asr/`（含 campplus wrapper）+ 构建期校验
- [ ] `qwen3-asr-offline:cu128-punct2` 构建成功；构建期校验通过
- [ ] 容器内 self_test 通过；CLI `--help` 可见全部新参数（split-mode / min·max-speakers / clustering-threshold / embedding / embedding-model）
- [ ] CAM++ 模型为外部挂载资产（不打进镜像）；`HF_HUB_OFFLINE=1` 下 campplus 模式零网络请求

## 文档（deployment-guide.md）

- [ ] 推荐镜像更新为 `cu128-punct2`；镜像版本表与 punct2 改动章节（含轻量构建命令）
- [ ] 全部新参数文档齐全：模式语义与回退组合、说话人数约束、聚类阈值方向与阶梯（默认值以部署机实测回填）、CAM++ 集成（原理/模型下载与挂载/一键回退）
- [ ] match 失败回退 trade-off 文档化（英文/缩写密集内容建议显式 hybrid；症状为约 30s 均匀粗块）
- [ ] 参数表将 `--segment-gap-threshold` 与 `--speaker-merge-gap` 一并标注"仅 hybrid 生效"（punctuation 模式下无操作）
- [ ] 导入/冒烟/启动/运维等章节镜像名与启动示例同步（含固定双人 + campplus 启动示例）
- [ ] 故障排查指向新参数与新镜像；回滚路径写明（参数级 hybrid/wespeaker；镜像级 punct）
- [ ] CAM++ A/B 实测结论回填
