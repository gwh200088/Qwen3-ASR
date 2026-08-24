# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CAM++ 中文声纹 embedding wrapper（``--diarizer-embedding campplus`` 后端）。

模型 ``speech_campplus_sv_zh-cn_16k-common``（约 200k 中文说话人训练，192 维
输出）：缓解中文男声相近被 pyannote community-1（WeSpeaker + VBx）合并的问题
（spec「CAM++ 中文声纹 embedding 集成」）。

组成：

- **模型定义 vendor 自 3D-Speaker**（Apache-2.0，文件内保留原作者署名与来源
  链接；仅合并 layers.py + DTDNN.py 两文件、去除 speakerlab 包内依赖，结构
  与命名原样保留）；
- **特征**：torchaudio Kaldi fbank 80 mel + CMVN（mean-only），对齐 3D-Speaker
  官方推理配方（``Fbank(MeanNorm)``：waveform × (1 << 15) 后提 fbank 再减
  均值）；
- **接口对齐 pyannote 4.0.7 embedding 组件**（``PyannoteAudioPretrainedSpeaker
  Embedding`` 调用面）：``__call__(waveforms, masks=...) -> np.ndarray [B, 192]``
  （waveforms 兼容 ``[B, T]`` / ``[B, 1, T]``）、``dimension`` / ``sample_rate`` /
  ``metric``（"cosine"）/ ``min_num_samples`` 属性、``to(device)`` / ``eval()`` /
  ``device``——供 SpeakerDiarization 管线组件替换注入使用；
- **fail fast**：模型目录缺失 ``campplus_cn_common.bin`` / ``config.yaml``、
  权重加载或结构不匹配 → 启动 RuntimeError（中文消息含目录、期望文件清单与
  回退参数 ``--diarizer-embedding wespeaker`` 提示），不静默回退。

已知近似：pyannote SpeakerDiarization 前向以 ``masks=`` 传入逐帧说话人活跃
权重（WeSpeaker 用于加权池化）；CAM++ 的 StatsPool 不支持加权，``masks`` 非
None 时忽略（均匀池化），重叠帧占比小、对整体归属影响有限。
"""

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn
from torch.nn.utils.rnn import pad_sequence

try:  # torchaudio 为特征提取依赖（cu128 基础镜像自带）；缺失时延迟到调用报错
    import torchaudio.compliance.kaldi as ta_kaldi
except ImportError:  # pragma: no cover - 仅缺依赖环境触发
    ta_kaldi = None

__all__ = ["CAMPPlus", "CampplusSpeakerEmbedding"]

logger = logging.getLogger(__name__)


# ===========================================================================
# 以下模型定义 vendor 自 3D-Speaker（Apache-2.0），来源：
#   https://github.com/alibaba-damo-academy/3D-Speaker
#     speakerlab/models/campplus/layers.py
#     speakerlab/models/campplus/DTDNN.py
# 仅合并两文件并去除包内 import，类/函数结构与命名原样保留。
# ---------------------------------------------------------------------------
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
# All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# ---------------------------------------------------------------------------
def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(self,
                 bn_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                      out_channels=out_channels,
                                      bn_channels=bn_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      bias=bias,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):
    def __init__(self,
                 block=BasicResBlock,
                 num_blocks=[2, 2],
                 m_channels=32,
                 feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    def __init__(self,
                 feat_dim=80,
                 embedding_size=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True):
        super(CAMPPlus, self).__init__()
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(channels,
                           init_channels,
                           5,
                           stride=2,
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                      in_channels=channels,
                                      out_channels=growth_rate,
                                      bn_channels=bn_size * growth_rate,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2
        self.xvector.add_module(
            'out_nonlinear', get_nonlinear(config_str, channels))
        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x


# ===========================================================================
# wrapper：加载 / 特征 / 推理（接口对齐 pyannote embedding 组件）
# ===========================================================================


def _parse_scalar(value: str) -> Any:
    """标量解析：int / bool / str（config.yaml 值域仅这三类）。"""
    text = str(value).strip()
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        return text


def _load_campplus_config(config_path: str) -> Dict[str, Any]:
    """解析 ModelScope ``config.yaml``（平面两层结构，无 yaml 依赖）。

    格式固定为 ``model`` / ``model_conf`` / ``frontend`` / ``frontend_conf``
    四个顶级键 + 标量值（缩进表示子键）；结构外的内容解析失败即抛
    RuntimeError（fail fast：config 损坏不得静默用默认值顶替）。
    """
    conf: Dict[str, Any] = {}
    section: Optional[str] = None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.rstrip()
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                key, sep, value = line.strip().partition(":")
                if not sep:
                    raise ValueError(f"第 {lineno} 行无法解析: {raw.strip()!r}")
                key = key.strip()
                if indent == 0:
                    value_text = value.strip()
                    if value_text:
                        conf[key] = _parse_scalar(value_text)
                        section = None
                    else:
                        conf[key] = {}
                        section = key
                else:
                    if section is None:
                        raise ValueError(f"第 {lineno} 行缩进键 {key!r} 缺少所属分组")
                    conf[section][key] = _parse_scalar(value)
    except ValueError as exc:
        raise RuntimeError(f"CAM++ config.yaml 解析失败（{config_path}）: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"CAM++ config.yaml 读取失败（{config_path}）: {exc}") from exc
    return conf


class CampplusSpeakerEmbedding:
    """CAM++ 声纹 embedding（pyannote 4.0.7 embedding 组件接口形态）。

    用法（serve 侧 ``--diarizer-embedding campplus`` 路径）::

        embedder = CampplusSpeakerEmbedding.from_pretrained(model_dir)
        embedder.to("cuda:0")
        embeddings = embedder(waveforms)  # [B, 1, T] -> np.ndarray [B, 192]

    pyannote SpeakerDiarization 前向调用面（``PyannoteAudioPretrainedSpeaker
    Embedding`` 对齐）：``__call__(waveforms, masks=...)`` 返回 ``np.ndarray``；
    ``sample_rate`` / ``dimension`` / ``metric`` / ``min_num_samples`` 属性在
    管线构造与前向中被读取。

    Attributes:
        model: vendor 的 ``CAMPPlus`` 模块（eval 模式，fp32）。
        dimension: 输出维度（192，``--diarizer-embedding-model`` 权重决定）。
        sample_rate: 采样率（16000，config.yaml ``frontend_conf.fs``）。
        metric: 距离度量（"cosine"，pyannote AHC 聚类读取）。
        min_num_samples: 前向所需最小采样数（= 1 秒；用于过滤过短的说话人
            活跃掩码，CAM++ 卷积结构无硬性下限，1 秒已覆盖感受野）。
    """

    #: 期望文件清单（spec fail fast 消息与校验共用）
    EXPECTED_FILES = ("campplus_cn_common.bin", "config.yaml")

    def __init__(
        self,
        model: CAMPPlus,
        feat_dim: int = 80,
        embedding_size: int = 192,
        sample_rate: int = 16000,
    ):
        self.model = model
        self.model.eval()
        self.feat_dim = int(feat_dim)
        self.embedding_size = int(embedding_size)
        self.sample_rate = int(sample_rate)
        self._device = torch.device("cpu")

    # -- pyannote 组件接口 ---------------------------------------------------

    @property
    def dimension(self) -> int:
        """输出向量维度（pyannote 管线聚类前读取）。"""
        return self.embedding_size

    @property
    def metric(self) -> str:
        """距离度量（pyannote AgglomerativeClustering 构造读取）。"""
        return "cosine"

    @property
    def min_num_samples(self) -> int:
        """前向所需最小采样数（speaker_diarization 过滤短活跃掩码读取）。"""
        return self.sample_rate

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[str, torch.device]) -> "CampplusSpeakerEmbedding":
        self.model.to(device)
        self._device = torch.device(device)
        return self

    def eval(self) -> "CampplusSpeakerEmbedding":
        self.model.eval()
        return self

    # -- 加载 -----------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "CampplusSpeakerEmbedding":
        """fail fast 加载 CAM++ 模型目录。

        Raises:
            RuntimeError: 目录缺失期望文件（``campplus_cn_common.bin`` /
                ``config.yaml``）、config 结构不是 CAM++、权重加载失败或
                结构不匹配——中文消息含目录、期望文件清单与回退参数提示
                （``--diarizer-embedding wespeaker``），不静默回退。
        """
        model_dir = str(model_dir).strip()
        if not model_dir or not os.path.isdir(model_dir):
            raise RuntimeError(
                f"CAM++ 声纹模型目录不存在: {model_dir!r}（--diarizer-embedding-model "
                "应为 speech_campplus_sv_zh-cn_16k-common 模型目录，期望包含 "
                f"{', '.join(cls.EXPECTED_FILES)}）；回退上一代行为请改用 "
                "--diarizer-embedding wespeaker。"
            )
        bin_path = os.path.join(model_dir, "campplus_cn_common.bin")
        config_path = os.path.join(model_dir, "config.yaml")
        missing = [
            name
            for name, path in (("campplus_cn_common.bin", bin_path), ("config.yaml", config_path))
            if not os.path.isfile(path)
        ]
        if missing:
            raise RuntimeError(
                f"CAM++ 声纹模型目录 {model_dir!r} 缺少文件: {', '.join(missing)}"
                f"（期望文件清单: {', '.join(cls.EXPECTED_FILES)}）；"
                "回退上一代行为请改用 --diarizer-embedding wespeaker。"
            )

        conf = _load_campplus_config(config_path)
        if str(conf.get("model", "")).upper() != "CAMPPLUS":
            raise RuntimeError(
                f"CAM++ config.yaml 的 model 字段为 {conf.get('model')!r}（期望 "
                f"CAMPPlus）：{config_path} 不是 CAM++ 声纹模型配置；回退上一代"
                "行为请改用 --diarizer-embedding wespeaker。"
            )
        model_conf = conf.get("model_conf") or {}
        frontend_conf = conf.get("frontend_conf") or {}
        sample_rate = int(frontend_conf.get("fs", 16000))
        if sample_rate != 16000:
            raise RuntimeError(
                f"CAM++ 模型采样率为 {sample_rate}Hz（本服务音频管线固定 16k 归一化，"
                "仅支持 fs: 16000 的模型）；回退上一代行为请改用 "
                "--diarizer-embedding wespeaker。"
            )
        feat_dim = int(model_conf.get("feat_dim", 80))
        embedding_size = int(model_conf.get("embedding_size", 192))

        model = CAMPPlus(
            feat_dim=feat_dim,
            embedding_size=embedding_size,
            growth_rate=int(model_conf.get("growth_rate", 32)),
            bn_size=int(model_conf.get("bn_size", 4)),
            init_channels=int(model_conf.get("init_channels", 128)),
            config_str=str(model_conf.get("config_str", "batchnorm-relu")),
            memory_efficient=bool(model_conf.get("memory_efficient", True)),
        )
        try:
            state_dict = torch.load(bin_path, map_location="cpu")
        except Exception:
            # torch 2.6+ 默认 weights_only=True 可能拒绝旧格式 pickle：显式放开重试
            try:
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise RuntimeError(
                    f"CAM++ 权重加载失败（{bin_path}）: {exc}；请确认文件完整"
                    "（ModelScope iic/speech_campplus_sv_zh-cn_16k-common 的 "
                    "campplus_cn_common.bin，28MB）；回退上一代行为请改用 "
                    "--diarizer-embedding wespeaker。"
                ) from exc
        # FunASR 产物可能带 "model." 前缀（防御式兼容，两种格式择一匹配）
        if state_dict and all(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"CAM++ 权重与模型结构不匹配（{bin_path}）: {exc}；请确认权重来自 "
                "ModelScope iic/speech_campplus_sv_zh-cn_16k-common；回退上一代"
                "行为请改用 --diarizer-embedding wespeaker。"
            ) from exc
        model.eval()
        logger.info(
            "CAM++ 声纹模型已加载: %s（feat_dim=%d, embedding_size=%d）",
            model_dir,
            feat_dim,
            embedding_size,
        )
        return cls(model, feat_dim=feat_dim, embedding_size=embedding_size, sample_rate=sample_rate)

    # -- 特征与推理 -----------------------------------------------------------

    def _fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        """Kaldi fbank（80 mel）+ CMVN mean-only，对齐 3D-Speaker 推理配方。

        Args:
            waveform: ``[T]`` float32 波形（[-1, 1]，服务侧已 16k 单声道归一化）。

        Returns:
            ``[T', feat_dim]`` 特征矩阵（逐 utterance 减均值）。
        """
        # Kaldi fbank 期望 int16 动态范围：× (1 << 15)（3D-Speaker Fbank 配方）
        scaled = waveform.float().to(self._device) * (1 << 15)
        mat = ta_kaldi.fbank(
            scaled.unsqueeze(0),
            num_mel_bins=self.feat_dim,
            sample_frequency=self.sample_rate,
        )
        return mat - mat.mean(dim=0, keepdim=True)

    def __call__(
        self,
        waveforms: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> "numpy.ndarray":
        """批量声纹提取（pyannote embedding 组件调用形态）。

        Args:
            waveforms: 波形批量，兼容 ``[B, T]``（二维）与 pyannote
                SpeakerDiarization 前向传入的 ``[B, 1, T]``（三维，channel 维
                squeeze）；变长输入按最长右补零，padding 帧计入统计池化为已知
                近似。张量在 ``self.device`` 上执行（调用方负责搬移或传 CPU
                张量时逐条搬移）。
            masks: pyannote overlap-aware 逐帧说话人活跃权重（``[B, T']``）——
                CAM++ 的 StatsPool 不支持加权，非 None 时忽略（见模块
                docstring「已知近似」）。
            weights: ``masks`` 的别名（speaker_verification.SpeakerEmbedding
                单文件管线的关键字），同样忽略。

        Returns:
            ``[B, embedding_size]`` 声纹矩阵（``np.ndarray`` fp32，对齐
            pyannote embedding 组件返回类型——管线侧 ``np.vstack`` 拼接）。
        """
        if ta_kaldi is None:
            raise ImportError(
                "torchaudio is required for CampplusSpeakerEmbedding but not "
                "installed. Install with: pip install torchaudio"
            )
        if waveforms.dim() == 3:
            if waveforms.shape[1] != 1:
                raise ValueError(
                    f"waveforms 三维输入 channel 维须为 1，收到 shape={tuple(waveforms.shape)}"
                )
            waveforms = waveforms[:, 0, :]
        if waveforms.dim() != 2:
            raise ValueError(
                f"waveforms 须为 [B, T] 或 [B, 1, T] 张量，收到 shape={tuple(waveforms.shape)}"
            )
        if masks is not None or weights is not None:
            logger.debug("CAM++ 不支持加权池化，忽略 masks/weights（均匀池化近似）")
        feats = [self._fbank(waveforms[i]) for i in range(waveforms.shape[0])]
        batch = pad_sequence(feats, batch_first=True)  # [B, Tmax, feat_dim]
        with torch.no_grad():
            embeddings = self.model(batch)  # [B, embedding_size]
        return embeddings.detach().cpu().numpy()


# ===========================================================================
# 离线自测（spec Task 7.3：随机张量前向 + 示例音频同人/异人余弦断言）
# ===========================================================================


def self_test(model_dir: str) -> None:
    """CAM++ wrapper 离线自测（容器内构建期/部署冒烟用，纯 CPU 可执行）。

    断言组：

    1. **加载**：``from_pretrained`` 正常加载，``dimension`` == config.yaml
       ``embedding_size``（192）、``sample_rate`` == 16000、``metric`` ==
       "cosine"；
    2. **随机张量前向**：``[B, 1, T]`` 三维输入（pyannote 前向形态）返回
       ``np.ndarray [B, dimension]``；``masks=`` 传入不报错（忽略近似）；
    3. **示例音频同人/异人余弦**：ModelScope ``examples/`` 三条中文示例
       （``speaker1_a_cn_16k.wav`` / ``speaker1_b_cn_16k.wav`` /
       ``speaker2_a_cn_16k.wav``）——同人余弦 > 异人余弦，且同人 >
       0.31（模型卡参考阈值；对纯中文短语音，实测同人普遍 > 0.6）。

    Args:
        model_dir: CAM++ 模型目录（含权重 / config.yaml / examples/ 示例音频，
            ModelScope ``iic/speech_campplus_sv_zh-cn_16k-common`` 全量下载）。

    Raises:
        AssertionError: 任一断言失败（构建期校验中止）。
        RuntimeError: 模型目录/权重缺失（fail fast，中文消息）。
    """
    import numpy as np

    embedder = CampplusSpeakerEmbedding.from_pretrained(model_dir)
    assert embedder.dimension == 192, f"dimension 应为 192，实得 {embedder.dimension}"
    assert embedder.sample_rate == 16000, f"sample_rate 应为 16000，实得 {embedder.sample_rate}"
    assert embedder.metric == "cosine", f"metric 应为 cosine，实得 {embedder.metric}"
    assert embedder.min_num_samples >= 16000, "min_num_samples 应不小于 1 秒采样数"

    # -- 随机张量前向（pyannote [B, 1, T] 形态 + masks 忽略近似）--------------
    torch.manual_seed(0)
    batch = torch.randn(4, 1, 3 * embedder.sample_rate)
    masks = torch.rand(4, 300)  # 逐帧活跃权重（帧数与前向解耦，忽略即可）
    out = embedder(batch, masks=masks)
    assert isinstance(out, np.ndarray), f"返回应为 np.ndarray，实得 {type(out)}"
    assert out.shape == (4, 192), f"输出 shape 应为 (4, 192)，实得 {out.shape}"
    out2d = embedder(torch.randn(2, 2 * embedder.sample_rate))  # 二维兼容
    assert out2d.shape == (2, 192), f"二维输入输出 shape 应为 (2, 192)，实得 {out2d.shape}"

    # -- 示例音频同人/异人余弦相似度 -------------------------------------------
    # soundfile 读 wav（pyannote.audio 依赖，离线容器必有；torchaudio.load 在
    # 无 FFmpeg 共享库的容器内走 torchcodec 会失败，不作为依赖）
    import soundfile as sf

    def _load(name: str) -> torch.Tensor:
        path = os.path.join(model_dir, "examples", name)
        data, sr = sf.read(path, dtype="float32")
        assert sr == embedder.sample_rate, f"{name} 采样率 {sr} != {embedder.sample_rate}"
        if data.ndim > 1:  # 多声道取均值单声道化
            data = data.mean(axis=1)
        return torch.from_numpy(data)

    # 三条示例长度不等，逐条推理（B=1，返回 (1, 192)）后拼接为 (3, 192)
    embs = [embedder(_load(n)[None]) for n in (
        "speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav")]
    emb = np.concatenate(embs, axis=0)
    normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    same = float((normed[0] * normed[1]).sum())    # speaker1_a vs speaker1_b（同人）
    diff = float((normed[0] * normed[2]).sum())    # speaker1_a vs speaker2_a（异人）
    print(f"[campplus self_test] 同人余弦 = {same:.4f}，异人余弦 = {diff:.4f}")
    assert same > diff, f"同人相似度 ({same:.4f}) 应大于异人 ({diff:.4f})"
    assert same > 0.31, f"同人相似度 ({same:.4f}) 应大于模型卡参考阈值 0.31"
    print("[campplus self_test] 全部断言通过（加载/前向形状/同人异人判别）")
