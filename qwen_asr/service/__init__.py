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
qwen_asr.service: qwen-asr-serve 的服务扩展逻辑模块（非独立服务）。

包含 GPU 显存感知调度器、segment 转写管道纯逻辑、扩展状态与加载器，
以及接管 /v1/audio/transcriptions 的 ASGI middleware。
"""

from .extensions import ExtensionState, load_extensions
from .middleware import TranscriptionsMiddleware
from .pipeline import (
    LANGUAGE_CODE_TO_NAME,
    LANGUAGE_NAME_TO_CODE,
    DiarizationTurn,
    build_segment_response,
    language_name_to_code,
    resolve_language_name,
    self_test,
)
from .scheduler import GpuScheduler

__all__ = [
    "GpuScheduler",
    "ExtensionState",
    "load_extensions",
    "TranscriptionsMiddleware",
    "DiarizationTurn",
    "LANGUAGE_NAME_TO_CODE",
    "LANGUAGE_CODE_TO_NAME",
    "resolve_language_name",
    "language_name_to_code",
    "build_segment_response",
    "self_test",
]
