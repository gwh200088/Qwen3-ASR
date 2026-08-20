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
GPU 显存感知任务调度器。

用于管控对齐（aligner）与说话人识别（diarizer）阶段的任务并发与排队，
避免长音频并发任务打爆显存（OOM）。核心机制（见 spec「显存预算方案（按设备）」）：

- 全局单一 FIFO 队列：队首任务一次性检查其涉及的全部 cuda 设备，
  任一设备空闲显存不足则继续等待（队首阻塞换取公平与无死锁）；
- 按设备准入：任务瞬态显存需求按设备拆分（同设备求和），CPU 设备
  不参与显存检查（退化为纯并发数限制）；
- 许可异常安全：``slot()`` 的 yield 处于 try/finally 中，任务任意阶段
  异常或取消均必然释放许可；
- torch 延迟 import：无 torch/CUDA 环境下本模块可正常导入与单测
  （显存查询函数可直接 monkeypatch）。
"""

import asyncio
import collections
import contextlib
import dataclasses
import math
from typing import Any, AsyncIterator, Deque, Dict, Optional

__all__ = [
    "GpuScheduler",
    "normalize_device",
    "device_index",
    "get_device_free_mb",
    "get_device_total_mb",
    "estimate_align_mb",
    "estimate_diar_mb",
    "estimate_task_need_mb",
]


# ---------------------------------------------------------------------------
# 设备工具
# ---------------------------------------------------------------------------


def normalize_device(name: Optional[str]) -> str:
    """规范化设备名。

    接受 ``"cuda"`` / ``"cuda:0"`` / ``"cuda:1"`` / ``"cpu"`` / ``None``：
    - ``None`` 或 ``"cuda"`` 补全序号为 ``"cuda:0"``；
    - ``"cpu"`` 原样返回；
    - 其余字符串去除首尾空白并转小写后原样返回。
    """
    if name is None:
        return "cuda:0"
    text = str(name).strip().lower()
    if text == "cpu":
        return "cpu"
    if text == "cuda":
        return "cuda:0"
    if text.startswith("cuda:"):
        try:
            index = int(text.split(":", 1)[1])
        except ValueError:
            return text
        return f"cuda:{index}"
    return text


def device_index(name: Optional[str]) -> Optional[int]:
    """返回 cuda 设备序号；cpu 或无法解析出序号的设备返回 ``None``。"""
    text = normalize_device(name)
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _query_mem_info(name: Optional[str]) -> Optional[tuple]:
    """查询 cuda 设备 ``(free_bytes, total_bytes)``。

    cpu 设备或查询失败（torch 未安装 / 无 CUDA / 调用异常）返回 ``None``，
    绝不向上抛异常。torch 延迟 import，便于测试环境注入假的 torch 命名空间。
    """
    index = device_index(name)
    if index is None:
        return None
    try:
        import torch  # 延迟 import：保证无 torch 环境下模块可导入、可单测

        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        return int(free_bytes), int(total_bytes)
    except Exception:
        return None


def get_device_free_mb(name: Optional[str]) -> Optional[int]:
    """返回设备空闲显存（MB）；cpu 设备或查询失败返回 ``None``（不抛异常）。"""
    info = _query_mem_info(name)
    return None if info is None else info[0] // (1024 * 1024)


def get_device_total_mb(name: Optional[str]) -> Optional[int]:
    """返回设备总显存（MB）；cpu 设备或查询失败返回 ``None``（不抛异常）。"""
    info = _query_mem_info(name)
    return None if info is None else info[1] // (1024 * 1024)


# ---------------------------------------------------------------------------
# 瞬态显存预估（不含常驻权重——权重启动加载后已从设备空闲显存中扣除）
# ---------------------------------------------------------------------------


def estimate_align_mb(align_batch_size: int) -> int:
    """对齐阶段瞬态显存预估（MB）：512MB 固定工作余量 + 256MB × batch。"""
    return 512 + 256 * int(align_batch_size)


def estimate_diar_mb(audio_seconds: float) -> int:
    """说话人识别瞬态显存预估（MB）：256MB 工作区 + 0.0625MB/s × 音频时长。"""
    return 256 + math.ceil(0.0625 * float(audio_seconds))


def estimate_task_need_mb(
    align_batch_size: int,
    audio_seconds: float,
    aligner_device: Optional[str],
    diarizer_device: Optional[str],
) -> Dict[str, int]:
    """按设备拆分单任务瞬态显存需求（MB）。

    对齐侧需求记在 aligner 设备、说话人侧记在 diarizer 设备；
    两者为同一设备时求和；cpu 设备不入表（不参与显存准入）。
    """
    need: Dict[str, int] = {}
    aligner = normalize_device(aligner_device)
    diarizer = normalize_device(diarizer_device)
    if device_index(aligner) is not None:
        need[aligner] = need.get(aligner, 0) + estimate_align_mb(align_batch_size)
    if device_index(diarizer) is not None:
        need[diarizer] = need.get(diarizer, 0) + estimate_diar_mb(audio_seconds)
    return need


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Ticket:
    """排队凭证：按设备的瞬态显存需求 + 放行 Future + 是否已获许可。"""

    need_mb_by_device: Dict[str, int]
    future: Optional["asyncio.Future"] = None
    admitted: bool = False


class GpuScheduler:
    """GPU 显存感知任务调度器（全局 FIFO + 按设备显存准入 + 并发上限）。

    放行条件（须同时满足）：
    1. 运行任务数 < ``max_concurrent_tasks``；
    2. 队首任务需求表中的每个 cuda 设备实测空闲显存
       >= 该设备需求 + ``gpu_reserve_mb``（查询失败的设备视为不满足，保守等待）。

    cpu 设备不参与显存检查（纯并发数限制）；只放行队首、队首不满足即停止，
    保证 FIFO 公平与无死锁。
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 2,
        gpu_reserve_mb: int = 1024,
        devices: Optional[Dict[str, str]] = None,
    ) -> None:
        self._max_concurrent_tasks = int(max_concurrent_tasks)
        self._gpu_reserve_mb = int(gpu_reserve_mb)
        # 设备名 -> 角色描述（用于 stats 展示），如 {"cuda:0": "vllm+aligner+diarizer"}
        self._devices: Dict[str, str] = {
            normalize_device(name): role for name, role in (devices or {}).items()
        }
        # 全局 FIFO 等待队列
        self._queue: Deque[_Ticket] = collections.deque()
        self._running: int = 0
        # 惰性创建：允许在无事件循环时构造调度器并同步单测 _try_admit
        self._cond: Optional[asyncio.Condition] = None

    # -- 内部工具 -----------------------------------------------------------

    def _get_cond(self) -> asyncio.Condition:
        """惰性获取 asyncio.Condition（首次调用须处于事件循环内）。"""
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def _try_admit(self) -> None:
        """尝试放行队首任务（纯同步逻辑，便于脱离 asyncio 单测）。

        只放行队首：队首不满足（并发已满 / 任一 cuda 设备空闲不足 /
        空闲查询失败）即停止检查，保证 FIFO 公平与无死锁。
        """
        while self._queue:
            ticket = self._queue[0]
            if self._running >= self._max_concurrent_tasks:
                return
            # 队首一次性检查其涉及的全部 cuda 设备
            for device, need_mb in ticket.need_mb_by_device.items():
                if device_index(device) is None:
                    continue  # cpu 设备不参与显存检查
                free_mb = get_device_free_mb(device)
                if free_mb is None or free_mb < need_mb + self._gpu_reserve_mb:
                    return  # 查询失败视为不满足，保守等待
            self._queue.popleft()
            self._running += 1
            ticket.admitted = True
            if ticket.future is not None and not ticket.future.done():
                ticket.future.set_result(None)

    # -- 核心接口 -----------------------------------------------------------

    @contextlib.asynccontextmanager
    async def slot(self, need_mb_by_device: Dict[str, int]) -> AsyncIterator[_Ticket]:
        """获取调度许可的异步上下文管理器（异常安全：许可必然释放）。

        ``need_mb_by_device`` 为按设备拆分的瞬态显存需求（MB），
        通常由 :func:`estimate_task_need_mb` 生成；cpu 设备可不入表。
        """
        ticket = await self._acquire(need_mb_by_device)
        try:
            yield ticket
        finally:
            await self._release(ticket)

    async def _acquire(self, need_mb_by_device: Dict[str, int]) -> _Ticket:
        """入队并等待放行；等待期间被取消时安全清理（不泄漏排队位/许可）。"""
        need = {
            normalize_device(device): int(mb)
            for device, mb in (need_mb_by_device or {}).items()
        }
        ticket = _Ticket(
            need_mb_by_device=need,
            future=asyncio.get_running_loop().create_future(),
        )
        async with self._get_cond():
            self._queue.append(ticket)
            self._try_admit()  # 入队后立即尝试放行
        if ticket.admitted:
            return ticket
        try:
            await ticket.future
        except asyncio.CancelledError:
            # 等待许可期间被取消（如客户端断连）：
            # 未放行 -> 移出队列并唤醒队首；已放行 -> 释放许可
            await self._abort_wait(ticket)
            raise
        return ticket

    async def _abort_wait(self, ticket: _Ticket) -> None:
        """处理等待放行期间被取消的 ticket（清理状态并唤醒后继）。"""
        async with self._get_cond():
            if ticket.admitted:
                ticket.admitted = False
                self._running -= 1
            else:
                try:
                    self._queue.remove(ticket)
                except ValueError:
                    pass
            self._try_admit()

    async def _release(self, ticket: _Ticket) -> None:
        """释放许可（slot 的 finally 必然调用），并唤醒队首重新准入。

        等待方 await 各自 ticket.future（非 cond.wait），放行由 _try_admit 的
        future.set_result 完成，Condition 仅作队列/计数操作的互斥锁使用。
        """
        async with self._get_cond():
            if ticket.admitted:
                ticket.admitted = False
                self._running -= 1
            self._try_admit()

    # -- 启动校验与状态 -----------------------------------------------------

    def start_up_validate(self, min_need_mb_by_device: Dict[str, int]) -> None:
        """启动校验：任一 cuda 设备空闲 < 最小需求 + reserve 时抛 RuntimeError。

        用于扩展模型加载后快速失败，避免带病进入"必然死锁"状态；
        查询失败的设备同样视为不满足。cpu 设备不做显存校验。
        """
        for device, min_need_mb in (min_need_mb_by_device or {}).items():
            name = normalize_device(device)
            if device_index(name) is None:
                continue
            min_need_mb = int(min_need_mb)
            free_mb = get_device_free_mb(name)
            required_mb = min_need_mb + self._gpu_reserve_mb
            if free_mb is None or free_mb < required_mb:
                free_text = (
                    "未知（mem_get_info 查询失败）" if free_mb is None else f"{free_mb}MB"
                )
                raise RuntimeError(
                    f"GPU 显存预算启动校验失败：设备 {name} 实测空闲 {free_text}，"
                    f"小于该设备最小瞬态需求 {min_need_mb}MB + 安全余量 "
                    f"gpu_reserve_mb {self._gpu_reserve_mb}MB（合计 {required_mb}MB）。"
                    f"建议：1) 调低 vLLM 显存预分配比例（如 "
                    f"--gpu-memory-utilization 0.55）为扩展模型腾出显存；"
                    f"或 2) 将对齐/说话人模型迁移到独立设备，例如："
                    f"--aligner-device cuda:1 --diarizer-device cuda:1"
                )

    def stats(self) -> Dict[str, Any]:
        """调度状态快照（供 /health/detail 等展示；cpu 设备 free/total 为 null）。"""
        devices = []
        for name, role in self._devices.items():
            is_cuda = device_index(name) is not None
            devices.append(
                {
                    "device": name,
                    "role": role,
                    "freeVramMb": get_device_free_mb(name) if is_cuda else None,
                    "totalVramMb": get_device_total_mb(name) if is_cuda else None,
                }
            )
        return {
            "runningTasks": self._running,
            "queuedTasks": len(self._queue),
            "maxConcurrentTasks": self._max_concurrent_tasks,
            "gpuReserveMb": self._gpu_reserve_mb,
            "devices": devices,
        }
