"""规划器输入/输出数据模型。

所有时间单位为毫秒，显存单位由调用方约定（例如 MB），全模块保持一致即可。
兼容 Python 3.8.2：仅使用 dataclasses 与 typing，不使用 PEP 585 泛型。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class UnschedulableReason(str, Enum):
    """请求无法被任何批次接纳的原因。"""

    MEMORY_EXCEEDED = "memory_exceeded"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    SLOT_LIMIT_EXCEEDED = "slot_limit_exceeded"
    DEADLINE_MISSED = "deadline_missed"


@dataclass(frozen=True)
class Request:
    """单个推理请求的输入假设。

    prompt_tokens/max_output_tokens/estimated_memory/step_time_ms 均为调用方
    提供的预估值，规划器本身不做显存测量或性能剖析。
    """

    request_id: str
    tenant_id: str
    adapter_id: str
    prompt_tokens: int
    max_output_tokens: int
    deadline_ms: float
    estimated_memory: float
    step_time_ms: float
    tenant_weight: float = 1.0
    arrival_time_ms: float = 0.0

    def __post_init__(self):
        if self.prompt_tokens < 0 or self.max_output_tokens < 0:
            raise ValueError("token 长度不能为负: %s" % self.request_id)
        if self.estimated_memory < 0 or self.step_time_ms < 0:
            raise ValueError("显存与单步耗时不能为负: %s" % self.request_id)
        if self.tenant_weight <= 0:
            raise ValueError("租户权重必须为正: %s" % self.tenant_id)
        if self.deadline_ms < self.arrival_time_ms:
            raise ValueError("deadline 早于到达时间: %s" % self.request_id)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.max_output_tokens

    @property
    def run_time_ms(self) -> float:
        """满额生成 max_output_tokens 个 token 的预估执行时间。"""
        return self.max_output_tokens * self.step_time_ms


@dataclass(frozen=True)
class RoundResources:
    """每一轮规划可用的物理资源。"""

    memory_budget: float
    slot_count: int
    max_tokens: Optional[int] = None

    def __post_init__(self):
        if self.memory_budget < 0 or self.slot_count <= 0:
            raise ValueError("资源预算非法")


@dataclass(frozen=True)
class BatchPlan:
    """一个待执行微批。"""

    batch_index: int
    request_ids: List[str]
    adapter_id: str
    start_time_ms: float
    switch_overhead_ms: float
    run_time_ms: float
    estimated_finish_time_ms: float
    total_tokens: int
    total_memory: float

    def to_dict(self):
        return {
            "batch_index": self.batch_index,
            "request_ids": list(self.request_ids),
            "adapter_id": self.adapter_id,
            "start_time_ms": self.start_time_ms,
            "switch_overhead_ms": self.switch_overhead_ms,
            "run_time_ms": self.run_time_ms,
            "estimated_finish_time_ms": self.estimated_finish_time_ms,
            "total_tokens": self.total_tokens,
            "total_memory": self.total_memory,
        }


@dataclass(frozen=True)
class UnschedulableRequest:
    """无法满足的请求及原因。"""

    request_id: str
    reason: UnschedulableReason
    detail: str

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PlanResult:
    """整轮规划结果。"""

    batches: List[BatchPlan] = field(default_factory=list)
    unschedulable: List[UnschedulableRequest] = field(default_factory=list)

    def to_dict(self):
        return {
            "batches": [batch.to_dict() for batch in self.batches],
            "unschedulable": [item.to_dict() for item in self.unschedulable],
        }
