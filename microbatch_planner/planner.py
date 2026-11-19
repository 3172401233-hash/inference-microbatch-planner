"""微批规划核心算法。

规划策略概述：

1. 每个批次只允许一个 adapter，避免批内适配器不兼容；同一请求至多进入一个
   批次，不会在互不兼容的批次间重复计算。
2. 批次内受显存、执行槽位、token 总量三重约束。
3. 批次选择顺序：临近 deadline 的请求优先（紧急层），其次按租户累计服务量
   除以租户权重（加权公平队列）选择最落后租户，再在租户内选择 adapter。
4. adapter 切换代价较高时，优先延续当前 adapter 合并同适配器请求；但当其他
   租户的等待时间超过 fairness_guard_ms 时强制打断，防止大租户长期独占。
5. 所有比较均带显式 tie-break 键，相同输入与时间基准下输出稳定。
"""

from typing import Dict, List, Optional, Tuple

from .models import (
    BatchPlan,
    PlanResult,
    Request,
    RoundResources,
    UnschedulableReason,
    UnschedulableRequest,
)

# 浮点预算的容差：仅用于消除舍入误差，不改变“刚好不足”语义。
_EPS = 1e-9


class MicroBatchPlanner:
    def __init__(
        self,
        resources,
        switch_overhead_ms,
        base_time_ms=0.0,
        urgency_window_ms=None,
        fairness_guard_ms=None,
    ):
        # type: (RoundResources, float, float, Optional[float], Optional[float]) -> None
        if switch_overhead_ms < 0:
            raise ValueError("switch_overhead_ms 不能为负")
        self.resources = resources
        self.switch_overhead_ms = float(switch_overhead_ms)
        self.base_time_ms = float(base_time_ms)
        # slack 小于该窗口（或等待超过该窗口）的请求视为紧急。
        self.urgency_window_ms = (
            float(switch_overhead_ms)
            if urgency_window_ms is None
            else float(urgency_window_ms)
        )
        # 其他租户等待超过该阈值时，禁止继续合并当前 adapter。
        self.fairness_guard_ms = (
            float("inf") if fairness_guard_ms is None else float(fairness_guard_ms)
        )

    def plan(self, requests):
        # type: (List[Request]) -> PlanResult
        # request_id 排序提供稳定的处理基准，与插入顺序解耦。
        pending = sorted(requests, key=lambda item: item.request_id)
        unschedulable = []  # type: List[UnschedulableRequest]

        # 永久不可调度的请求（任何单批都放不下）直接剔除并给出原因。
        feasible = []  # type: List[Request]
        for request in pending:
            permanent = self._permanent_reason(request)
            if permanent is not None:
                unschedulable.append(
                    UnschedulableRequest(request.request_id, permanent[0], permanent[1])
                )
            else:
                feasible.append(request)

        batches = []  # type: List[BatchPlan]
        # 各租户累计被服务时间（含为其承担的切换开销），用于加权公平。
        tenant_usage = {}  # type: Dict[str, float]
        for request in feasible:
            tenant_usage.setdefault(request.tenant_id, 0.0)

        clock = self.base_time_ms
        current_adapter = None  # type: Optional[str]
        batch_index = 0

        while feasible:
            # 当前时钟下已经无法满足 deadline 的请求立即判定失败，不再占用批次。
            remaining = []  # type: List[Request]
            for request in feasible:
                earliest_finish = self._earliest_finish(
                    request, clock, current_adapter
                )
                if earliest_finish > request.deadline_ms + _EPS:
                    unschedulable.append(
                        UnschedulableRequest(
                            request.request_id,
                            UnschedulableReason.DEADLINE_MISSED,
                            "最早可完成时间 %.3f 晚于 deadline %.3f"
                            % (earliest_finish, request.deadline_ms),
                        )
                    )
                else:
                    remaining.append(request)
            feasible = remaining
            if not feasible:
                break

            seed = self._select_seed(
                feasible, clock, current_adapter, tenant_usage
            )
            if seed is None:
                # 没有请求能在当前资源下组成批次：按优先级给出失败原因。
                blocked = sorted(
                    feasible,
                    key=lambda item: (
                        0 if self._is_urgent(item, clock, current_adapter) else 1,
                        self._fair_score(item, tenant_usage),
                        item.deadline_ms,
                        item.tenant_id,
                        item.adapter_id,
                        item.request_id,
                    ),
                )
                request = blocked[0]
                reason = self._contextual_reason(request, clock, current_adapter)
                unschedulable.append(
                    UnschedulableRequest(request.request_id, reason[0], reason[1])
                )
                feasible.remove(request)
                continue

            batch, switch_cost, feasible = self._build_batch(
                seed, feasible, clock, current_adapter, tenant_usage
            )
            run_time = max(item.run_time_ms for item in batch)
            finish = clock + switch_cost + run_time
            plan = BatchPlan(
                batch_index=batch_index,
                request_ids=sorted(item.request_id for item in batch),
                adapter_id=seed.adapter_id,
                start_time_ms=clock,
                switch_overhead_ms=switch_cost,
                run_time_ms=run_time,
                estimated_finish_time_ms=finish,
                total_tokens=sum(item.total_tokens for item in batch),
                total_memory=sum(item.estimated_memory for item in batch),
            )
            batches.append(plan)
            for item in batch:
                tenant_usage[item.tenant_id] = (
                    tenant_usage.get(item.tenant_id, 0.0)
                    + switch_cost
                    + item.run_time_ms
                )
            clock = finish
            current_adapter = seed.adapter_id
            batch_index += 1

        unschedulable.sort(key=lambda item: item.request_id)
        return PlanResult(batches=batches, unschedulable=unschedulable)

    def _permanent_reason(self, request):
        # type: (Request) -> Optional[Tuple[UnschedulableReason, str]]
        if request.estimated_memory > self.resources.memory_budget + _EPS:
            return (
                UnschedulableReason.MEMORY_EXCEEDED,
                "单请求显存 %.3f 超过批次预算 %.3f"
                % (request.estimated_memory, self.resources.memory_budget),
            )
        if (
            self.resources.max_tokens is not None
            and request.total_tokens > self.resources.max_tokens + _EPS
        ):
            return (
                UnschedulableReason.TOKEN_LIMIT_EXCEEDED,
                "单请求 token 数 %d 超过批次上限 %d"
                % (request.total_tokens, self.resources.max_tokens),
            )
        return None

    def _switch_cost(self, current_adapter, target_adapter):
        # type: (Optional[str], str) -> float
        if current_adapter == target_adapter:
            return 0.0
        return self.switch_overhead_ms

    def _earliest_finish(self, request, clock, current_adapter):
        # type: (Request, float, Optional[str]) -> float
        """请求独占一个批次时的最早可完成时间（含适配器切换开销）。"""
        return (
            clock
            + self._switch_cost(current_adapter, request.adapter_id)
            + request.run_time_ms
        )

    def _is_urgent(self, request, clock, current_adapter):
        # type: (Request, float, Optional[str]) -> bool
        earliest_finish = (
            clock
            + self._switch_cost(current_adapter, request.adapter_id)
            + request.run_time_ms
        )
        slack = request.deadline_ms - earliest_finish
        if slack <= self.urgency_window_ms + _EPS:
            return True
        waited = clock - request.arrival_time_ms
        return waited >= self.fairness_guard_ms - _EPS

    def _fair_score(self, request, tenant_usage):
        # type: (Request, Dict[str, float]) -> float
        return tenant_usage.get(request.tenant_id, 0.0) / request.tenant_weight

    def _select_seed(self, feasible, clock, current_adapter, tenant_usage):
        # type: (List[Request], float, Optional[str], Dict[str, float]) -> Optional[Request]
        candidates = [
            item
            for item in feasible
            if self._contextual_reason(item, clock, current_adapter) is None
        ]
        if not candidates:
            return None

        def seed_key(item):
            urgent = 0 if self._is_urgent(item, clock, current_adapter) else 1
            same_adapter = 0 if item.adapter_id == current_adapter else 1
            return (
                urgent,
                self._fair_score(item, tenant_usage),
                item.deadline_ms,
                item.tenant_id,
                same_adapter,
                item.adapter_id,
                item.request_id,
            )

        return min(candidates, key=seed_key)

    def _build_batch(self, seed, feasible, clock, current_adapter, tenant_usage):
        # type: (Request, List[Request], float, Optional[str], Dict[str, float]) -> Tuple[List[Request], float, List[Request]]
        adapter_id = seed.adapter_id
        switch_cost = self._switch_cost(current_adapter, adapter_id)
        batch = [seed]
        remaining = [item for item in feasible if item.request_id != seed.request_id]

        used_memory = seed.estimated_memory
        used_tokens = seed.total_tokens
        used_slots = 1
        run_time = seed.run_time_ms

        while True:
            fillable = []  # type: List[Request]
            for item in remaining:
                if item.adapter_id != adapter_id:
                    continue
                if used_slots >= self.resources.slot_count:
                    continue
                if used_memory + item.estimated_memory > self.resources.memory_budget + _EPS:
                    continue
                if (
                    self.resources.max_tokens is not None
                    and used_tokens + item.total_tokens
                    > self.resources.max_tokens + _EPS
                ):
                    continue
                projected_run = max(run_time, item.run_time_ms)
                projected_finish = clock + switch_cost + projected_run
                if projected_finish > item.deadline_ms + _EPS:
                    continue
                # 新成员可能拉长批次执行时间，批内已有请求同样不能突破 deadline。
                if any(
                    projected_finish > member.deadline_ms + _EPS
                    for member in batch
                ):
                    continue
                if self._other_tenant_starved(item, remaining, clock):
                    continue
                fillable.append(item)
            if not fillable:
                break

            def fill_key(item):
                return (
                    self._fair_score(item, tenant_usage),
                    item.deadline_ms,
                    item.tenant_id,
                    item.request_id,
                )

            picked = min(fillable, key=fill_key)
            batch.append(picked)
            remaining.remove(picked)
            used_memory += picked.estimated_memory
            used_tokens += picked.total_tokens
            used_slots += 1
            run_time = max(run_time, picked.run_time_ms)

        return batch, switch_cost, remaining

    def _other_tenant_starved(self, candidate, remaining, clock):
        # type: (Request, List[Request], float) -> bool
        """若有别的租户等待已超过公平保护阈值，则停止向当前 adapter 合并。"""
        for item in remaining:
            if item.tenant_id == candidate.tenant_id:
                continue
            if clock - item.arrival_time_ms >= self.fairness_guard_ms - _EPS:
                return True
        return False

    def _contextual_reason(self, request, clock, current_adapter):
        # type: (Request, float, Optional[str]) -> Optional[Tuple[UnschedulableReason, str]]
        # 调用前永久原因已剔除，这里只处理与当前时钟/批次容量相关的约束。
        if request.estimated_memory > self.resources.memory_budget + _EPS:
            return (
                UnschedulableReason.MEMORY_EXCEEDED,
                "剩余显存不足以容纳请求 %s" % request.request_id,
            )
        if (
            self.resources.max_tokens is not None
            and request.total_tokens > self.resources.max_tokens + _EPS
        ):
            return (
                UnschedulableReason.TOKEN_LIMIT_EXCEEDED,
                "剩余 token 预算不足以容纳请求 %s" % request.request_id,
            )
        # 槽位以单请求为最小单位，槽位耗尽等价于当前批次无法再接纳该请求。
        if self.resources.slot_count < 1:
            return (
                UnschedulableReason.SLOT_LIMIT_EXCEEDED,
                "没有可用执行槽位",
            )
        earliest_finish = self._earliest_finish(
            request, clock, current_adapter
        )
        if earliest_finish > request.deadline_ms + _EPS:
            return (
                UnschedulableReason.DEADLINE_MISSED,
                "最早可完成时间 %.3f 晚于 deadline %.3f"
                % (earliest_finish, request.deadline_ms),
            )
        return None

