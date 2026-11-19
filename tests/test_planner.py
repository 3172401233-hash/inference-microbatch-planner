import unittest

from microbatch_planner import (
    MicroBatchPlanner,
    PlanResult,
    Request,
    RoundResources,
    UnschedulableReason,
)


def make_request(
    request_id,
    tenant_id="A",
    adapter_id="X",
    prompt=10,
    max_output=10,
    deadline=10000.0,
    memory=10.0,
    step_time=10.0,
    weight=1.0,
    arrival=0.0,
):
    return Request(
        request_id=request_id,
        tenant_id=tenant_id,
        adapter_id=adapter_id,
        prompt_tokens=prompt,
        max_output_tokens=max_output,
        deadline_ms=deadline,
        estimated_memory=memory,
        step_time_ms=step_time,
        tenant_weight=weight,
        arrival_time_ms=arrival,
    )


class PlannerTest(unittest.TestCase):
    def test_long_short_mix_short_deadline_scheduled_first(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=4)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        long_req = make_request(
            "long",
            tenant_id="A",
            adapter_id="X",
            prompt=100,
            max_output=100,
            step_time=10.0,
            deadline=2000.0,
            memory=100.0,
        )
        short_req = make_request(
            "short",
            tenant_id="B",
            adapter_id="Y",
            prompt=10,
            max_output=5,
            step_time=10.0,
            deadline=100.0,
            memory=10.0,
        )
        result = planner.plan([long_req, short_req])
        self.assertEqual(len(result.batches), 2)
        self.assertEqual(result.batches[0].request_ids, ["short"])
        self.assertEqual(result.batches[0].estimated_finish_time_ms, 60.0)
        self.assertEqual(result.batches[1].request_ids, ["long"])
        self.assertEqual(result.batches[1].estimated_finish_time_ms, 1070.0)
        self.assertEqual([item.reason for item in result.unschedulable], [])

    def test_adapter_switch_and_same_adapter_merge(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=4)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        requests = [
            make_request("a1", adapter_id="X"),
            make_request("b1", tenant_id="B", adapter_id="Y"),
            make_request("a2", adapter_id="X"),
        ]
        result = planner.plan(requests)
        self.assertEqual(len(result.batches), 2)
        self.assertEqual(result.batches[0].adapter_id, "X")
        self.assertEqual(result.batches[0].request_ids, ["a1", "a2"])
        self.assertEqual(result.batches[0].switch_overhead_ms, 10.0)
        self.assertEqual(result.batches[1].adapter_id, "Y")
        self.assertEqual(result.batches[1].request_ids, ["b1"])
        self.assertEqual(result.batches[1].switch_overhead_ms, 10.0)

    def test_memory_just_short_splits_batch_and_oversized_rejected(self):
        resources = RoundResources(memory_budget=100.0, slot_count=4)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        requests = [
            make_request("r1", memory=60.0),
            make_request("r2", memory=45.0),
            make_request("r3", memory=101.0),
        ]
        result = planner.plan(requests)
        self.assertEqual([batch.request_ids for batch in result.batches], [["r1"], ["r2"]])
        self.assertEqual(len(result.unschedulable), 1)
        self.assertEqual(result.unschedulable[0].request_id, "r3")
        self.assertEqual(
            result.unschedulable[0].reason, UnschedulableReason.MEMORY_EXCEEDED
        )

    def test_memory_exactly_fitting_requests_share_batch(self):
        resources = RoundResources(memory_budget=100.0, slot_count=4)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        result = planner.plan(
            [make_request("r1", memory=60.0), make_request("r2", memory=40.0)]
        )
        self.assertEqual(len(result.batches), 1)
        self.assertEqual(result.batches[0].request_ids, ["r1", "r2"])
        self.assertEqual(result.batches[0].total_memory, 100.0)

    def test_deadline_conflict_marks_missed_request(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=4)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        impossible = make_request(
            "impossible", max_output=10, step_time=10.0, deadline=50.0
        )
        normal = make_request("normal", deadline=1000.0)
        result = planner.plan([impossible, normal])
        self.assertEqual(result.batches[0].request_ids, ["normal"])
        self.assertEqual(result.unschedulable[0].request_id, "impossible")
        self.assertEqual(
            result.unschedulable[0].reason, UnschedulableReason.DEADLINE_MISSED
        )

    def test_urgent_long_request_preempts_and_starves_late_short_one(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=4)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        urgent_long = make_request(
            "long",
            adapter_id="X",
            max_output=100,
            step_time=10.0,
            deadline=1010.0,
        )
        short = make_request(
            "short",
            tenant_id="B",
            adapter_id="Y",
            max_output=5,
            step_time=10.0,
            deadline=200.0,
        )
        result = planner.plan([urgent_long, short])
        self.assertEqual(result.batches[0].request_ids, ["long"])
        self.assertEqual(result.batches[0].estimated_finish_time_ms, 1010.0)
        self.assertEqual(result.unschedulable[0].request_id, "short")
        self.assertEqual(
            result.unschedulable[0].reason, UnschedulableReason.DEADLINE_MISSED
        )

    def test_tenant_fairness_uses_weights(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=1)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        requests = [
            make_request("a1", tenant_id="A", weight=3.0),
            make_request("a2", tenant_id="A", weight=3.0),
            make_request("a3", tenant_id="A", weight=3.0),
            make_request("b1", tenant_id="B", weight=1.0),
        ]
        result = planner.plan(requests)
        order = [batch.request_ids[0] for batch in result.batches]
        self.assertEqual(order, ["a1", "b1", "a2", "a3"])

    def test_fairness_guard_blocks_indefinite_adapter_merging(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=2)
        planner = MicroBatchPlanner(
            resources,
            switch_overhead_ms=10.0,
            fairness_guard_ms=100.0,
        )
        requests = [
            make_request("a1", tenant_id="A", adapter_id="X", step_time=10.0),
            make_request("a2", tenant_id="A", adapter_id="X", step_time=10.0),
            make_request("a3", tenant_id="A", adapter_id="X", step_time=10.0),
            make_request("b1", tenant_id="B", adapter_id="Y", step_time=10.0),
            make_request("b2", tenant_id="B", adapter_id="Y", step_time=10.0),
        ]
        result = planner.plan(requests)
        self.assertEqual(len(result.batches), 4)
        self.assertEqual(result.batches[0].request_ids, ["a1", "a2"])
        # b1 单独成批：继续合并 b2 会让 A 租户剩余请求超过保护阈值。
        self.assertEqual(result.batches[1].request_ids, ["b1"])
        scheduled = [
            request_id
            for batch in result.batches
            for request_id in batch.request_ids
        ]
        self.assertEqual(sorted(scheduled), ["a1", "a2", "a3", "b1", "b2"])

    def test_tie_break_is_stable_and_input_order_independent(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=1)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        requests = [
            make_request("r2", deadline=1000.0),
            make_request("r1", deadline=1000.0),
        ]
        first = planner.plan(requests)
        second = planner.plan(list(reversed(requests)))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            [batch.request_ids[0] for batch in first.batches], ["r1", "r2"]
        )
        # 相同时间基准重复规划结果完全一致。
        self.assertEqual(planner.plan(requests).to_dict(), first.to_dict())

    def test_each_request_scheduled_at_most_once(self):
        resources = RoundResources(memory_budget=100.0, slot_count=2)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        requests = [
            make_request("r1", memory=60.0),
            make_request("r2", memory=60.0),
            make_request("r3", memory=60.0),
            make_request("r4", memory=200.0),
        ]
        result = planner.plan(requests)
        scheduled = [
            request_id
            for batch in result.batches
            for request_id in batch.request_ids
        ]
        rejected = [item.request_id for item in result.unschedulable]
        self.assertEqual(len(scheduled), len(set(scheduled)))
        self.assertEqual(sorted(scheduled + rejected), ["r1", "r2", "r3", "r4"])

    def test_token_limit_splits_and_rejects_oversized(self):
        resources = RoundResources(
            memory_budget=1000.0, slot_count=4, max_tokens=100
        )
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        requests = [
            make_request("r1", prompt=70, max_output=10),
            make_request("r2", prompt=20, max_output=10),
            make_request("r3", prompt=95, max_output=10),
        ]
        result = planner.plan(requests)
        self.assertEqual([batch.request_ids for batch in result.batches], [["r1"], ["r2"]])
        self.assertEqual(
            result.unschedulable[0].reason,
            UnschedulableReason.TOKEN_LIMIT_EXCEEDED,
        )

    def test_slot_limit_caps_batch_size(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=2)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=10.0)
        result = planner.plan([make_request("r%d" % index) for index in range(4)])
        self.assertEqual([len(batch.request_ids) for batch in result.batches], [2, 2])

    def test_batch_members_keep_deadline_when_longer_request_joins(self):
        # 回归：批次后续加入更长请求会拉长整体执行时间，已在批内的短
        # deadline 请求不能因此被拖过 deadline；短请求应单独成批。
        resources = RoundResources(memory_budget=1000.0, slot_count=4)
        planner = MicroBatchPlanner(resources, switch_overhead_ms=5.0)
        tight = make_request(
            "tight",
            adapter_id="y",
            max_output=10,
            step_time=5.0,
            deadline=100.0,
        )
        long_one = make_request(
            "long",
            adapter_id="y",
            max_output=40,
            step_time=5.0,
            deadline=500.0,
        )
        result = planner.plan([tight, long_one])
        self.assertEqual(result.batches[0].request_ids, ["tight"])
        self.assertLessEqual(
            result.batches[0].estimated_finish_time_ms, tight.deadline_ms
        )
        for batch in result.batches:
            for request_id in batch.request_ids:
                deadline = tight.deadline_ms if request_id == "tight" else long_one.deadline_ms
                self.assertLessEqual(batch.estimated_finish_time_ms, deadline)

    def test_base_time_shifts_all_timestamps_stably(self):
        resources = RoundResources(memory_budget=1000.0, slot_count=4)
        planner = MicroBatchPlanner(
            resources, switch_overhead_ms=10.0, base_time_ms=500.0
        )
        result = planner.plan([make_request("r1", deadline=2000.0)])
        self.assertEqual(result.batches[0].start_time_ms, 500.0)
        self.assertEqual(result.batches[0].estimated_finish_time_ms, 610.0)


if __name__ == "__main__":
    unittest.main()
