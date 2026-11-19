# inference-microbatch-planner

在延迟上限（deadline）、显存/槽位预算与租户公平配额下规划推理微批的纯 Python
实现，兼容 **Python 3.8.2**，不依赖 CUDA 或任何第三方库。

## 输入假设（重要）

规划器只做调度决策，不做任何性能剖析或显存测量。以下数值全部由调用方作为
**假设输入**提供，规划器不保证其与真实硬件行为一致：

- **显存估算 `estimated_memory`**：单个请求在批内占用的预估显存（单位自定义，
  例如 MB），批内显存为各请求估算值之和。规划器只做加总和预算比较。
- **计算时间 `step_time_ms` 与 `max_output_tokens`**：请求每生成一个 token 的
  预估单步耗时；请求执行时间按 `max_output_tokens * step_time_ms` 估算，批次
  执行时间取批内最大值（同一批次并行步进）。
- `prompt_tokens` 仅参与总 token 预算核算，不单独换算耗时。
- 适配器切换开销 `switch_overhead_ms` 同样是输入假设，仅在相邻批次适配器
  不同（或首批冷加载）时计入。

若输入假设偏差较大，规划结果的时间与显存判断也会随之偏差。

## 数据模型

见 `microbatch_planner/models.py`。

- `Request`：`request_id`、`tenant_id`、`adapter_id`、`prompt_tokens`、
  `max_output_tokens`、`deadline_ms`、`estimated_memory`、`step_time_ms`、
  `tenant_weight`、`arrival_time_ms`。
- `RoundResources`：每轮 `memory_budget`、`slot_count` 与可选 `max_tokens`。
- `BatchPlan`：批次请求、适配器、起始时间、切换开销、执行时间、预计完成时间、
  总 token 与总显存。
- `UnschedulableRequest`：无法满足的请求及原因
  （`memory_exceeded` / `token_limit_exceeded` / `slot_limit_exceeded` /
  `deadline_missed`）。

## 规划策略

实现位于 `microbatch_planner/planner.py`。

1. **单适配器批次**：每个批次只含一个 `adapter_id`，从根本上避免同一请求被
   放进互不兼容的批次；每个请求至多被调度一次，未调度者进入 `unschedulable`。
2. **三重资源约束**：批内总显存不超过 `memory_budget`，请求数不超过
   `slot_count`，总 token（prompt + 最大输出）不超过可选的 `max_tokens`。
   浮点预算比较带 `1e-9` 容差，因此“刚好等于预算”可以同批，“刚好超出”则拆分。
3. **Deadline 优先**：当请求的 slack（deadline 减去最早可完成时间）落入
   `urgency_window_ms` 窗口，或等待时间达到公平保护阈值时标记为紧急，紧急请求
   跨批次优先；当前时钟下即使最早执行也无法按时完成的请求立即判为
   `deadline_missed`，不再阻塞其他请求。
4. **加权租户公平**：非紧急请求按 `租户累计服务时间 / tenant_weight` 选择最落后
   的租户，权重大的租户获得更密集的服务，但不会永久压制小租户。
5. **适配器亲和与防饿死**：批次填充时优先合并相同适配器请求以省去切换开销；
   一旦其他租户等待时间超过 `fairness_guard_ms`，停止继续合并当前适配器，强制
   下一批切换，避免大租户长期独占。
6. **稳定性**：所有选择都带显式 tie-break 键（紧急度 → 公平分 → deadline →
   租户 → 适配器 → request_id），并按 `request_id` 建立初始处理顺序。相同输入与
   相同 `base_time_ms` 时间基准下，无论传入顺序如何，批次结果逐字段一致。

时间推进公式：`下一批起始时间 = 上一批起始时间 + 切换开销 + 批内最大执行时间`。

## 使用示例

```python
from microbatch_planner import MicroBatchPlanner, Request, RoundResources

requests = [
    Request("r1", "tenantA", "lora-x", 120, 40, 800.0, 240.0, 6.0,
            tenant_weight=2.0),
    Request("r2", "tenantB", "lora-y", 20, 8, 300.0, 60.0, 5.0),
]
resources = RoundResources(memory_budget=400.0, slot_count=2, max_tokens=400)
planner = MicroBatchPlanner(
    resources,
    switch_overhead_ms=15.0,
    base_time_ms=0.0,
    urgency_window_ms=15.0,
    fairness_guard_ms=200.0,
)
result = planner.plan(requests)
result.to_dict()  # 可直接 JSON 序列化的批次与失败原因
```

## 测试

覆盖长短请求混排、适配器切换与合并、显存刚好不足/刚好满足、deadline 冲突、
租户权重公平、公平保护防饿死、tie-break 稳定性，以及 token/槽位约束与请求不重复
调度：

```bash
python -m unittest
```
