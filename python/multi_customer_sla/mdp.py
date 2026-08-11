"""
Multi-customer, multi-item inventory allocation under heterogeneous SLAs — DynaML port.

Ported from src/lib/models/multi_customer_sla/{mdp.h,mdp.cpp,policies.h,policies.cpp}
(the C++ model written against Tarkan Temizoz's DynaPlex fork) to run on the
official dynaplex package (https://dynaplex.github.io/DynaPlex/latest/), which
ships as a precompiled wheel with models authored in DynaML rather than C++.

Model summary (see docs/paper_improved_complete.tex for the full formulation):
  - A single warehouse stocks |I| items for |C| customers, each with its own
    numeric backorder allowance beta_c per review horizon.
  - Ordering is STATIC: every period, every item is replenished to a fixed
    base-stock level via IP_i(t) = OH'_i(t) + pipeline - backorders.
  - Allocation is DYNAMIC: the action selects which of four rationing rules
    (FCFS / SLA-gap myopic / proportional / cost-based greedy) is used to
    split available stock across customers once demand is realised.
  - Cost = holding cost every period (on post-allocation on-hand) + an SLA
    penalty at each customer's review-horizon end (on backorders in excess
    of beta_c, inclusive of that period's own backorder).

Scope note (prototype pass): this is a faithful, complete port of the MDP's
transition dynamics (all four rationing rules) and of BaseStockPolicy (a
constant-action policy — action 0 = FCFS by default), meant to validate the
DynaML toolchain end-to-end before porting GreedyDynamicPolicy, the
allocation-network featurizer, and the executables. Per-customer reporting-only
telemetry (ShortfallPerReviewPeriod, SuccessPerReviewPeriod, ReviewHorizonIndex,
HorizonStartPeriod) was dropped from state — it never affects dynamics or cost,
only diagnostics — and lead times are assumed >= 1 (matching dynaplex's own
lost_sales.py convention: L == 0, the C++ model's immediate-replenishment edge
case, is not yet supported here).
"""
from dataclasses import dataclass

from dynaplex.modelling import (
    AliasSampler,
    const_dataclass,
    DiscreteDist,
    FifoQueue,
    HorizonType,
    StateCategory,
    TrajectoryContext,
    Validity,
)


@dataclass(slots=True)
class State:
    """
    pipelines[i]: FIFO queue of item i's on-hand + pipeline, front first.
      pipelines[i][0] = OH'_i(t), on-hand INCLUDING this period's arriving
        order Q_i(t-L) but BEFORE this period's allocation (see docs/
        paper_improved_complete.tex, "On-Hand Inventory Before Allocation").
      pipelines[i][1:] = the remaining L_i - 1 orders still in transit.
    backorder[c*|I|+i]: BO_{c,i}(t), flattened per (customer, item).
    inventory_position[i]: IP_i(t), the base-stock ordering target quantity.
    current_demand / current_allocation[c*|I|+i]: this period's D_{c,i}(t) /
      A_{c,i}(t), flattened — kept for feature extraction / inspection.
    cumulative_backorder[c]: BO-bar_c(t), reset to 0 at each horizon end.
    time_remaining[c]: T_rem_c(t), periods left in customer c's current
      review horizon.
    last_rationing_action: the rationing rule chosen for the CURRENT period
      (set in modify_state_with_action, consumed in modify_state_with_event).
    """
    pipelines: list[FifoQueue]
    backorder: list[int]
    inventory_position: list[int]
    current_demand: list[int]
    current_allocation: list[int]
    cumulative_backorder: list[float]
    time_remaining: list[int]
    last_rationing_action: int
    period: int
    category: StateCategory


@const_dataclass(init=False, slots=True)
class MultiCustomerSlaMDP:
    number_of_customers: int
    number_of_items: int
    lead_times: list[int]
    holding_costs: list[float]
    backorder_allowances: list[int]
    review_horizons: list[int]
    penalty_costs: list[float]
    base_stock_level: list[int]
    demand_dists: list[DiscreteDist]
    demand_samplers: list[AliasSampler]
    benchmark_rationing_action: int
    num_actions: int
    horizon_type: HorizonType

    def __init__(
        self,
        number_of_customers: int,
        number_of_items: int,
        lead_times: list[int],
        holding_costs: list[float],
        backorder_allowances: list[int],
        review_horizons: list[int],
        penalty_costs: list[float],
        demand_rates: list[float],
        base_stock_level: list[int],
        high_demand_variance: list[int],
        total_rationing_actions: int = 4,
        benchmark_rationing_action: int = 0,
    ) -> None:
        # NOTE: __init__ is never called by the DynaML compiler; full CPython is allowed here.
        assert all(lt >= 1 for lt in lead_times), \
            "this port assumes lead_times >= 1 for every item (L == 0 is not yet supported)"
        assert len(lead_times) == number_of_items
        assert len(holding_costs) == number_of_items
        assert len(base_stock_level) == number_of_items
        assert len(backorder_allowances) == number_of_customers
        assert len(review_horizons) == number_of_customers
        assert len(penalty_costs) == number_of_customers
        assert len(demand_rates) == number_of_customers * number_of_items
        assert len(high_demand_variance) == number_of_customers * number_of_items

        self.number_of_customers = number_of_customers
        self.number_of_items = number_of_items
        self.lead_times = list(lead_times)
        self.holding_costs = list(holding_costs)
        self.backorder_allowances = list(backorder_allowances)
        self.review_horizons = list(review_horizons)
        self.penalty_costs = list(penalty_costs)
        self.base_stock_level = list(base_stock_level)

        demand_dists: list[DiscreteDist] = []
        demand_samplers: list[AliasSampler] = []
        for idx in range(number_of_customers * number_of_items):
            dist = (
                DiscreteDist.geometric(demand_rates[idx])
                if high_demand_variance[idx] == 1
                else DiscreteDist.poisson(demand_rates[idx])
            )
            demand_dists.append(dist)
            demand_samplers.append(dist.alias_sampler())
        self.demand_dists = demand_dists
        self.demand_samplers = demand_samplers

        self.benchmark_rationing_action = benchmark_rationing_action
        self.num_actions = total_rationing_actions
        self.horizon_type = HorizonType.INFINITE

    def _descending_order(self, keys: list[int]) -> list[int]:
        """Stable descending-order permutation of range(len(keys)) by key,
        ties broken by ascending index — an O(n^2) selection pass standing in
        for the C++ std::sort(..., comparator) calls, since DynaML compiled
        code has no lambdas / no sorted(key=...)."""
        n = len(keys)
        used: list[bool] = [False] * n
        order: list[int] = [0] * n
        for pos in range(n):
            best = -1
            best_key = 0
            for idx in range(n):
                if not used[idx]:
                    if best == -1 or keys[idx] > best_key:
                        best = idx
                        best_key = keys[idx]
            used[best] = True
            order[pos] = best
        return order

    def get_initial_state(self, context: TrajectoryContext) -> State:
        pipelines: list[FifoQueue] = []
        inventory_position: list[int] = [0] * self.number_of_items
        for i in range(self.number_of_items):
            q = FifoQueue()
            q.reserve(self.lead_times[i] + 1)
            q.fill(self.lead_times[i])
            q[0] = self.base_stock_level[i]
            pipelines.append(q)
            inventory_position[i] = self.base_stock_level[i]

        time_remaining: list[int] = [0] * self.number_of_customers
        for c in range(self.number_of_customers):
            time_remaining[c] = self.review_horizons[c]

        return State(
            pipelines=pipelines,
            backorder=[0] * (self.number_of_customers * self.number_of_items),
            inventory_position=inventory_position,
            current_demand=[0] * (self.number_of_customers * self.number_of_items),
            current_allocation=[0] * (self.number_of_customers * self.number_of_items),
            cumulative_backorder=[0.0] * self.number_of_customers,
            time_remaining=time_remaining,
            last_rationing_action=self.benchmark_rationing_action,
            period=0,
            category=StateCategory.AWAIT_ACTION,
        )

    def modify_state_with_action(self, state: State, context: TrajectoryContext, action: int) -> None:
        # Ordering is STATIC and independent of `action`: `action` only picks
        # the rationing rule used later, once demand is realised. This mirrors
        # docs/paper_improved_complete.tex "Ordering Decision (static)" /
        # "On-Hand Inventory Before Allocation" / "Inventory Position".
        for i in range(self.number_of_items):
            on_hand = state.pipelines[i][0]
            on_the_way = 0
            for j in range(1, len(state.pipelines[i])):
                on_the_way += state.pipelines[i][j]
            backorder_sum = 0
            for c in range(self.number_of_customers):
                backorder_sum += state.backorder[c * self.number_of_items + i]
            ip = on_hand + on_the_way - backorder_sum

            to_order = self.base_stock_level[i] - ip
            order_qty = to_order if to_order > 0 else 0
            state.pipelines[i].push_back(order_qty)
            state.inventory_position[i] = ip + order_qty

        state.last_rationing_action = action
        state.category = StateCategory.AWAIT_EVENT

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        state.period += 1

        for c in range(self.number_of_customers):
            for i in range(self.number_of_items):
                idx = c * self.number_of_items + i
                state.current_demand[idx] = self.demand_samplers[idx].sample(context.rng)
                # Action 3 below writes allocations via `+=`, so this must be
                # zeroed every period (matches the C++ code's explicit
                # `current_allocation.assign(..., 0)` at this same point) —
                # every other branch reassigns every (c, i) pair unconditionally
                # and would silently tolerate a missing reset, but action 3 would
                # accumulate stale allocation across periods without it.
                state.current_allocation[idx] = 0

        # Action 1 (SLA-gap myopic) priority order: customers furthest past
        # their allowance served first. Computed once per period (not per
        # item), matching the C++ code.
        exceed_keys: list[int] = [0] * self.number_of_customers
        for c in range(self.number_of_customers):
            exceed = state.cumulative_backorder[c] - float(self.backorder_allowances[c])
            exceed_keys[c] = int(exceed) if exceed > 0.0 else 0
        priority_order = self._descending_order(exceed_keys)

        # Each customer's total backorder across items as of the start of
        # this period (BO_c(t) = sum_i BO_{c,i}(t)) — used by action 3.
        prior_backorder_total: list[int] = [0] * self.number_of_customers
        for c in range(self.number_of_customers):
            total = 0
            for i in range(self.number_of_items):
                total += state.backorder[c * self.number_of_items + i]
            prior_backorder_total[c] = total

        period_stockouts: list[int] = [0] * self.number_of_customers

        for i in range(self.number_of_items):
            on_hand = state.pipelines[i].pop_front()  # OH'_i(t)

            owed: list[int] = [0] * self.number_of_customers
            total_owed = 0
            for c in range(self.number_of_customers):
                d = state.current_demand[c * self.number_of_items + i]
                owed[c] = state.backorder[c * self.number_of_items + i] + d
                total_owed += owed[c]

            needs_rationing = total_owed > on_hand
            available = on_hand if on_hand > 0 else 0

            if (not needs_rationing) or available >= total_owed:
                # Full coverage: no allocation policy discretion needed.
                for c in range(self.number_of_customers):
                    state.current_allocation[c * self.number_of_items + i] = owed[c]
                    state.backorder[c * self.number_of_items + i] = 0
            elif state.last_rationing_action == 2:
                # Proportional (largest-remainder method).
                served: list[int] = [0] * self.number_of_customers
                handed_out = 0
                for c in range(self.number_of_customers):
                    if total_owed > 0:
                        served[c] = (available * owed[c]) // total_owed
                    handed_out += served[c]
                remainder = available - handed_out
                remainder_keys: list[int] = [0] * self.number_of_customers
                for c in range(self.number_of_customers):
                    if total_owed > 0:
                        remainder_keys[c] = (available * owed[c]) % total_owed
                frac_order = self._descending_order(remainder_keys)
                pos = 0
                while pos < self.number_of_customers and remainder > 0:
                    c = frac_order[pos]
                    if served[c] < owed[c]:
                        served[c] += 1
                        remainder -= 1
                    pos += 1
                for c in range(self.number_of_customers):
                    stockout = owed[c] - served[c]
                    state.current_allocation[c * self.number_of_items + i] = served[c]
                    state.backorder[c * self.number_of_items + i] = stockout
                    period_stockouts[c] += stockout
            elif state.last_rationing_action == 3:
                # Cost-based greedy: allocate unit-by-unit to whoever is
                # about to breach their allowance (exploratory, outside the
                # paper's scope, kept for parity with the C++ model).
                not_served: list[int] = [0] * self.number_of_customers
                for c in range(self.number_of_customers):
                    not_served[c] = owed[c]
                unit = 0
                while unit < available:
                    best_c = -1
                    best_cost = -1.0
                    for c in range(self.number_of_customers):
                        if not_served[c] > 0:
                            prior_c = (prior_backorder_total[c]
                                       - state.backorder[c * self.number_of_items + i]) + period_stockouts[c]
                            total_withheld = prior_c + not_served[c]
                            marginal_cost = self.penalty_costs[c] if total_withheld > self.backorder_allowances[c] else 0.0
                            if marginal_cost > best_cost:
                                best_cost = marginal_cost
                                best_c = c
                    if best_c >= 0:
                        not_served[best_c] -= 1
                        state.current_allocation[best_c * self.number_of_items + i] += 1
                    unit += 1
                for c in range(self.number_of_customers):
                    state.backorder[c * self.number_of_items + i] = not_served[c]
                    period_stockouts[c] += not_served[c]
            else:
                # Action 0 (FCFS, natural index order) or action 1 (SLA-gap,
                # priority_order): serve sequentially against backlog + new demand.
                remaining = available
                for pos in range(self.number_of_customers):
                    c = priority_order[pos] if state.last_rationing_action == 1 else pos
                    served_c = min(remaining, owed[c])
                    remaining -= served_c
                    stockout = owed[c] - served_c
                    state.current_allocation[c * self.number_of_items + i] = served_c
                    state.backorder[c * self.number_of_items + i] = stockout
                    period_stockouts[c] += stockout

            period_allocation = 0
            for c in range(self.number_of_customers):
                period_allocation += state.current_allocation[c * self.number_of_items + i]
            new_on_hand = on_hand - period_allocation
            state.inventory_position[i] -= period_allocation

            # Holding cost: h_i * OH_i(t+1), on inventory remaining AFTER
            # allocation (docs/paper_improved_complete.tex, Step 6).
            if new_on_hand > 0:
                context.cumulative_cost += float(new_on_hand) * self.holding_costs[i]

            # `state.pipelines[i][0]` post-pop is the slot that will be read
            # as OH'_i(t+1) next period; folding new_on_hand into it here is
            # what keeps that identity true (see FifoQueue / lost_sales.py).
            state.pipelines[i][0] += new_on_hand

        for c in range(self.number_of_customers):
            new_bo_c = 0
            for i in range(self.number_of_items):
                new_bo_c += state.backorder[c * self.number_of_items + i]
            state.cumulative_backorder[c] += float(new_bo_c)

            state.time_remaining[c] -= 1
            if state.time_remaining[c] == 0:
                # SLA penalty: p_c * max(0, BO-bar_c(t) - beta_c), inclusive
                # of this period's own backorder (already folded into
                # cumulative_backorder[c] above).
                exceeded = state.cumulative_backorder[c] - float(self.backorder_allowances[c])
                if exceeded > 0.0:
                    context.cumulative_cost += exceeded * self.penalty_costs[c]
                state.cumulative_backorder[c] = 0.0
                state.time_remaining[c] = self.review_horizons[c]

        state.category = StateCategory.AWAIT_ACTION
        context.time_elapsed += 1

    def write_action_validity(self, state: State, valid: Validity) -> None:
        # All four rationing rules are always feasible (matches the C++
        # model's IsAllowedAction, which always returns true).
        for a in range(self.num_actions):
            valid.set(a, True)


@const_dataclass(slots=True)
class BaseStockPolicy:
    """Constant-action policy: always plays the same fixed rationing rule
    (ordering itself is static and handled inside the MDP regardless of
    action). Ported from policies.cpp's BaseStockPolicy — the benchmark/
    rollout policy used to seed DCL, not a learned policy."""
    mdp: MultiCustomerSlaMDP
    action: int = 0  # 0 = FCFS

    def get_action(self, state: State) -> int:
        return self.action


@const_dataclass(slots=True)
class GreedyDynamicPolicy:
    """Rule-based dynamic policy: switches rationing rule based on how close
    each customer's cumulative backorder is to its numeric backorder
    allowance (beta_c) — cost-based greedy (action 3) once any customer has
    exceeded it, SLA-gap myopic (action 1) once any customer is within 80% of
    it, otherwise FCFS (action 0) since there is no SLA pressure yet. Ported
    from policies.cpp's GreedyDynamicPolicy — the "smart" rule-based
    benchmark, as opposed to BaseStockPolicy's constant action."""
    mdp: MultiCustomerSlaMDP
    # Kept for config parity with the C++ class's constructor field, but
    # NOTE: it is dead in the branching logic below, exactly as in the
    # original — every branch of the if/elif/else reassigns the action
    # regardless of this value, so it never actually affects get_action.
    service_level_policy: int = 0

    def get_action(self, state: State) -> int:
        any_near_limit = False
        any_exceeds_limit = False
        for c in range(self.mdp.number_of_customers):
            allowance = float(self.mdp.backorder_allowances[c])
            exceed_amount = state.cumulative_backorder[c] - allowance
            if exceed_amount > 0.0:
                any_exceeds_limit = True
            elif state.cumulative_backorder[c] > 0.8 * allowance:
                any_near_limit = True

        if any_exceeds_limit:
            return 3
        elif any_near_limit:
            return 1
        else:
            return 0
