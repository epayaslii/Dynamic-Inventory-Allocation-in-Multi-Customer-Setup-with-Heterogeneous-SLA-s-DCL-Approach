"""
Multi-customer, multi-item inventory allocation under heterogeneous SLAs.

A DynaML implementation of docs/paper_improved_complete.tex ("Dynamic Inventory
Allocation in Multi-Customer Systems with Heterogeneous Service Level
Agreements"). The number of customers |C| and items |I|, the lead time tau and
the review horizon T are all constructor arguments; nothing below is specific
to a particular instance size.

Notation (paper symbol -> code)
-------------------------------
    OH_it                 state.on_hand[i]        on-hand at the START of period t
    q_{i,t-tau..t-1}      state.pipeline[i]       tau orders in transit, oldest first
                                                  (front = q_{i,t-tau}, arrives in Step 1)
    b_cit                 state.backorder[c*|I|+i] owed at the START of period t
    d_cit                 state.demand[c*|I|+i]    demand observed in Step 3
    a_cit                 state.allocation[c*|I|+i] allocation (built up during Step 4)
    B_{c,t-1}(T)          state.cum_backorder[c]   cumulative backorder in the horizon,
                                                  END-of-period counts of periods before t
    R_{t-1}(T)            state.time_remaining     periods left in the horizon, before t
    beta_c, p_c, h_i, S_i backorder_allowances[c], penalty_costs[c], holding_costs[i],
                          base_stock_level[i]
    tau, T                lead_time, review_horizon
    OH'_it = OH_it + q_{i,t-tau}  is not stored; it is computed where needed.
Customer-item pairs are flattened as c * |I| + i.

Period workflow (paper Sec. "Period Workflow")
----------------------------------------------
modify_state_with_event runs Steps 1-3:
    1. the order q_{i,t-tau} arrives: OH'_it = OH_it + q_{i,t-tau}
    2. IP_it = OH'_it + sum of the other tau-1 pipeline orders - sum_c b_cit,
       and the order q_it = (S_i - IP_it)^+ is placed (stored in state.order)
    3. demands d_cit are sampled
and then starts Step 4 (allocation). Steps 5 and 6 (state update and cost) run in
_finalize_period once every item is allocated. Until then the state variables
on_hand, pipeline and backorder keep their START-of-period values, exactly as in
the paper's definition of s_t; the allocation made so far lives in state.allocation.

Backorder convention (paper Sec. "Review Horizon Statistics")
-------------------------------------------------------------
Backorders are counted at the END of each period, after its allocation. The
backorder of customer c at the end of period t is e_ct = sum_i (b_cit + d_cit -
a_cit) (which equals sum_i b_{ci,t+1}). Within a horizon the statistics are
    B_ct = B_{c,t-1} + e_ct                      (accumulated)
    R_t  = R_{t-1} - 1
and at the horizon's last period (R_t = 0) the penalty is
    p_c * max(0, B_ct - beta_c),
so the last period's own end-of-period backorder e_ct counts on its own as the last
term of the accumulated sum. The statistics then restart at (0, ..., 0, T). A unit
that is backordered at the end of k periods of the horizon counts k times
(unit-periods). Individual backorders b_cit persist across horizon boundaries.

The action
----------
DynaML actions are a single scalar int, while the paper's decision is the matrix
a_cit. Following DynaPlex's own bundled models (e.g. binpacking), the matrix is
built from a SEQUENCE of scalar decisions. For an item whose stock cannot cover
what is owed (sum_c (b_cit + d_cit) > OH'_it) the policy is asked, once per
available unit, "who gets this unit?":
    action c in {0, ..., |C|-1}   give one unit of the item to customer c
    action |C|                    HOLD: keep the unit in the warehouse
so num_actions = |C| + 1. The allocation is the tally of these choices, and every
allocation in the paper's feasible set
    {a : a_ci <= b_cit + d_cit,  sum_c a_ci <= OH'_it}
can be produced. Customer c is a valid action iff it is still owed a unit of the
item (state.alloc_owed[c] > 0); HOLD is always valid. The policy is not asked when
the stock covers everything owed (the code allocates in full) or when there is no
stock. This restricts the policies considered to those that allocate in full when
supply covers everything owed; holding stock back can still be optimal in general
(e.g. to reserve it for a customer close to its allowance), so the restriction is
not lossless.

Scratch state (not in the paper): `order`, `alloc_item`, `alloc_owed`,
`alloc_units`, and `allocation` exist only because the allocation is decomposed
into unit decisions. They are exposed to the policy and the network through the
featurizer, otherwise consecutive unit decisions would look identical.

DynaML restrictions respected here: no Python builtins such as max/min/sum/list
inside compiled methods (plain loops instead), and no chained comparisons.
"""
from dataclasses import dataclass
from typing import Final

from dynaplex.modelling import (
    AliasSampler,
    const_dataclass,
    DiscreteDist,
    Dtype,
    Featurizer,
    FifoQueue,
    GlobalStateWriter,
    HorizonType,
    StateCategory,
    TensorSpec,
    TrajectoryContext,
    Validity,
    featurizer,
)


@dataclass(slots=True)
class State:
    """Start-of-period state variables of the paper, plus scratch fields for the
    unit-by-unit allocation. See the module docstring for the symbol mapping."""
    on_hand: list[int]
    pipeline: list[FifoQueue]
    backorder: list[int]
    demand: list[int]
    cum_backorder: list[int]
    time_remaining: int
    # scratch
    order: list[int]
    allocation: list[int]
    alloc_item: int
    alloc_owed: list[int]
    alloc_units: int
    period: int
    category: StateCategory


@const_dataclass(init=False, slots=True)
class MultiCustomerSlaMDP:
    """
    Parameters (lists indexed as in the paper; demand_rates is flattened c*|I|+i):
        number_of_customers  |C|
        number_of_items      |I|
        lead_time            tau >= 1
        review_horizon       T >= 1, shared by all customers
        holding_costs        h_i  (size |I|)
        backorder_allowances beta_c (size |C|), in unit-periods
        penalty_costs        p_c  (size |C|)
        demand_rates         lambda_ci (size |C|*|I|), the mean demand per period
        base_stock_level     S_i  (size |I|), fixed; only the allocation is optimized
        high_demand_variance 1 for a Geometric demand with the same mean,
                             0 (default) for Poisson, per customer-item pair
    """
    number_of_customers: int
    number_of_items: int
    lead_time: int
    review_horizon: int
    holding_costs: list[float]
    backorder_allowances: list[int]
    penalty_costs: list[float]
    base_stock_level: list[int]
    demand_dists: list[DiscreteDist]
    demand_samplers: list[AliasSampler]
    num_actions: int
    horizon_type: HorizonType

    def __init__(
        self,
        number_of_customers: int,
        number_of_items: int,
        lead_time: int,
        review_horizon: int,
        holding_costs: list[float],
        backorder_allowances: list[int],
        penalty_costs: list[float],
        demand_rates: list[float],
        base_stock_level: list[int],
        high_demand_variance: list[int] | None = None,
    ) -> None:
        # NOTE: __init__ is never called by the DynaML compiler; full CPython is allowed here.
        assert number_of_customers >= 1 and number_of_items >= 1
        assert lead_time >= 1, "lead_time (tau) must be >= 1"
        assert review_horizon >= 1
        assert len(holding_costs) == number_of_items
        assert len(base_stock_level) == number_of_items
        assert len(backorder_allowances) == number_of_customers
        assert len(penalty_costs) == number_of_customers
        assert all(b >= 0 for b in backorder_allowances)
        assert all(s >= 0 for s in base_stock_level)
        assert len(demand_rates) == number_of_customers * number_of_items

        variance_flags = (
            list(high_demand_variance) if high_demand_variance is not None
            else [0] * (number_of_customers * number_of_items)
        )
        assert len(variance_flags) == number_of_customers * number_of_items

        self.number_of_customers = number_of_customers
        self.number_of_items = number_of_items
        self.lead_time = lead_time
        self.review_horizon = review_horizon
        self.holding_costs = list(holding_costs)
        self.backorder_allowances = list(backorder_allowances)
        self.penalty_costs = list(penalty_costs)
        self.base_stock_level = list(base_stock_level)

        demand_dists: list[DiscreteDist] = []
        demand_samplers: list[AliasSampler] = []
        for idx in range(number_of_customers * number_of_items):
            dist = (
                DiscreteDist.geometric(demand_rates[idx])
                if variance_flags[idx] == 1
                else DiscreteDist.poisson(demand_rates[idx])
            )
            demand_dists.append(dist)
            demand_samplers.append(dist.alias_sampler())
        self.demand_dists = demand_dists
        self.demand_samplers = demand_samplers

        self.num_actions = number_of_customers + 1  # + 1 for HOLD
        self.horizon_type = HorizonType.INFINITE

    # -- lifecycle -------------------------------------------------------

    def get_initial_state(self, context: TrajectoryContext) -> State:
        # Paper, initial state: OH_i1 = S_i, empty pipeline, no backorders,
        # and period 1 starts a review horizon (B = 0, R = T).
        on_hand: list[int] = []
        pipeline: list[FifoQueue] = []
        for i in range(self.number_of_items):
            on_hand.append(self.base_stock_level[i])
            q = FifoQueue()
            q.reserve(self.lead_time + 1)
            q.fill(self.lead_time)  # tau zero orders
            pipeline.append(q)

        n_pairs = self.number_of_customers * self.number_of_items
        return State(
            on_hand=on_hand,
            pipeline=pipeline,
            backorder=[0] * n_pairs,
            demand=[0] * n_pairs,
            cum_backorder=[0] * self.number_of_customers,
            time_remaining=self.review_horizon,
            order=[0] * self.number_of_items,
            allocation=[0] * n_pairs,
            alloc_item=0,
            alloc_owed=[0] * self.number_of_customers,
            alloc_units=0,
            period=0,
            category=StateCategory.AWAIT_EVENT,
        )

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        # Steps 1-2: receive the arriving order, compute IP and place the order.
        # The order depends only on the state, not on the allocation, so it is
        # computed here, before demand is observed.
        for i in range(self.number_of_items):
            on_hand_before_allocation = state.on_hand[i] + state.pipeline[i][0]  # OH'_it
            pipeline_rest = 0
            for j in range(1, self.lead_time):
                pipeline_rest += state.pipeline[i][j]
            backorder_sum = 0
            for c in range(self.number_of_customers):
                backorder_sum += state.backorder[c * self.number_of_items + i]
            ip = on_hand_before_allocation + pipeline_rest - backorder_sum

            to_order = self.base_stock_level[i] - ip
            state.order[i] = to_order if to_order > 0 else 0

        # Step 3: demand for every customer-item pair.
        for c in range(self.number_of_customers):
            for i in range(self.number_of_items):
                idx = c * self.number_of_items + i
                state.demand[idx] = self.demand_samplers[idx].sample(context.rng)
                state.allocation[idx] = 0

        # Step 4: allocate item by item.
        self._advance_items(state, context, 0)

    def modify_state_with_action(self, state: State, context: TrajectoryContext, action: int) -> None:
        assert action >= 0 and action <= self.number_of_customers, f"invalid action: {action}"

        i = state.alloc_item
        if action < self.number_of_customers:
            state.allocation[action * self.number_of_items + i] += 1
            state.alloc_owed[action] -= 1
        # action == number_of_customers: HOLD, nothing is assigned.
        state.alloc_units -= 1

        total_remaining = 0
        for c in range(self.number_of_customers):
            total_remaining += state.alloc_owed[c]

        if state.alloc_units <= 0 or total_remaining == 0:
            self._advance_items(state, context, i + 1)
        else:
            state.category = StateCategory.AWAIT_ACTION

    def write_action_validity(self, state: State, valid: Validity) -> None:
        for c in range(self.number_of_customers):
            valid.set(c, state.alloc_owed[c] > 0)
        valid.set(self.number_of_customers, True)

    # -- internals ---------------------------------------------------------

    def _advance_items(self, state: State, context: TrajectoryContext, start_item: int) -> None:
        """Settle items start_item, start_item+1, ... in order. An item whose
        available supply OH'_it covers everything owed, sum_c (b_cit + d_cit), is
        allocated in full without asking the policy; an item with no stock gets
        nothing. The first item that is genuinely scarce hands control to the
        policy (AWAIT_ACTION) and returns; modify_state_with_action resumes the
        loop at the next item. Once all items are settled the period is finalized."""
        i = start_item
        while i < self.number_of_items:
            on_hand_before_allocation = state.on_hand[i] + state.pipeline[i][0]  # OH'_it

            owed: list[int] = [0] * self.number_of_customers
            total_owed = 0
            for c in range(self.number_of_customers):
                idx = c * self.number_of_items + i
                owed[c] = state.backorder[idx] + state.demand[idx]
                total_owed += owed[c]

            if total_owed <= on_hand_before_allocation:
                for c in range(self.number_of_customers):
                    state.allocation[c * self.number_of_items + i] = owed[c]
                i += 1
            elif on_hand_before_allocation <= 0:
                # Nothing to hand out, so there is no decision to make.
                for c in range(self.number_of_customers):
                    state.allocation[c * self.number_of_items + i] = 0
                i += 1
            else:
                state.alloc_item = i
                state.alloc_owed = owed
                state.alloc_units = on_hand_before_allocation
                state.category = StateCategory.AWAIT_ACTION
                return

        self._finalize_period(state, context)

    def _finalize_period(self, state: State, context: TrajectoryContext) -> None:
        # Step 5, on-hand inventory and pipeline; Step 6, holding cost.
        for i in range(self.number_of_items):
            on_hand_before_allocation = state.on_hand[i] + state.pipeline[i].pop_front()  # OH'_it
            assigned = 0
            for c in range(self.number_of_customers):
                assigned += state.allocation[c * self.number_of_items + i]
            new_on_hand = on_hand_before_allocation - assigned  # OH_{i,t+1}
            state.on_hand[i] = new_on_hand
            state.pipeline[i].push_back(state.order[i])
            context.cumulative_cost += float(new_on_hand) * self.holding_costs[i]

        # Step 5, backorders: b_{ci,t+1} = b_cit + d_cit - a_cit.
        for idx in range(self.number_of_customers * self.number_of_items):
            state.backorder[idx] = state.backorder[idx] + state.demand[idx] - state.allocation[idx]

        # Step 5, review horizon statistics: B_ct = B_{c,t-1} + e_ct, where e_ct is
        # customer c's backorder at the END of this period (after its allocation).
        for c in range(self.number_of_customers):
            end_of_period_backorder = 0
            for i in range(self.number_of_items):
                end_of_period_backorder += state.backorder[c * self.number_of_items + i]
            state.cum_backorder[c] += end_of_period_backorder
        state.time_remaining -= 1  # R_t

        # Step 6, SLA penalty at the horizon's last period (R_t = 0); then the
        # statistics restart at (0, ..., 0, T).
        if state.time_remaining == 0:
            for c in range(self.number_of_customers):
                excess = state.cum_backorder[c] - self.backorder_allowances[c]
                if excess > 0:
                    context.cumulative_cost += float(excess) * self.penalty_costs[c]
                state.cum_backorder[c] = 0
            state.time_remaining = self.review_horizon

        state.period += 1
        state.category = StateCategory.AWAIT_EVENT
        context.time_elapsed += 1


@featurizer
@dataclass(slots=True)
class MultiCustomerSlaFeaturizer(Featurizer):
    """
    Features, in this order (the block sizes below are used by network.py):
      1. per item: OH_it, then its tau pipeline orders, oldest first
                                                   |I| * (1 + tau)
      2. b_cit                                      |C| * |I|
      3. d_cit                                      |C| * |I|
      4. B_{c,t-1}(T)                               |C|
      5. R_{t-1}(T)                                 1
      6. beta_c                                     |C|
    Blocks 1 to 6 are the paper's state s_t (dimension |I| + |I|tau + 2|C||I| +
    2|C| + 1). The remaining blocks describe the unit-by-unit allocation in
    progress and have no counterpart in the paper:
      7. one-hot of the item being rationed         |I|
      8. units still owed per customer for it       |C|   (alloc_owed)
      9. units of it still to be decided            1     (alloc_units)
     10. allocation made so far, a_cit              |C| * |I|
    Without blocks 7 to 10 consecutive unit decisions of the same item would be
    indistinguishable to the policy.
    """
    mdp: Final[MultiCustomerSlaMDP]
    v: Final[GlobalStateWriter]

    def write_features(self, state: State) -> None:
        for i in range(self.mdp.number_of_items):
            self.v.append(float(state.on_hand[i]))
            for j in range(self.mdp.lead_time):
                self.v.append(float(state.pipeline[i][j]))

        self.v.extend(state.backorder)
        self.v.extend(state.demand)
        self.v.extend(state.cum_backorder)
        self.v.append(float(state.time_remaining))
        for c in range(self.mdp.number_of_customers):
            self.v.append(float(self.mdp.backorder_allowances[c]))

        for i in range(self.mdp.number_of_items):
            self.v.append(1.0 if i == state.alloc_item else 0.0)
        self.v.extend(state.alloc_owed)
        self.v.append(float(state.alloc_units))
        self.v.extend(state.allocation)

    def spec(self) -> dict:
        n_c = self.mdp.number_of_customers
        n_i = self.mdp.number_of_items
        size = (
            n_i * (1 + self.mdp.lead_time)  # on hand + pipeline
            + n_c * n_i                     # b
            + n_c * n_i                     # d
            + n_c                           # B
            + 1                             # R
            + n_c                           # beta
            + n_i                           # alloc_item one-hot
            + n_c                           # alloc_owed
            + 1                             # alloc_units
            + n_c * n_i                     # allocation so far
        )
        return {"v": TensorSpec(Dtype.float32, (size,))}
