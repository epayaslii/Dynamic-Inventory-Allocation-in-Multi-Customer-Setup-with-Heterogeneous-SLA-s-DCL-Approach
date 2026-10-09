"""
Multi-customer, multi-item inventory allocation under heterogeneous SLAs.

A direct DynaML port of docs/paper_improved_complete.tex ("Dynamic Inventory
Allocation in Multi-Customer Systems with Heterogeneous Service Level
Agreements"). Unlike the earlier prototype on this branch (which exposed a
menu of four fixed rationing HEURISTICS as the action), this implements the
paper's actual decision variable directly: the allocation A_{c,i}(t) itself,
subject only to sum_c A_{c,i}(t) <= OH'_i(t) (paper Sec. "Allocation Decision
(dynamic)"). The learned policy decides who gets each unit of scarce stock.

Why not a single vector-valued action
--------------------------------------
DynaML's MDPProtocol requires `modify_state_with_action(state, context,
action: int) -> None` and `PolicyProtocol.get_action(state) -> int`: actions
are always a single scalar int (verified against the installed `dynaplex`
package, not just the docs site). There is no vector/combinatorial action
type. Following the idiom the framework's own bundled models use to build up
a multi-part decision (see dynaplex.models.binpacking, which places one
weight into one bin per AWAIT_ACTION/AWAIT_EVENT step), a full per-item
allocation is built here from a SEQUENCE of scalar decisions: for an item
under scarcity, the policy is asked, once per unit of available stock,
"who gets this unit?" -- action in {0, ..., |C|-1} names a customer,
action == |C| means HOLD (paper: allocation "may [be] less than what is
available -- a strategic hold -- even when demand could be fully covered").
The resulting A_{c,i}(t) is exactly the tally of per-unit choices, so this
enumerates precisely the feasible set {A : 0 <= A_c <= BO_{c,i}+D_{c,i},
sum_c A_c <= OH'_i(t)} -- not a restriction to a handful of heuristics.

Full coverage needs no decision
--------------------------------
When available supply already covers every customer's backlog and new
demand for an item (sum_c owed_c <= OH'_i(t)), the code allocates in full
without asking the policy anything. This is not just a simplification: it is
WLOG cost-minimizing under this paper's cost structure. Holding back a unit
that somebody needs, when supply allows serving them, can only (a) raise
this period's holding cost (charged on post-allocation on-hand, Step 6) and
(b) create backorder that only ever adds to a customer's cumulative SLA
exposure -- never reduces it. So under full coverage, allocating in full is
always weakly better than any partial hold, for every possible continuation.
The action space (and the HOLD action) is therefore only exercised under
genuine scarcity, where such reservation-for-later tradeoffs are real.

Per-customer review horizons
-------------------------------
The paper's Sec. 3.1 formalizes T and r(t) = ceil(t/T) as a SINGLE, shared
quantity, with "Time Remaining" listed as exactly 1 state feature -- but the
paper's own subject is heterogeneous, customer-specific SLAs (beta_c, p_c
already vary by customer), and this repo's C++/README model already allows
each customer its own review horizon T_k (README: `reviewHorizons`). This
port follows the more general C++/README convention: review_horizon is a
list of length |C| (one T_k per customer), time_remaining is |C| independent
counters, and each customer's cumulative backorder is reset and its SLA
penalty assessed on ITS OWN period boundary, not tied to any other
customer's. The paper's literal single-T model is the special case
T_1 = ... = T_|C|; nothing here prevents recovering it by passing equal
values.

Single, shared lead time
--------------------------
Likewise Sec. 3.1 declares L as one scalar ("replenishment lead time"), not
one per item, and the state-dimension formula |I|*L uses it uniformly. This
port assumes L >= 1 (L == 0 -- same-period replenishment -- would need a
different FifoQueue convention below and is not implemented).

SLA penalty timing: backorders are counted at the BEGINNING of each period
--------------------------------------------------------------------------
BO_c(t) is the backlog owed to customer c ENTERING period t, i.e. before
period t's demand is added and before period t's allocation (paper Sec.
3.3.3). The cumulative sum BObar_c(t) adds up BO_c(j) over the EARLIER periods
j of the current review horizon and does NOT include period t itself:
    BObar_c(t)   = sum_{j = (r-1)T+1}^{t-1} BO_c(j)       (0 in a horizon's
                                                            first period)
    BObar_c(t+1) = BObar_c(t) + BO_c(t)    inside a horizon
                 = 0                       when period t+1 starts a horizon
and the penalty at a horizon's last period t = rT is
    p_c * max(0, BObar_c(t) + BO_c(t) - beta_c).
So the backorder that period t's OWN allocation creates, BO_c(t+1), is not
charged in the horizon containing t; it is the next period's beginning-of-
period backorder. In particular the backorder created by a horizon's last
period counts towards the NEXT horizon.

Implementation: modify_state_with_event records each customer's total entering
backlog in state.entering_backorder before anything in the period changes
state.backorder; _finalize_period adds that to cumulative_backorder[c], then
runs the per-customer boundary/penalty check. Adding the entering backlog and
then testing the boundary is exactly the two cases above. (An earlier version
of this file instead added the backorder left AFTER the period's allocation,
i.e. BO_c(t+1); the paper's text was ambiguous and has since been settled in
favour of the beginning-of-period reading above.)

On-hand representation (OH'_i(t), not separately OH_i(t))
------------------------------------------------------------
The paper's own state dimension count folds OH'_i(t) = OH_i(t) + Q_i(t-L)
back out of the total ("it is not an independent piece of state ... given
its own symbol because it recurs"), and the project's README already
documents this simplification for the C++ model: "there is no
separately-tracked 'before allocation' state variable." Consistent with
that, and with how dynaplex's own bundled lost_sales.py model represents
inventory (`state.on_hand += state.pipeline.pop_front()`, arrival folded
straight in), the FifoQueue used below stores OH'_i(t) at slot 0 and the
remaining L-1 in-transit orders at slots 1..L-1 -- |I|*L numbers total per
item, not |I|*(L+1) -- since OH_i(t) and Q_i(t-L) are never used separately
anywhere in the model, only their sum.
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
    """
    pipelines[i]: FIFO queue of item i's available stock + pipeline orders,
      front first. pipelines[i][0] = OH'_i(t) (on-hand INCLUDING this
      period's arriving order, BEFORE this period's allocation) while item i
      has not yet been popped for processing this period; once popped (item
      i's allocation has been decided, in full or unit by unit), slot 0
      instead holds the order that will be OH'_i(t+1)'s arrival component,
      pending the leftover stock `_finalize_item` folds in.
      pipelines[i][1:] = the remaining L-1 orders still in transit.
    backorder[c*|I|+i]: BO_{c,i}(t), flattened per (customer, item) --
      updated for item i once its allocation for this period is finalized.
    current_demand / current_allocation[c*|I|+i]: this period's D_{c,i}(t)
      (fixed once sampled) / A_{c,i}(t) (grows as unit-decisions are made),
      flattened the same way.
    cumulative_backorder[c]: BObar_c(t), the sum of beginning-of-period
      backorders of EARLIER periods in the horizon (excludes period t); reset
      for customer c at ITS OWN review-horizon boundary (see module
      docstring: review horizons are per-customer, T_k).
    entering_backorder[c]: BO_c(t), customer c's total backlog entering the
      current period, recorded before the period's allocation changes
      `backorder`; added to cumulative_backorder when the period finalizes.
    time_remaining[c]: T_rem for customer c -- one independent counter per
      customer, each cycling through its own review_horizon[c].
    alloc_item / alloc_owed / alloc_on_hand / alloc_available: scratch state
      for the item currently being rationed unit by unit (meaningful only
      while category == AWAIT_ACTION). alloc_owed[c] = BO_{c,i}+D_{c,i}
      remaining unassigned for customer c on item alloc_item; alloc_on_hand
      = OH'_i(t) for that item (fixed for the item's duration, used to
      recover the leftover once decisions finish); alloc_available = how
      many of alloc_on_hand's units are still undecided.
    """
    pipelines: list[FifoQueue]
    backorder: list[int]
    current_demand: list[int]
    current_allocation: list[int]
    cumulative_backorder: list[float]
    entering_backorder: list[int]
    time_remaining: list[int]
    alloc_item: int
    alloc_owed: list[int]
    alloc_on_hand: int
    alloc_available: int
    period: int
    category: StateCategory


@const_dataclass(init=False, slots=True)
class MultiCustomerSlaMDP:
    """
    Actions: 0 .. number_of_customers - 1 assign the next available unit of
    the item currently being rationed to that customer; action
    number_of_customers HOLDs the unit (see module docstring). Ordering is
    entirely static (base-stock, paper Sec. "Ordering Decision (static)")
    and does not depend on the action at all.
    """
    number_of_customers: int
    number_of_items: int
    lead_time: int
    review_horizon: list[int]
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
        review_horizon: list[int],
        holding_costs: list[float],
        backorder_allowances: list[int],
        penalty_costs: list[float],
        demand_rates: list[float],
        base_stock_level: list[int],
        high_demand_variance: list[int] | None = None,
    ) -> None:
        # NOTE: __init__ is never called by the DynaML compiler; full CPython is allowed here.
        assert number_of_customers >= 1 and number_of_items >= 1
        assert lead_time >= 1, "this port assumes lead_time >= 1 (L == 0 is not yet supported)"
        assert len(review_horizon) == number_of_customers, "one review horizon T_k per customer"
        assert all(t >= 1 for t in review_horizon)
        assert len(holding_costs) == number_of_items
        assert len(base_stock_level) == number_of_items
        assert len(backorder_allowances) == number_of_customers
        assert len(penalty_costs) == number_of_customers
        assert all(b >= 0 for b in backorder_allowances)
        assert len(demand_rates) == number_of_customers * number_of_items

        variance_flags = (
            list(high_demand_variance) if high_demand_variance is not None
            else [0] * (number_of_customers * number_of_items)
        )
        assert len(variance_flags) == number_of_customers * number_of_items

        self.number_of_customers = number_of_customers
        self.number_of_items = number_of_items
        self.lead_time = lead_time
        self.review_horizon = list(review_horizon)
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
        pipelines: list[FifoQueue] = []
        for i in range(self.number_of_items):
            q = FifoQueue()
            q.reserve(self.lead_time + 1)
            q.fill(self.lead_time)
            q[0] = self.base_stock_level[i]
            pipelines.append(q)

        time_remaining: list[int] = []
        for c in range(self.number_of_customers):
            time_remaining.append(self.review_horizon[c])

        n_pairs = self.number_of_customers * self.number_of_items
        return State(
            pipelines=pipelines,
            backorder=[0] * n_pairs,
            current_demand=[0] * n_pairs,
            current_allocation=[0] * n_pairs,
            cumulative_backorder=[0.0] * self.number_of_customers,
            entering_backorder=[0] * self.number_of_customers,
            time_remaining=time_remaining,
            alloc_item=0,
            alloc_owed=[0] * self.number_of_customers,
            alloc_on_hand=0,
            alloc_available=0,
            period=0,
            category=StateCategory.AWAIT_EVENT,
        )

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        # BO_c(t): backlog entering this period, recorded before anything below
        # (allocation, in particular) changes state.backorder.
        for c in range(self.number_of_customers):
            entering = 0
            for i in range(self.number_of_items):
                entering += state.backorder[c * self.number_of_items + i]
            state.entering_backorder[c] = entering

        # Steps 1-2 (paper): static base-stock ordering for every item.
        # Action-independent, so it runs unconditionally here, before any
        # allocation decision -- matching the paper's period workflow, where
        # ordering happens before demand is even observed.
        for i in range(self.number_of_items):
            on_hand_and_arriving = state.pipelines[i][0]  # OH'_i(t)
            pipeline_rest = 0
            for j in range(1, len(state.pipelines[i])):
                pipeline_rest += state.pipelines[i][j]
            backorder_sum = 0
            for c in range(self.number_of_customers):
                backorder_sum += state.backorder[c * self.number_of_items + i]
            ip = on_hand_and_arriving + pipeline_rest - backorder_sum

            to_order = self.base_stock_level[i] - ip
            order_qty = to_order if to_order > 0 else 0
            state.pipelines[i].push_back(order_qty)

        # Step 3: demand realisation for every (customer, item) pair. This
        # is part of s_t, which the allocation policy below is allowed to
        # use -- unlike ordering, allocation is decided AFTER demand.
        for c in range(self.number_of_customers):
            for i in range(self.number_of_items):
                idx = c * self.number_of_items + i
                state.current_demand[idx] = self.demand_samplers[idx].sample(context.rng)
                state.current_allocation[idx] = 0

        # Step 4: allocate item by item. Items with sufficient supply are
        # resolved immediately inside _advance_items; the first item found
        # under scarcity hands control to the policy.
        self._advance_items(state, context, 0)

    def modify_state_with_action(self, state: State, context: TrajectoryContext, action: int) -> None:
        assert action >= 0 and action <= self.number_of_customers, f"invalid action: {action}"

        i = state.alloc_item
        if action < self.number_of_customers:
            c = action
            state.current_allocation[c * self.number_of_items + i] += 1
            state.alloc_owed[c] -= 1
        # action == number_of_customers: HOLD -- nothing to assign.
        state.alloc_available -= 1

        total_remaining = 0
        for c in range(self.number_of_customers):
            total_remaining += state.alloc_owed[c]

        if state.alloc_available <= 0 or total_remaining == 0:
            # This item's rationing is done: whatever wasn't handed to a
            # customer (explicit HOLDs, plus anything left once nobody
            # still needs it) becomes next period's on-hand leftover.
            total_assigned = 0
            for c in range(self.number_of_customers):
                total_assigned += state.current_allocation[c * self.number_of_items + i]
                state.backorder[c * self.number_of_items + i] = state.alloc_owed[c]
            leftover = state.alloc_on_hand - total_assigned
            self._finalize_item(state, context, i, leftover)
            self._advance_items(state, context, i + 1)
        else:
            state.category = StateCategory.AWAIT_ACTION

    def write_action_validity(self, state: State, valid: Validity) -> None:
        # Only customers still owed a unit of the item under rationing are
        # eligible; HOLD is always allowed (paper: allocation may fall
        # short of what's available).
        for c in range(self.number_of_customers):
            valid.set(c, state.alloc_owed[c] > 0)
        valid.set(self.number_of_customers, True)

    # -- internals ---------------------------------------------------------

    def _advance_items(self, state: State, context: TrajectoryContext, start_item: int) -> None:
        """Resolve items start_item, start_item+1, ... in order. Items with
        supply covering owed backlog+demand are settled immediately, with no
        policy call (see module docstring: this is WLOG cost-minimizing).
        The first item found under scarcity hands control to the policy via
        AWAIT_ACTION and returns; modify_state_with_action resumes this loop
        (at i + 1) once that item's rationing is done. Once every item is
        settled, the period is finalized."""
        i = start_item
        while i < self.number_of_items:
            on_hand = state.pipelines[i].pop_front()  # OH'_i(t)

            owed: list[int] = [0] * self.number_of_customers
            total_owed = 0
            for c in range(self.number_of_customers):
                idx = c * self.number_of_items + i
                owed[c] = state.backorder[idx] + state.current_demand[idx]
                total_owed += owed[c]

            if total_owed <= on_hand:
                for c in range(self.number_of_customers):
                    idx = c * self.number_of_items + i
                    state.current_allocation[idx] = owed[c]
                    state.backorder[idx] = 0
                self._finalize_item(state, context, i, on_hand - total_owed)
                i += 1
            elif on_hand <= 0:
                # Nothing at all to hand out: every customer's owed amount
                # becomes backorder, with no decision to make (alloc_available
                # would start at 0 -- entering AWAIT_ACTION here would ask the
                # policy to decide something with zero units to place).
                for c in range(self.number_of_customers):
                    idx = c * self.number_of_items + i
                    state.current_allocation[idx] = 0
                    state.backorder[idx] = owed[c]
                self._finalize_item(state, context, i, 0)
                i += 1
            else:
                state.alloc_item = i
                state.alloc_owed = owed
                state.alloc_on_hand = on_hand
                state.alloc_available = on_hand
                state.category = StateCategory.AWAIT_ACTION
                return

        self._finalize_period(state, context)

    def _finalize_item(self, state: State, context: TrajectoryContext, i: int, leftover: int) -> None:
        # Holding cost (paper Step 6): h_i * OH_i(t+1), charged on stock
        # remaining AFTER this item's allocation.
        if leftover > 0:
            context.cumulative_cost += float(leftover) * self.holding_costs[i]
        # Slot 0 currently holds the order that will arrive at the start of
        # next period (see State docstring); folding leftover in here is
        # what keeps that slot equal to OH'_i(t+1) once next period starts.
        state.pipelines[i][0] += leftover

    def _finalize_period(self, state: State, context: TrajectoryContext) -> None:
        state.period += 1

        # Beginning-of-period accounting (see module docstring): add BO_c(t),
        # the backlog that ENTERED this period, not the backlog left after it.
        for c in range(self.number_of_customers):
            state.cumulative_backorder[c] += float(state.entering_backorder[c])

        # Per-customer review horizons (see module docstring): each
        # customer's SLA is assessed, and its cumulative sum reset, on ITS
        # OWN boundary -- independent of every other customer's.
        for c in range(self.number_of_customers):
            state.time_remaining[c] -= 1
            if state.time_remaining[c] == 0:
                exceeded = state.cumulative_backorder[c] - float(self.backorder_allowances[c])
                if exceeded > 0.0:
                    context.cumulative_cost += exceeded * self.penalty_costs[c]
                state.cumulative_backorder[c] = 0.0
                state.time_remaining[c] = self.review_horizon[c]

        state.category = StateCategory.AWAIT_EVENT
        context.time_elapsed += 1


@featurizer
@dataclass(slots=True)
class MultiCustomerSlaFeaturizer(Featurizer):
    """
    Features, in order:
      - per item: [OH'_i(t)-or-alloc_on_hand, L-1 pipeline-in-transit values]
        (|I| * lead_time numbers -- see module docstring on OH' vs OH)
      - backorder[c,i]                     (|C| * |I|)
      - current_demand[c,i]                (|C| * |I|)
      - cumulative_backorder[c]            (|C|)
      - time_remaining[c]                  (|C|)
      - backorder_allowances[c] (beta_c)   (|C|, constant, from mdp config)
      - one-hot: which item is under rationing right now (alloc_item)
                                            (|I|)
    The last block has no counterpart in the paper's state list: the paper
    does not itself decompose the allocation decision into unit-by-unit
    sub-steps (that is purely this DynaML encoding's doing, see module
    docstring), so nothing else here identifies which item a given
    unit-decision is about. Similarly, time_remaining is |C| numbers (one per
    customer) rather than the paper's literal 1, since this port generalizes
    to per-customer review horizons T_k (see module docstring). Everything
    else mirrors paper Sec. 3.3 exactly.
    """
    mdp: Final[MultiCustomerSlaMDP]
    v: Final[GlobalStateWriter]

    def write_features(self, state: State) -> None:
        for i in range(self.mdp.number_of_items):
            if i == state.alloc_item:
                self.v.append(float(state.alloc_on_hand))
            else:
                self.v.append(float(state.pipelines[i][0]))
            for j in range(1, self.mdp.lead_time):
                self.v.append(float(state.pipelines[i][j]))

        self.v.extend(state.backorder)
        self.v.extend(state.current_demand)
        self.v.extend(state.cumulative_backorder)
        self.v.extend(state.time_remaining)
        for c in range(self.mdp.number_of_customers):
            self.v.append(float(self.mdp.backorder_allowances[c]))
        for i in range(self.mdp.number_of_items):
            self.v.append(1.0 if i == state.alloc_item else 0.0)

    def spec(self) -> dict:
        n_c = self.mdp.number_of_customers
        n_i = self.mdp.number_of_items
        size = (
            n_i * self.mdp.lead_time
            + n_c * n_i  # backorder
            + n_c * n_i  # current_demand
            + n_c        # cumulative_backorder
            + n_c        # time_remaining
            + n_c        # backorder_allowances
            + n_i        # alloc_item one-hot
        )
        return {"v": TensorSpec(Dtype.float32, (size,))}
