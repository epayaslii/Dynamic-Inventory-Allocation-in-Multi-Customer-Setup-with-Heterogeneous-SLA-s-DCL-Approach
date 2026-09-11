"""
Rule-based rationing policies for MultiCustomerSlaMDP.

Ported from src/lib/models/multi_customer_sla/policies.{h,cpp}. The C++
policies pick, once per period, WHICH of four fixed rationing rules to run
(BaseStockPolicy: always the same rule; GreedyDynamicPolicy: switches rule
based on SLA pressure). That framing doesn't survive the move to this
model's action encoding, where a "rule" isn't a period-level choice anymore
-- it's applied unit by unit (see mdp.py's module docstring). So each C++
rule below is translated into a get_action(state) that answers "who gets
THIS unit of the item currently being rationed" (state.alloc_item /
state.alloc_owed), and GreedyDynamicPolicy is translated into switching
which of those three per-unit rules currently applies, using the same
SLA-pressure test as the C++ version.
"""
from dynaplex.modelling import const_dataclass

from mdp import MultiCustomerSlaMDP, State


@const_dataclass(slots=True)
class FCFSPolicy:
    """At each unit-decision, serves the lowest-index customer still owed a
    unit of the item under rationing -- never HOLDs. The natural
    unit-by-unit translation of first-come-first-served rationing
    (BaseStockPolicy's action 0 in the C++ model)."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        for c in range(self.mdp.number_of_customers):
            if state.alloc_owed[c] > 0:
                return c
        return self.mdp.number_of_customers  # unreachable: AWAIT_ACTION implies some owed[c] > 0


@const_dataclass(slots=True)
class SlaGapPolicy:
    """At each unit-decision, serves whichever eligible customer is furthest
    past their SLA backorder allowance (largest cumulative_backorder[c] -
    beta_c) -- a myopic, SLA-aware rationing heuristic; never HOLDs.
    Translation of the C++ model's action 1 (backorder-gap myopic)."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        best_c = -1
        best_gap = 0.0
        for c in range(self.mdp.number_of_customers):
            if state.alloc_owed[c] > 0:
                gap = state.cumulative_backorder[c] - float(self.mdp.backorder_allowances[c])
                if best_c == -1 or gap > best_gap:
                    best_c = c
                    best_gap = gap
        if best_c == -1:
            return self.mdp.number_of_customers  # unreachable, see FCFSPolicy
        return best_c


@const_dataclass(slots=True)
class CostGreedyPolicy:
    """At each unit-decision, serves whichever eligible customer has the
    largest projected SLA-penalty exposure if this unit is withheld:
    cumulative_backorder[c] (prior periods this horizon) + this period's
    running total across items (already-finalized items' new backorder for
    items other than the one under rationing, plus the live remaining need
    -- alloc_owed[c] -- for the item currently being decided) minus
    beta_c. Falls back to the lowest eligible customer index if nobody is
    currently projected to breach. Translation of the C++ model's action 3
    (cost-based greedy); not yet-decided items later in this same period
    aren't known yet, so this is a running projection, not a final one --
    the same limitation the C++ per-period version had."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        best_c = -1
        best_cost = -1.0
        for c in range(self.mdp.number_of_customers):
            if state.alloc_owed[c] > 0:
                total_owed_now = state.alloc_owed[c]
                for i2 in range(self.mdp.number_of_items):
                    if i2 != state.alloc_item:
                        total_owed_now += state.backorder[c * self.mdp.number_of_items + i2]
                projected = (
                    state.cumulative_backorder[c]
                    + float(total_owed_now)
                    - float(self.mdp.backorder_allowances[c])
                )
                marginal_cost = self.mdp.penalty_costs[c] if projected > 0.0 else 0.0
                if marginal_cost > best_cost:
                    best_cost = marginal_cost
                    best_c = c
        if best_c == -1:
            return self.mdp.number_of_customers  # unreachable, see FCFSPolicy
        return best_c


@const_dataclass(init=False, slots=True)
class GreedyDynamicPolicy:
    """Rule-based dynamic policy: switches which per-unit rationing rule
    applies based on how close each customer's cumulative backorder is to
    its numeric backorder allowance (beta_c) -- CostGreedyPolicy once any
    customer has exceeded it, SlaGapPolicy once any customer is within 80%
    of it, otherwise FCFSPolicy since there is no SLA pressure yet. Ported
    from policies.cpp's GreedyDynamicPolicy -- the "smart" rule-based
    benchmark, as opposed to the single fixed rule of FCFSPolicy /
    SlaGapPolicy / CostGreedyPolicy alone. The switching test itself is
    unchanged from the C++ version; only what it switches BETWEEN changed
    (a per-unit customer choice, not a whole-period rule index)."""
    mdp: MultiCustomerSlaMDP
    fcfs: FCFSPolicy
    sla_gap: SlaGapPolicy
    cost_greedy: CostGreedyPolicy

    def __init__(self, mdp: MultiCustomerSlaMDP) -> None:
        # NOTE: __init__ is never called by the DynaML compiler; full CPython is allowed here.
        self.mdp = mdp
        self.fcfs = FCFSPolicy(mdp=mdp)
        self.sla_gap = SlaGapPolicy(mdp=mdp)
        self.cost_greedy = CostGreedyPolicy(mdp=mdp)

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
            return self.cost_greedy.get_action(state)
        elif any_near_limit:
            return self.sla_gap.get_action(state)
        else:
            return self.fcfs.get_action(state)
