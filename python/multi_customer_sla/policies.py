"""
Rule-based allocation policies for MultiCustomerSlaMDP.

The paper prescribes no allocation policy, so these are benchmarks (and the
rollout policy of generation 0 of DCL). Each answers one question, asked once per
available unit of the item i = state.alloc_item that is scarce in period t:

    "Which customer gets the next unit?"

Notation used below (see mdp.py for the full mapping):
    o_c       = state.alloc_owed[c]      units of item i still owed to customer c
                                         (b_cit + d_cit minus what it already got)
    E         = { c : o_c > 0 }          the eligible customers (never empty here)
    B_c       = state.cum_backorder[c]   B_{c,t-1}(T), the cumulative backorder of
                                         the horizon before period t
    beta_c, p_c                          the allowance and the penalty cost
    other_c   = sum over items j != i of (b_cjt + d_cjt - a_cjt)
                                         what customer c is still owed on the
                                         other items, given the allocation so far

Every policy returns a customer index (never HOLD); ties go to the lowest index.
The pseudocode of the four policies, with worked examples, is in the paper.
"""
from dynaplex.modelling import const_dataclass

from mdp import MultiCustomerSlaMDP, State


@const_dataclass(slots=True)
class FCFSPolicy:
    """FCFS: serve the lowest-indexed eligible customer."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        for c in range(self.mdp.number_of_customers):
            if state.alloc_owed[c] > 0:
                return c
        return self.mdp.number_of_customers  # unreachable: AWAIT_ACTION implies E is not empty


@const_dataclass(slots=True)
class SlaGapPolicy:
    """SLA-gap: serve the eligible customer with the largest gap B_c - beta_c, i.e.
    the one furthest above (or least far below) its allowance."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        best_c = -1
        best_gap = 0
        for c in range(self.mdp.number_of_customers):
            if state.alloc_owed[c] > 0:
                gap = state.cum_backorder[c] - self.mdp.backorder_allowances[c]
                if best_c == -1 or gap > best_gap:
                    best_c = c
                    best_gap = gap
        if best_c == -1:
            return self.mdp.number_of_customers  # unreachable
        return best_c


@const_dataclass(slots=True)
class CostGreedyPolicy:
    """Cost-greedy: for each eligible customer c, project its cumulative backorder
    at the end of the period if no further unit were given to it,
        Bhat_c = B_c + o_c + other_c,
    and call it at risk if Bhat_c > beta_c. Serve the at-risk customer with the
    highest penalty p_c; if no eligible customer is at risk, serve the
    lowest-indexed one.

    Bhat_c is a projection: it treats the customer's remaining owed units on every
    item as backorder at the end of the period, and ignores that the other items
    may still receive stock (later items are still undecided). It is a
    heuristic, not a bound."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        best_c = -1
        best_penalty = -1.0
        for c in range(self.mdp.number_of_customers):
            if state.alloc_owed[c] > 0:
                projected = state.cum_backorder[c] + state.alloc_owed[c]
                for j in range(self.mdp.number_of_items):
                    if j != state.alloc_item:
                        idx = c * self.mdp.number_of_items + j
                        projected += state.backorder[idx] + state.demand[idx] - state.allocation[idx]
                penalty = self.mdp.penalty_costs[c] if projected > self.mdp.backorder_allowances[c] else 0.0
                if penalty > best_penalty:
                    best_penalty = penalty
                    best_c = c
        if best_c == -1:
            return self.mdp.number_of_customers  # unreachable
        return best_c


@const_dataclass(init=False, slots=True)
class GreedyDynamicPolicy:
    """Greedy-dynamic: switch rule by how close the customers are to their
    allowances, measured on the cumulative backorder B_c before this period:
        if B_c > beta_c for some customer c          -> cost-greedy
        else if B_c > 0.8 * beta_c for some customer -> SLA-gap
        else                                         -> FCFS
    The 80% threshold is a design choice carried over from the earlier model."""
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
            if float(state.cum_backorder[c]) > allowance:
                any_exceeds_limit = True
            elif float(state.cum_backorder[c]) > 0.8 * allowance:
                any_near_limit = True

        if any_exceeds_limit:
            return self.cost_greedy.get_action(state)
        elif any_near_limit:
            return self.sla_gap.get_action(state)
        else:
            return self.fcfs.get_action(state)
