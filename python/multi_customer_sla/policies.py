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

Every policy returns a customer index (never HOLD). Tie-break rule shared by all
of them (replaces "lowest index"): among the customers tied for the best score,
serve the first one after state.last_served in cyclic order, i.e. units are handed
out one by one, rotating between the tied customers (and the rotation carries over
from one tie to the next, across items and periods).
The pseudocode of the four policies, with worked examples, is in the paper.
"""
from dynaplex.modelling import const_dataclass

from mdp import MultiCustomerSlaMDP, State


@const_dataclass(slots=True)
class FCFSPolicy:
    """FCFS: customers with an outstanding backlog of this item come first, the one
    backlogged for the most consecutive periods (state.backorder_age) first, and
    each gets one unit at a time until its backlog b_ci is covered; then the older
    backlog of the next customer, and so on. A customer's backlog is served before
    its new demand d_ci. When nobody has backlog left, the remaining units go to
    the customers owed new demand, one by one in rotation (tie-break rule)."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        n_c = self.mdp.number_of_customers
        n_i = self.mdp.number_of_items
        i = state.alloc_item
        best_score = -1
        for c in range(n_c):
            if state.alloc_owed[c] > 0:
                score = 0
                if state.alloc_owed[c] - state.demand[c * n_i + i] > 0:  # backlog of item i still unserved
                    score = state.backorder_age[c * n_i + i]
                if score > best_score:
                    best_score = score
        for k in range(1, n_c + 1):
            c = (state.last_served + k + n_c) % n_c
            if state.alloc_owed[c] > 0:
                score = 0
                if state.alloc_owed[c] - state.demand[c * n_i + i] > 0:
                    score = state.backorder_age[c * n_i + i]
                if score == best_score:
                    return c
        return n_c  # unreachable: AWAIT_ACTION implies E is not empty


@const_dataclass(slots=True)
class SlaGapPolicy:
    """SLA-gap: serve the eligible customer with the largest gap B_c - beta_c, i.e.
    the one furthest above (or least far below) its allowance. Ties: rotation."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        n_c = self.mdp.number_of_customers
        found = False
        best_gap = 0
        for c in range(n_c):
            if state.alloc_owed[c] > 0:
                gap = state.cum_backorder[c] - self.mdp.backorder_allowances[c]
                if (not found) or gap > best_gap:
                    found = True
                    best_gap = gap
        for k in range(1, n_c + 1):
            c = (state.last_served + k + n_c) % n_c
            if state.alloc_owed[c] > 0:
                if state.cum_backorder[c] - self.mdp.backorder_allowances[c] == best_gap:
                    return c
        return n_c  # unreachable


@const_dataclass(slots=True)
class CostGreedyPolicy:
    """Cost-greedy: for each eligible customer c, project its cumulative backorder
    at the end of the period if no further unit were given to it,
        Bhat_c = B_c + o_c + other_c,
    and call it at risk if Bhat_c > beta_c. Serve the at-risk customer with the
    highest penalty p_c; ties, and the case where no eligible customer is at risk,
    are broken by the rotation rule.

    Bhat_c is a projection: it treats the customer's remaining owed units on every
    item as backorder at the end of the period, and ignores that the other items
    may still receive stock (later items are still undecided). It is a
    heuristic, not a bound."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state: State) -> int:
        n_c = self.mdp.number_of_customers
        pen: list[float] = [-1.0] * n_c
        best_penalty = -1.0
        for c in range(n_c):
            if state.alloc_owed[c] > 0:
                projected = state.cum_backorder[c] + state.alloc_owed[c]
                for j in range(self.mdp.number_of_items):
                    if j != state.alloc_item:
                        idx = c * self.mdp.number_of_items + j
                        projected += state.backorder[idx] + state.demand[idx] - state.allocation[idx]
                pen[c] = self.mdp.penalty_costs[c] if projected > self.mdp.backorder_allowances[c] else 0.0
                if pen[c] > best_penalty:
                    best_penalty = pen[c]
        for k in range(1, n_c + 1):
            c = (state.last_served + k + n_c) % n_c
            if state.alloc_owed[c] > 0:
                if pen[c] == best_penalty:
                    return c
        return n_c  # unreachable


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
