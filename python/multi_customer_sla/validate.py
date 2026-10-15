"""
Plain-CPython invariant check for MultiCustomerSlaMDP: drives raw trajectories
(bypassing the DynaML JIT) for several policies and checks, period by period,
that the implementation matches the paper. Run from this directory
(`python validate.py`).

Checks that do NOT reuse the MDP's own bookkeeping (they recompute the quantity
from the recorded history with the paper's formulas):
  - the order q_it = (S_i - IP_it)^+, with IP_it recomputed from the start-of-
    period state before the period runs;
  - inventory-position conservation: after the period, OH + pipeline - backorders
    equals max(S_i, IP_it) - d_it;
  - the review horizon statistics B_ct(T) (accumulated end-of-period backorders)
    and R_t(T);
  - the period cost = holding cost + SLA penalty at horizon ends.
Plus the nonnegativity and action-validity invariants.

This complements (does not replace) the DynaML-compilation check: run
compare_baselines.py or train_and_compare.py to confirm the model compiles under
the real DynaML JIT (this script alone would not catch compile-only restrictions).
"""
from dynaplex.modelling import StateCategory, const_dataclass, new_context

from mdp import MultiCustomerSlaMDP
from policies import CostGreedyPolicy, FCFSPolicy, GreedyDynamicPolicy, SlaGapPolicy


def run(mdp: MultiCustomerSlaMDP, policy, periods: int, seed: int):
    n_c, n_i, tau, big_t = mdp.number_of_customers, mdp.number_of_items, mdp.lead_time, mdp.review_horizon
    context = new_context(mdp, seed)
    state = mdp.get_initial_state(context)

    action_calls = 0
    max_actions_this_period = 0
    actions_this_period = 0
    end_backorder_history: list[list[int]] = [[] for _ in range(n_c)]  # e_ct for t = 1, 2, ...
    ip_before = [0] * n_i
    cost_before = 0.0

    def check_period_end() -> None:
        """Run once a period has finished, whether it finished inside the event step
        (no scarcity) or after the last allocation decision."""
        t = state.period  # the period that just finished
        for i in range(n_i):
            assert state.order[i] == max(0, mdp.base_stock_level[i] - ip_before[i]), "order != (S - IP)^+"
            assert state.on_hand[i] >= 0, "negative on-hand"
            assert len(state.pipeline[i]) == tau, "pipeline length != tau"
            for j in range(tau):
                assert state.pipeline[i][j] >= 0
            total_demand = sum(state.demand[c * n_i + i] for c in range(n_c))
            ip_after = (state.on_hand[i] + sum(state.pipeline[i][j] for j in range(tau))
                        - sum(state.backorder[c * n_i + i] for c in range(n_c)))
            assert ip_after == max(mdp.base_stock_level[i], ip_before[i]) - total_demand, \
                "inventory position not conserved"
            for c in range(n_c):
                idx = c * n_i + i
                # b_{t+1} = b_t + d - a >= 0 already implies a <= b_t + d; also a >= 0
                assert state.allocation[idx] >= 0, "negative allocation"
        for v in state.backorder:
            assert v >= 0, f"negative backorder: {v}"

        # Horizon statistics: B_ct = sum of end-of-period backorders e_cl over the
        # horizon's periods l <= t; they restart after the horizon's last period.
        holding = sum(mdp.holding_costs[i] * state.on_hand[i] for i in range(n_i))
        penalty = 0.0
        for c in range(n_c):
            end_backorder_history[c].append(sum(state.backorder[c * n_i + i] for i in range(n_i)))
            first = ((t - 1) // big_t) * big_t + 1  # first period of the horizon containing t
            cum = sum(end_backorder_history[c][first - 1:t])
            if t % big_t == 0:
                penalty += mdp.penalty_costs[c] * max(0, cum - mdp.backorder_allowances[c])
                assert state.cum_backorder[c] == 0, "statistics not restarted"
            else:
                assert state.cum_backorder[c] == cum, (
                    f"B[{c}]={state.cum_backorder[c]} != paper formula {cum} after period {t}")
        expected_remaining = big_t if t % big_t == 0 else big_t - (t % big_t)
        assert state.time_remaining == expected_remaining

        period_cost = context.cumulative_cost - cost_before
        assert abs(period_cost - (holding + penalty)) < 1e-6, (
            f"period {t}: cost {period_cost} != holding {holding} + penalty {penalty}")

    periods_seen = 0
    while periods_seen < periods:
        if state.category == StateCategory.AWAIT_EVENT:
            # Independent recomputation of IP_it from the start-of-period state (paper eq. for IP).
            for i in range(n_i):
                ip_before[i] = (state.on_hand[i] + sum(state.pipeline[i][j] for j in range(tau))
                                - sum(state.backorder[c * n_i + i] for c in range(n_c)))
            cost_before = context.cumulative_cost
            mdp.modify_state_with_event(state, context)
        elif state.category == StateCategory.AWAIT_ACTION:
            assert state.alloc_units > 0, "entered AWAIT_ACTION with nothing to decide"

            mdp.write_action_validity(state, context.valid)
            mask = context.valid.arr[context.valid.row]
            assert any(bool(mask[a]) for a in range(mdp.num_actions)), "no valid actions"
            assert bool(mask[mdp.number_of_customers]), "HOLD must always be valid"

            action = policy.get_action(state)
            assert 0 <= action <= mdp.number_of_customers
            assert bool(mask[action]), f"policy chose invalid action {action}"

            i_before = state.alloc_item
            owed_before = list(state.alloc_owed)
            units_before = state.alloc_units

            mdp.modify_state_with_action(state, context, action)
            action_calls += 1
            actions_this_period += 1

            if state.category == StateCategory.AWAIT_ACTION and state.alloc_item == i_before:
                assert state.alloc_units == units_before - 1
                if action < mdp.number_of_customers:
                    assert state.alloc_owed[action] == owed_before[action] - 1
        else:
            raise AssertionError(f"unexpected category {state.category}")

        if state.category == StateCategory.AWAIT_EVENT and state.period > periods_seen:
            periods_seen = state.period
            check_period_end()
            max_actions_this_period = max(max_actions_this_period, actions_this_period)
            actions_this_period = 0

    return context.cumulative_cost, action_calls, max_actions_this_period


@const_dataclass(slots=True)
class AlwaysHoldPolicy:
    """Withholds every unit under scarcity -- exercises the HOLD branch."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state) -> int:
        return self.mdp.number_of_customers


def main() -> None:
    # Deliberately not the paper's instance: 3 customers, 4 items, tau = 3 and T = 7,
    # some Geometric demands, base stock far below demand so rationing happens all
    # the time. The same checks must hold for any sizes.
    mdp = MultiCustomerSlaMDP(
        number_of_customers=3,
        number_of_items=4,
        lead_time=3,
        review_horizon=7,
        holding_costs=[1.0, 0.8, 1.5, 0.5],
        backorder_allowances=[3, 6, 8],
        penalty_costs=[50.0, 60.0, 40.0],
        demand_rates=[1.0, 0.8, 0.7, 0.4, 1.2, 0.9, 0.6, 0.5, 0.3, 0.7, 0.9, 0.6],
        base_stock_level=[3, 3, 4, 2],
        high_demand_variance=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    )

    policies = [
        ("FCFS", FCFSPolicy(mdp=mdp)),
        ("SlaGap", SlaGapPolicy(mdp=mdp)),
        ("CostGreedy", CostGreedyPolicy(mdp=mdp)),
        ("GreedyDynamic", GreedyDynamicPolicy(mdp=mdp)),
    ]
    for name, policy in policies:
        cost, calls, max_actions = run(mdp, policy, periods=500, seed=1234)
        print(f"{name}: cumulative_cost={cost:.2f} action_calls={calls} max_actions/period={max_actions}")

    cost, calls, max_actions = run(mdp, AlwaysHoldPolicy(mdp=mdp), periods=200, seed=99)
    print(f"AlwaysHold: cumulative_cost={cost:.2f} action_calls={calls} max_actions/period={max_actions}")

    # A single customer and a single item (the smallest instance), tau = 1, T = 1.
    tiny = MultiCustomerSlaMDP(1, 1, 1, 1, [1.0], [0], [10.0], [1.5], [2])
    run(tiny, FCFSPolicy(mdp=tiny), periods=300, seed=5)
    print("1 customer x 1 item (tau=1, T=1): OK")

    print("ALL INVARIANTS OK")


if __name__ == "__main__":
    main()
