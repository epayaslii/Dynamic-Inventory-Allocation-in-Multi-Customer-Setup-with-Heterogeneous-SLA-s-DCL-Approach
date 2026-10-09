"""
Plain-CPython invariant check for MultiCustomerSlaMDP: drives raw
trajectories (bypassing the DynaML JIT) for several policies and asserts the
paper's invariants hold period by period. Run directly from this directory
(`python validate.py`).

This complements (does not replace) the DynaML-compilation check: run
train_and_compare.py, or `assert_mdp`/`assert_policy_for_mdp` plus a
PolicyComparer(..., backend="engine") call, to additionally confirm the
model compiles under the real DynaML JIT (this script alone would not catch
compile-only restrictions, e.g. chained comparisons, that plain CPython
tolerates).
"""
from dynaplex.modelling import StateCategory, const_dataclass, new_context

from mdp import MultiCustomerSlaMDP
from policies import CostGreedyPolicy, FCFSPolicy, GreedyDynamicPolicy, SlaGapPolicy


def run(mdp: MultiCustomerSlaMDP, policy, periods: int, seed: int):
    context = new_context(mdp, seed)
    state = mdp.get_initial_state(context)

    action_calls = 0
    periods_seen = 0
    # entering_history[c][j-1] = BO_c(j), customer c's total backlog at the BEGINNING of period j
    entering_history: list[list[int]] = [[] for _ in range(mdp.number_of_customers)]
    max_actions_this_period = 0
    actions_this_period = 0

    while periods_seen < periods:
        if state.category == StateCategory.AWAIT_EVENT:
            before_period = state.period
            for c in range(mdp.number_of_customers):
                entering_history[c].append(sum(
                    state.backorder[c * mdp.number_of_items + i] for i in range(mdp.number_of_items)))
            mdp.modify_state_with_event(state, context)
            max_actions_this_period = max(max_actions_this_period, actions_this_period)
            actions_this_period = 0
            if state.period != before_period:
                periods_seen += 1
                for v in state.backorder:
                    assert v >= 0, f"negative backorder: {v}"
                for i in range(mdp.number_of_items):
                    for j in range(len(state.pipelines[i])):
                        assert state.pipelines[i][j] >= 0, "negative pipeline/on-hand slot"
                for v in state.cumulative_backorder:
                    assert v >= 0.0
                # Paper: BObar_c(t+1) = sum of BO_c(j) over the periods j of the
                # horizon containing t+1 that are <= t (beginning-of-period
                # backorders, excluding period t+1 itself). Recomputed here from
                # the recorded history, independently of the state's running sum.
                t = state.period
                for c in range(mdp.number_of_customers):
                    horizon = mdp.review_horizon[c]
                    first_period_of_next_horizon = ((t + horizon) // horizon - 1) * horizon + 1
                    expected = sum(entering_history[c][first_period_of_next_horizon - 1:t])
                    assert state.cumulative_backorder[c] == float(expected), (
                        f"cumulative_backorder[{c}]={state.cumulative_backorder[c]} != "
                        f"paper formula {expected} after period {t}")
                for c in range(mdp.number_of_customers):
                    assert 1 <= state.time_remaining[c] <= mdp.review_horizon[c]
                assert context.cumulative_cost >= 0.0
        elif state.category == StateCategory.AWAIT_ACTION:
            assert state.alloc_available > 0, "entered AWAIT_ACTION with nothing to decide"

            mdp.write_action_validity(state, context.valid)
            mask = context.valid.arr[context.valid.row]
            assert any(bool(mask[a]) for a in range(mdp.num_actions)), "no valid actions"
            assert bool(mask[mdp.number_of_customers]), "HOLD must always be valid"

            action = policy.get_action(state)
            assert 0 <= action <= mdp.number_of_customers
            assert bool(mask[action]), f"policy chose invalid action {action}"

            i_before = state.alloc_item
            owed_before = list(state.alloc_owed)
            avail_before = state.alloc_available

            mdp.modify_state_with_action(state, context, action)
            action_calls += 1
            actions_this_period += 1

            if state.category == StateCategory.AWAIT_ACTION and state.alloc_item == i_before:
                assert state.alloc_available == avail_before - 1
                if action < mdp.number_of_customers:
                    assert state.alloc_owed[action] == owed_before[action] - 1
        else:
            raise AssertionError(f"unexpected category {state.category}")

    return context.cumulative_cost, action_calls, max_actions_this_period


@const_dataclass(slots=True)
class AlwaysHoldPolicy:
    """Withholds every unit under scarcity -- exercises the HOLD branch."""
    mdp: MultiCustomerSlaMDP

    def get_action(self, state) -> int:
        return self.mdp.number_of_customers


def main() -> None:
    mdp = MultiCustomerSlaMDP(
        number_of_customers=3,
        number_of_items=2,
        lead_time=2,
        # unequal T_k on purpose -- exercises independent per-customer
        # review-horizon boundaries, not just the paper's equal-T special case.
        review_horizon=[10, 15, 8],
        holding_costs=[1.0, 0.8],
        backorder_allowances=[3, 6, 8],
        penalty_costs=[50.0, 60.0, 40.0],
        # deliberately understocked to force lots of rationing decisions
        demand_rates=[1.0, 0.8, 0.7, 1.2, 0.9, 0.6],
        base_stock_level=[3, 3],
        high_demand_variance=[0, 1, 0, 0, 1, 0],
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

    print("ALL INVARIANTS OK")


if __name__ == "__main__":
    main()
