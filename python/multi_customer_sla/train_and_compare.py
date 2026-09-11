"""
Train a DCL agent on MultiCustomerSlaMDP -- using network.BlockMLP, a custom
per-feature-block PyTorch network (see network.py's module docstring for why
this exists rather than just dp.MLP) -- and compare it against the FCFS,
SLA-gap, cost-greedy, and greedy-dynamic rationing heuristics on common
random numbers.

Run directly from this directory (`python train_and_compare.py`). Artifacts
(samples, trained agents) land in dynaplex_runs/; rerunning resumes rather
than recomputes.
"""
import dynaplex as dp

from mdp import MultiCustomerSlaFeaturizer, MultiCustomerSlaMDP
from policies import CostGreedyPolicy, FCFSPolicy, GreedyDynamicPolicy, SlaGapPolicy


def main() -> None:
    # 2 customers, 3 items -- same illustrative instance as the project
    # README's C++ configuration example, translated to this MDP's fields
    # (a single shared lead_time per paper Sec. 3.1; review_horizon is
    # per-customer T_k, matching the README's `reviewHorizons`).
    mdp = MultiCustomerSlaMDP(
        number_of_customers=2,
        number_of_items=3,
        lead_time=2,
        review_horizon=[20, 20],
        holding_costs=[1.0, 0.8, 1.2],
        backorder_allowances=[6, 8],
        penalty_costs=[50.0, 60.0],
        # customer 0: items 0,1,2 ; customer 1: items 0,1,2
        demand_rates=[1.0, 0.8, 0.5, 0.7, 1.2, 0.6],
        base_stock_level=[6, 7, 5],  # understocked relative to demand, so
        # rationing (and thus the learned allocation decision) actually
        # happens -- generous base stock would make every period fully
        # covered and the action space moot, per the module docstring.
        high_demand_variance=[0, 0, 0, 0, 0, 0],
    )

    fcfs = FCFSPolicy(mdp=mdp)
    sla_gap = SlaGapPolicy(mdp=mdp)
    cost_greedy = CostGreedyPolicy(mdp=mdp)
    greedy_dynamic = GreedyDynamicPolicy(mdp=mdp)

    # No-ops at runtime; they make pyright statically verify that the MDP,
    # policies, and featurizer satisfy the interfaces DynaPlex expects.
    dp.modelling.assert_mdp(mdp)
    dp.modelling.assert_policy_for_mdp(mdp, fcfs)
    dp.modelling.assert_policy_for_mdp(mdp, sla_gap)
    dp.modelling.assert_policy_for_mdp(mdp, cost_greedy)
    dp.modelling.assert_policy_for_mdp(mdp, greedy_dynamic)
    dp.modelling.assert_featurizer_for_mdp(MultiCustomerSlaFeaturizer, mdp)

    # network.block_mlp_factory encodes each feature block (inventory,
    # backorder, demand, cumulative backorder, time remaining, SLA
    # allowances, alloc-item pointer) separately before a shared trunk --
    # see network.py. dp.MLP(hidden=[128, 128]) is a drop-in alternative
    # (also tested against this model, works fine) if you'd rather not use
    # the block-structured network.
    custom_network = dp.Net(
        "network.block_mlp_factory",
        number_of_customers=mdp.number_of_customers,
        number_of_items=mdp.number_of_items,
        lead_time=mdp.lead_time,
        block_hidden=64,
        trunk_hidden=[128, 128],
    )

    d = dp.DCL(
        mdp, greedy_dynamic,                # generation-0 rollout policy
        features=MultiCustomerSlaFeaturizer,
        n=8000,                             # labeled samples per generation
        m=200,                              # rollouts per candidate action
        h=100,                              # rollout horizon (periods)
        workers=8, slots=256,
        network=custom_network,
        train=dict(loss="ce", epochs=50, batch_size=64, lr=1e-3,
                   patience=10, val_fraction=0.1),
    )
    agents = d.run(generations=3)

    comparer = dp.PolicyComparer(
        mdp,
        number_of_trajectories=100,
        warmup_time=100,
        horizon=1000,
        seed=0,
        backend="engine",
    )
    to_compare = {
        "FCFS": fcfs,
        "SlaGap": sla_gap,
        "CostGreedy": cost_greedy,
        "GreedyDynamic": greedy_dynamic,
    }
    for agent in agents:
        to_compare[f"DCL_gen{agent.info['generation']}"] = agent
    print(comparer.compare(to_compare))


if __name__ == "__main__":
    main()
