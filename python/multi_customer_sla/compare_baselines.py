"""
Cost of each rule-based allocation policy at ONE constant base-stock level.

No learning here: the base stock is the same fixed value for every item and
never changes, and only the allocation rule differs between runs. Gives the
reference costs a DCL-learned allocation policy has to beat (see
train_and_compare.py). Run from this directory (`python compare_baselines.py`).
"""
import dynaplex as dp

from mdp import MultiCustomerSlaMDP
from policies import CostGreedyPolicy, FCFSPolicy, GreedyDynamicPolicy, SlaGapPolicy

BASE_STOCK = 6  # same constant for every item; change here to rerun at another level


def main() -> None:
    mdp = MultiCustomerSlaMDP(
        number_of_customers=2,
        number_of_items=3,
        lead_time=2,
        review_horizon=[20, 20],
        holding_costs=[1.0, 0.8, 1.2],
        backorder_allowances=[6, 8],
        penalty_costs=[50.0, 60.0],
        demand_rates=[1.0, 0.8, 0.5, 0.7, 1.2, 0.6],
        base_stock_level=[BASE_STOCK] * 3,
        high_demand_variance=[0, 0, 0, 0, 0, 0],
    )

    comparer = dp.PolicyComparer(
        mdp,
        number_of_trajectories=100,
        warmup_time=100,
        horizon=1000,
        seed=0,
        backend="engine",
    )
    print(f"base stock = {BASE_STOCK} for every item")
    print(comparer.compare({
        "FCFS": FCFSPolicy(mdp=mdp),
        "SlaGap": SlaGapPolicy(mdp=mdp),
        "CostGreedy": CostGreedyPolicy(mdp=mdp),
        "GreedyDynamic": GreedyDynamicPolicy(mdp=mdp),
    }))


if __name__ == "__main__":
    main()
