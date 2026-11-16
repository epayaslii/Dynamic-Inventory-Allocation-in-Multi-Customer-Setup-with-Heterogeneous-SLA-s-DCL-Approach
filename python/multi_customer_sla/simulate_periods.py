"""
Period-by-period simulation of MultiCustomerSlaMDP under a trained DCL agent (and,
for reference, a rule-based policy on the same demand stream), logging EVERY state
variable of the paper in every period. Used to check, over a long run, that the
holding cost and the SLA penalty are charged in the periods the model says.

Drives the MDP's raw trajectory in plain CPython (like validate.py). Per period it
records, in the paper's notation:
    start of period      OH_it, pipeline q_{i,t-tau..t-1}, b_cit, B_{c,t-1}, R_{t-1}
    after Steps 1-3      OH'_it, IP_it, order q_it, demand d_cit
    after Step 4         allocation a_cit
    after Steps 5-6      OH_{i,t+1}, b_{ci,t+1}, e_ct, B_ct, R_t, holding cost,
                         SLA penalty per customer
and re-derives OH', IP, the order, the end-of-period backorder, B_ct, and the cost
from the recorded values with the paper's formulas, asserting they match the MDP.

Run from this directory:
    python simulate_periods.py --periods 2000 --seed 7 --out ../../docs/sim2000
Writes <out>_<policy>.json (one record per period) for each policy.
"""
import argparse
import glob
import json
import os

from dynaplex.nn import NNAgent
from dynaplex.modelling import StateCategory, new_context

from mdp import MultiCustomerSlaMDP
from policies import CostGreedyPolicy

BASE_STOCK = 6


def build_mdp() -> MultiCustomerSlaMDP:
    # The instance of the paper's numerical-results section.
    return MultiCustomerSlaMDP(
        number_of_customers=2,
        number_of_items=3,
        lead_time=2,
        review_horizon=20,
        holding_costs=[1.0, 0.8, 1.2],
        backorder_allowances=[6, 8],
        penalty_costs=[50.0, 60.0],
        demand_rates=[1.0, 0.8, 0.5, 0.7, 1.2, 0.6],
        base_stock_level=[BASE_STOCK] * 3,
        high_demand_variance=[0, 0, 0, 0, 0, 0],
    )


def latest_agent_dir() -> str:
    """The generation-3 agent of the newest DCL experiment that has one."""
    paths = glob.glob(os.path.join("dynaplex_runs", "*", "*", "agent_gen3"))
    assert paths, "no trained generation-3 agent found; run train_and_compare.py first"
    return max(paths, key=os.path.getmtime)


def simulate(mdp: MultiCustomerSlaMDP, choose, periods: int, seed: int) -> list[dict]:
    """choose(state) -> action. Returns one record per period."""
    n_c, n_i, tau, big_t = mdp.number_of_customers, mdp.number_of_items, mdp.lead_time, mdp.review_horizon
    context = new_context(mdp, seed)
    state = mdp.get_initial_state(context)
    records: list[dict] = []
    rec: dict = {}
    cost_before = 0.0
    holds = 0
    cum_ref = [0] * n_c  # own bookkeeping of B_ct, to compare with the MDP's

    while len(records) < periods:
        if state.category == StateCategory.AWAIT_EVENT:
            t = state.period + 1
            rec = {
                "t": t,
                "horizon": (t - 1) // big_t + 1,
                "OH": list(state.on_hand),
                "pipeline": [[state.pipeline[i][j] for j in range(tau)] for i in range(n_i)],
                "b": list(state.backorder),
                "B_prev": list(state.cum_backorder),
                "R_prev": state.time_remaining,
            }
            cost_before = context.cumulative_cost
            holds = 0
            mdp.modify_state_with_event(state, context)
        elif state.category == StateCategory.AWAIT_ACTION:
            action = int(choose(state))
            if action == n_c:
                holds += 1
            mdp.modify_state_with_action(state, context, action)
        else:
            raise AssertionError(state.category)

        if state.category == StateCategory.AWAIT_EVENT and state.period == rec["t"]:
            records.append(finish_record(mdp, state, rec, context.cumulative_cost - cost_before, holds, cum_ref))
    return records


def finish_record(mdp, state, rec, period_cost, holds, cum_ref) -> dict:
    n_c, n_i, tau, big_t = mdp.number_of_customers, mdp.number_of_items, mdp.lead_time, mdp.review_horizon
    t = rec["t"]
    rec["d"] = list(state.demand)
    rec["order"] = list(state.order)
    rec["a"] = list(state.allocation)
    rec["holds"] = holds

    # OH', IP and the order, re-derived from the start-of-period state (paper formulas).
    oh_prime, ip = [], []
    for i in range(n_i):
        arriving = rec["pipeline"][i][0]
        oh_p = rec["OH"][i] + arriving
        ip_i = oh_p + sum(rec["pipeline"][i][1:]) - sum(rec["b"][c * n_i + i] for c in range(n_c))
        oh_prime.append(oh_p)
        ip.append(ip_i)
        assert rec["order"][i] == max(0, mdp.base_stock_level[i] - ip_i), "order != (S-IP)^+"
    rec["OH_prime"], rec["IP"] = oh_prime, ip

    # Feasibility of the allocation, then the end-of-period state.
    for i in range(n_i):
        assert sum(rec["a"][c * n_i + i] for c in range(n_c)) <= oh_prime[i]
        for c in range(n_c):
            assert 0 <= rec["a"][c * n_i + i] <= rec["b"][c * n_i + i] + rec["d"][c * n_i + i]
    rec["OH_next"] = [oh_prime[i] - sum(rec["a"][c * n_i + i] for c in range(n_c)) for i in range(n_i)]
    rec["b_next"] = [rec["b"][k] + rec["d"][k] - rec["a"][k] for k in range(n_c * n_i)]
    assert rec["OH_next"] == list(state.on_hand) and rec["b_next"] == list(state.backorder)

    # e_ct, B_ct, R_t and the cost from the recorded values.
    rec["e"] = [sum(rec["b_next"][c * n_i + i] for i in range(n_i)) for c in range(n_c)]
    if rec["R_prev"] == big_t:
        for c in range(n_c):
            cum_ref[c] = 0
    assert rec["B_prev"] == cum_ref, "B_{c,t-1} != own accumulation"
    for c in range(n_c):
        cum_ref[c] += rec["e"][c]
    rec["B"] = list(cum_ref)
    rec["R"] = rec["R_prev"] - 1
    holding = sum(mdp.holding_costs[i] * rec["OH_next"][i] for i in range(n_i))
    penalty = [0.0] * n_c
    if rec["R"] == 0:
        for c in range(n_c):
            penalty[c] = mdp.penalty_costs[c] * max(0, rec["B"][c] - mdp.backorder_allowances[c])
    rec["holding_cost"], rec["penalty_cost"] = holding, penalty
    rec["cost"] = holding + sum(penalty)
    assert abs(rec["cost"] - period_cost) < 1e-6, f"period {t}: cost {rec['cost']} != MDP {period_cost}"
    assert t % big_t != 0 or list(state.cum_backorder) == [0] * n_c
    assert t % big_t == 0 or list(state.cum_backorder) == rec["B"]
    if rec["R"] == 0:
        for c in range(n_c):
            cum_ref[c] = 0
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="../../docs/sim2000")
    args = parser.parse_args()

    mdp = build_mdp()
    agent_dir = latest_agent_dir()
    agent = NNAgent.load(agent_dir)
    print("DCL agent:", agent_dir, agent.info.get("generation"))
    cost_greedy = CostGreedyPolicy(mdp=mdp)

    runs = {
        "dcl": lambda s: agent([s], mdp=mdp)[0],
        "costgreedy": cost_greedy.get_action,
    }
    meta = {
        "periods": args.periods, "seed": args.seed, "agent": agent_dir,
        "h": mdp.holding_costs, "p": mdp.penalty_costs, "beta": mdp.backorder_allowances,
        "S": mdp.base_stock_level, "tau": mdp.lead_time, "T": mdp.review_horizon,
        "n_c": mdp.number_of_customers, "n_i": mdp.number_of_items,
    }
    for name, choose in runs.items():
        records = simulate(mdp, choose, args.periods, args.seed)
        total = sum(r["cost"] for r in records)
        print(f"{name}: {len(records)} periods, total cost {total:.1f}, avg {total / len(records):.2f}, "
              f"holds {sum(r['holds'] for r in records)}")
        with open(f"{args.out}_{name}.json", "w") as f:
            json.dump({"meta": meta, "records": records}, f, default=int)


if __name__ == "__main__":
    main()
