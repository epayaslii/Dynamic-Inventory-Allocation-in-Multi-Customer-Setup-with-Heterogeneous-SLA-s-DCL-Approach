"""
A small made-up example for following the period workflow and for seeing what each
allocation policy costs. No training is involved. Run from this directory:

    python example_policies.py                # trace + costs on the scripted demand + stochastic costs
    python example_policies.py --policy fcfs  # trace another policy (fcfs, sla_gap, cost_greedy, greedy_dynamic)

Instance (made up, small enough to check by hand): customers A, B, C and items X, Y;
lead time tau = 2, review horizon T = 5, base stock S = (7, 8).

Part 1 traces every step of every period for one policy on a SCRIPTED demand
sequence (so you can redo the arithmetic yourself). Part 2 gives the cost of each
policy on that same scripted demand. Part 3 compares the policies on random
(Poisson) demand, averaged over many trajectories with common random numbers.
"""
import argparse
import math

from dynaplex.modelling import StateCategory, new_context

from mdp import MultiCustomerSlaMDP
from policies import CostGreedyPolicy, FCFSPolicy, GreedyDynamicPolicy, SlaGapPolicy

CUSTOMERS = ["A", "B", "C"]
ITEMS = ["X", "Y"]
TAU = 2
T = 5
S = [7, 8]
H = [1.0, 2.0]                 # holding cost per unit per period, items X and Y
BETA = [3, 6, 9]               # allowance of A, B, C in unit-periods
P = [20.0, 40.0, 10.0]         # penalty per unit-period above the allowance
RATES = [1.0, 0.5,             # lambda_{A,X}, lambda_{A,Y}
         0.8, 0.8,             # lambda_{B,X}, lambda_{B,Y}
         0.4, 1.2]             # lambda_{C,X}, lambda_{C,Y}

# Scripted demand d_cit: SCRIPT[t-1][c][i], periods 1..8 (two review horizons are
# started, so the penalty at the end of period 5 is visible)
SCRIPT = [
    [[1, 0], [1, 1], [0, 1]],   # t = 1   normal demand
    [[1, 1], [0, 1], [1, 1]],   # t = 2
    [[2, 1], [2, 1], [1, 2]],   # t = 3   demand burst
    [[2, 1], [2, 2], [1, 1]],   # t = 4   demand burst
    [[1, 0], [1, 1], [0, 2]],   # t = 5   last period of horizon 1
    [[1, 1], [0, 1], [1, 1]],   # t = 6   horizon 2 starts
    [[0, 0], [2, 1], [0, 1]],   # t = 7
    [[1, 0], [1, 1], [1, 0]],   # t = 8
]


class ScriptedSampler:
    """Stands in for a demand sampler in plain CPython and returns scripted values."""

    def __init__(self, values):
        self.values = list(values)
        self.k = 0

    def sample(self, rng) -> int:
        v = self.values[self.k] if self.k < len(self.values) else 0
        self.k += 1
        return v


def make_mdp(scripted: bool) -> MultiCustomerSlaMDP:
    mdp = MultiCustomerSlaMDP(
        number_of_customers=len(CUSTOMERS), number_of_items=len(ITEMS), lead_time=TAU,
        review_horizon=T, holding_costs=H, backorder_allowances=BETA, penalty_costs=P,
        demand_rates=RATES, base_stock_level=S,
    )
    if scripted:
        for c in range(len(CUSTOMERS)):
            for i in range(len(ITEMS)):
                mdp.demand_samplers[c * len(ITEMS) + i] = ScriptedSampler([SCRIPT[t][c][i] for t in range(len(SCRIPT))])  # type: ignore  (plain CPython only: not DynaML-compilable)
    return mdp


def policies_for(mdp):
    return {
        "fcfs": FCFSPolicy(mdp=mdp),
        "sla_gap": SlaGapPolicy(mdp=mdp),
        "cost_greedy": CostGreedyPolicy(mdp=mdp),
        "greedy_dynamic": GreedyDynamicPolicy(mdp=mdp),
    }


def run_period(mdp, state, context, policy, log):
    """Advance exactly one period; if log is a list, append (item, action) decisions to it."""
    start = state.period
    while state.period == start:
        if state.category == StateCategory.AWAIT_EVENT:
            mdp.modify_state_with_event(state, context)
        else:
            action = policy.get_action(state)
            if log is not None:
                log.append((state.alloc_item, action, list(state.alloc_owed)))
            mdp.modify_state_with_action(state, context, action)


def trace(mdp, policy, periods):
    n_c, n_i = len(CUSTOMERS), len(ITEMS)
    context = new_context(mdp, 0)
    state = mdp.get_initial_state(context)
    total_h = total_p = 0.0
    for _ in range(periods):
        t = state.period + 1
        # snapshot of the START-of-period state s_t (before the arriving order is received)
        oh = list(state.on_hand)
        pipe = [[state.pipeline[i][j] for j in range(TAU)] for i in range(n_i)]
        b = list(state.backorder)
        cum = list(state.cum_backorder)
        r_prev = state.time_remaining
        cost_before = context.cumulative_cost

        decisions = []
        run_period(mdp, state, context, policy, decisions)

        d = list(state.demand)
        q = list(state.order)
        a = list(state.allocation)
        holding = sum(H[i] * state.on_hand[i] for i in range(n_i))
        period_cost = context.cumulative_cost - cost_before
        penalty = period_cost - holding
        total_h += holding
        total_p += penalty

        print(f"\n=== Period t = {t}   (horizon {math.ceil(t / T)}, start of period: B_(t-1) = {cum}, R_(t-1) = {r_prev})")
        for i, item in enumerate(ITEMS):
            arriving = pipe[i][0]
            oh_prime = oh[i] + arriving
            rest = sum(pipe[i][1:])
            bo_sum = sum(b[c * n_i + i] for c in range(n_c))
            ip = oh_prime + rest - bo_sum
            print(f" Item {item}: OH = {oh[i]}, pipeline (oldest first) = {pipe[i]}")
            print(f"   Step 1  q arrives = {arriving}  ->  OH' = {oh[i]} + {arriving} = {oh_prime}")
            print(f"   Step 2  IP = {oh_prime} + {rest} - {bo_sum} = {ip}   order q = max(0, {S[i]} - {ip}) = {q[i]}")
        print("   Step 3  demand d (rows A, B, C; columns X, Y):", [[d[c * n_i + i] for i in range(n_i)] for c in range(n_c)])
        print("   Step 4  owed = b + d:", [[b[c * n_i + i] + d[c * n_i + i] for i in range(n_i)] for c in range(n_c)])
        if decisions:
            for item_idx, action, owed in decisions:
                who = CUSTOMERS[action] if action < n_c else "HOLD"
                print(f"           unit decision on item {ITEMS[item_idx]}: owed left {owed} -> {who}")
        else:
            print("           no decision needed: for every item the stock covers everything owed, or there is no stock")
        print("           allocation a (rows A, B, C; columns X, Y):", [[a[c * n_i + i] for i in range(n_i)] for c in range(n_c)])
        e = [sum(state.backorder[c * n_i + i] for i in range(n_i)) for c in range(n_c)]
        print(f"   Step 5  OH(t+1) = {state.on_hand}   pipeline(t+1) = {[[state.pipeline[i][j] for j in range(TAU)] for i in range(n_i)]}")
        print(f"           b(t+1) = {[[state.backorder[c * n_i + i] for i in range(n_i)] for c in range(n_c)]}   end-of-period backorder e = {e}")
        if t % T == 0:
            print(f"           B_t = B_(t-1) + e = (horizon ends)  -> statistics restart at (0, 0, 0, R = {state.time_remaining})")
        else:
            print(f"           B_t = B_(t-1) + e = {state.cum_backorder}   R_t = {state.time_remaining}")
        pen_note = ""
        if t % T == 0:
            full = [cum[c] + e[c] for c in range(n_c)]
            pen_note = f"   (B_t = {full}, allowances {BETA}, excess = {[max(0, full[c] - BETA[c]) for c in range(n_c)]})"
        print(f"   Step 6  holding = {holding:.2f}   SLA penalty = {penalty:.2f}{pen_note}   period cost = {period_cost:.2f}")
    print(f"\nTotal over {periods} periods: holding {total_h:.2f} + penalty {total_p:.2f} = {total_h + total_p:.2f}")


def scripted_totals(periods):
    rows = []
    for name in ["fcfs", "sla_gap", "cost_greedy", "greedy_dynamic"]:
        mdp = make_mdp(scripted=True)
        policy = policies_for(mdp)[name]
        context = new_context(mdp, 0)
        state = mdp.get_initial_state(context)
        hold = pen = 0.0
        for _ in range(periods):
            before = context.cumulative_cost
            run_period(mdp, state, context, policy, None)
            h = sum(H[i] * state.on_hand[i] for i in range(len(ITEMS)))
            hold += h
            pen += (context.cumulative_cost - before) - h
        rows.append((name, hold, pen))
    return rows


def stochastic(n_traj=300, periods=100, warmup=20):
    names = ["fcfs", "sla_gap", "cost_greedy", "greedy_dynamic"]
    res = {n: [] for n in names}
    for seed in range(n_traj):
        for name in names:
            mdp = make_mdp(scripted=False)
            policy = policies_for(mdp)[name]
            context = new_context(mdp, seed)
            state = mdp.get_initial_state(context)
            at_warmup = 0.0
            hold = 0.0
            for t in range(warmup + periods):
                if t == warmup:
                    at_warmup = context.cumulative_cost
                    hold = 0.0
                run_period(mdp, state, context, policy, None)
                if t >= warmup:
                    hold += sum(H[i] * state.on_hand[i] for i in range(len(ITEMS)))
            total = context.cumulative_cost - at_warmup
            res[name].append((total / periods, hold / periods, (total - hold) / periods))
    return res


def decision_example():
    """One hand-designed decision state (used in the paper's policy examples): one
    item, three customers, two units of stock for seven units owed, and the last
    period of the horizon, so the penalty follows immediately. The three
    benchmarks pick three different customers for the first unit."""
    names = ["A", "B", "C"]
    beta = [6, 6, 7]
    pen = [10.0, 40.0, 15.0]
    cum_before = [2, 5, 8]          # B_{c,t-1}
    owed = [2, 3, 2]                # b_ci + d_ci
    print("State: one item, OH' = 2 units, owed (A, B, C) =", owed, ", B_(t-1) =", cum_before,
          ", beta =", beta, ", p =", pen, ", last period of the horizon (R_(t-1) = 1)")
    for label, factory in [("FCFS", FCFSPolicy), ("SLA-gap", SlaGapPolicy),
                           ("Cost-greedy", CostGreedyPolicy), ("Greedy-dynamic", GreedyDynamicPolicy)]:
        mdp = MultiCustomerSlaMDP(3, 1, 1, 5, [1.0], beta, pen, [1.0, 1.0, 1.0], [2])
        policy = factory(mdp=mdp)
        context = new_context(mdp, 0)
        state = mdp.get_initial_state(context)
        state.on_hand = [2]
        state.backorder = [0, 1, 0]
        state.demand = [2, 2, 2]
        state.cum_backorder = list(cum_before)
        state.time_remaining = 1
        state.order = [0]
        state.alloc_item = 0
        state.alloc_owed = list(owed)
        state.alloc_units = 2
        state.category = StateCategory.AWAIT_ACTION
        picks = []
        while state.category == StateCategory.AWAIT_ACTION:
            action = policy.get_action(state)
            picks.append(names[action])
            mdp.modify_state_with_action(state, context, action)
        end_backorder = [owed[c] - state.allocation[c] for c in range(3)]
        cum_after = [cum_before[c] + end_backorder[c] for c in range(3)]
        excess = [max(0, cum_after[c] - beta[c]) for c in range(3)]
        print(f"  {label:15s} units go to {picks}  a = {state.allocation}  e = {end_backorder}  "
              f"B_t = {cum_after}  excess = {excess}  penalty = {context.cumulative_cost:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="cost_greedy", choices=["fcfs", "sla_gap", "cost_greedy", "greedy_dynamic"])
    parser.add_argument("--periods", type=int, default=6, help="periods to trace in part 1")
    args = parser.parse_args()

    print("Instance: customers", CUSTOMERS, "items", ITEMS, f"tau = {TAU}, T = {T}, S = {S}")
    print("h =", H, " beta =", BETA, " p =", P, " lambda (A, B, C rows; X, Y columns) =", [RATES[0:2], RATES[2:4], RATES[4:6]])

    print(f"\n########## Part 1: trace of the '{args.policy}' policy on the scripted demand ##########")
    mdp = make_mdp(scripted=True)
    trace(mdp, policies_for(mdp)[args.policy], args.periods)

    n = len(SCRIPT)
    print(f"\n########## Part 2: cost of each policy on the same scripted demand ({n} periods) ##########")
    print(f"{'policy':16s}{'holding':>10s}{'penalty':>10s}{'total':>10s}")
    for name, hold, pen in scripted_totals(n):
        print(f"{name:16s}{hold:10.2f}{pen:10.2f}{hold + pen:10.2f}")

    print("\n########## Part 4: the one-item decision example used in the paper ##########")
    decision_example()

    n_traj, periods = 300, 100
    print(f"\n########## Part 3: random Poisson demand, {n_traj} trajectories x {periods} periods (after 20 warm-up), common random numbers ##########")
    res = stochastic(n_traj, periods)
    base = [x[0] for x in res["fcfs"]]
    print(f"{'policy':16s}{'cost/period':>12s}{'(std err)':>11s}{'holding':>10s}{'penalty':>10s}{'diff vs FCFS':>14s}{'(std err)':>11s}")
    for name, rows in res.items():
        c = [x[0] for x in rows]
        mean = sum(c) / n_traj
        se = (sum((v - mean) ** 2 for v in c) / (n_traj - 1) / n_traj) ** 0.5
        dif = [c[k] - base[k] for k in range(n_traj)]
        dm = sum(dif) / n_traj
        dse = (sum((v - dm) ** 2 for v in dif) / (n_traj - 1) / n_traj) ** 0.5
        hold = sum(x[1] for x in rows) / n_traj
        pen = sum(x[2] for x in rows) / n_traj
        print(f"{name:16s}{mean:12.3f}{se:11.3f}{hold:10.3f}{pen:10.3f}{dm:14.3f}{dse:11.3f}")


if __name__ == "__main__":
    main()
