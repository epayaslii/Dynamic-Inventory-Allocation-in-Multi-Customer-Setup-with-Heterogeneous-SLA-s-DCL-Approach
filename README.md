# Dynamic Inventory Allocation in Multi-Customer Systems with Heterogeneous SLAs

## Overview

This repository implements a sophisticated inventory management system for a central warehouse serving multiple customers, each with distinct SLA requirements. The key innovation is the **Dynamic Allocation Policy** learned via Deep Controlled Learning, which decides how to allocate scarce inventory among customers to minimize long-run average cost while respecting per-customer backorder allowance constraints.

### Core Problem

**Multi-Customer Inventory Allocation Under SLAs:**
- **Static Ordering:** A fixed base-stock policy (per item) maintains target inventory positions
- **Dynamic Allocation:** A learned policy decides which customers receive priority when inventory is scarce
- **SLA Compliance:** Each customer has a cumulative backorder allowance (β_c, in unit-periods) per review horizon, with a linear penalty on the excess

## Model Formulation

The full model is in `docs/paper_improved_complete.tex`. This is a short summary in the paper's notation (lowercase symbols; customer-item pairs are flattened as `c * |I| + i` in the code).

### System
- A supplier stocks items `i ∈ I` and serves customers `c ∈ C` from one shared stock.
- Lead time `τ ≥ 1` periods; one review horizon of `T` periods shared by all customers; the horizons repeat indefinitely.
- Demand `d_cit` is Poisson (or Geometric with the same mean) per customer-item pair.
- Parameters: holding cost `h_i`, base-stock level `S_i`, SLA allowance `β_c` (in unit-periods), penalty cost `p_c`.

### State variables (start of period `t`)
| Symbol | Code | Meaning |
|---|---|---|
| `OH_it` | `state.on_hand[i]` | on-hand inventory, nonnegative |
| `q_{i,t-τ..t-1}` | `state.pipeline[i]` | the `τ` orders in transit, oldest first |
| `b_cit` | `state.backorder[c*n_items+i]` | units owed to customer `c` at the beginning of the period |
| `d_cit` | `state.demand[c*n_items+i]` | demand observed in the period |
| `B_{c,t-1}(T)` | `state.cum_backorder[c]` | cumulative backorder in the horizon before period `t` |
| `R_{t-1}(T)` | `state.time_remaining` | periods left in the horizon before period `t` |
| `β_c` | `backorder_allowances[c]` | allowance (a constant, part of the featurizer) |

`OH'_it = OH_it + q_{i,t-τ}` (the stock available for allocation) is computed where needed, not stored.

### Decisions
- **Ordering (static):** `q_it = (S_i − IP_it)^+` with `IP_it = OH'_it + Σ_{j=t-τ+1}^{t-1} q_ij − Σ_c b_cit`. The base-stock levels are fixed.
- **Allocation (dynamic):** `a_cit` with `a_cit ≤ b_cit + d_cit` and `Σ_c a_cit ≤ OH'_it`. The policy may hold stock back.

### Period workflow
1. The order `q_{i,t-τ}` arrives. 2. `IP_it` is computed and `q_it` is placed. 3. Demand `d_cit` is observed. 4. The policy allocates `a_cit`. 5. The state is updated: `OH_{i,t+1} = OH'_it − Σ_c a_cit`, `b_{ci,t+1} = b_cit + d_cit − a_cit`, and the horizon statistics. 6. The cost is charged.

### Backorders and the SLA (end-of-period convention)
Backorders are counted at the **end** of each period, after its allocation. With `e_ct = Σ_i (b_cit + d_cit − a_cit)`, the horizon statistics are
```
B_ct(T) = B_{c,t-1}(T) + e_ct        R_t(T) = R_{t-1}(T) − 1
```
and in the horizon's last period (`R_t(T) = 0`) the penalty is `Σ_c p_c · (B_ct(T) − β_c)^+`, after which the statistics restart at `(0, …, 0, T)`. The last period's own `e_ct` is therefore counted on its own as the last term of the accumulated sum. A unit backordered at the end of `k` periods counts `k` times. The individual backorders `b_cit` are **not** reset at a horizon boundary: units still owed after a horizon's last period are carried into the next period (they compete for stock there) and count again in the new horizon only if they are still unserved at the end of its periods.

### Cost
`C(s_t, a_t) = Σ_i h_i (OH'_it − Σ_c a_cit) + Σ_c p_c · (B_ct(T) − β_c)^+ · 1[R_t(T) = 0]`, minimized as the long-run average cost per period.

### Rule-based allocation policies
`policies.py` has four rules. Each answers "who gets the next unit of the scarce item?" and never holds a unit back (see the module docstring of `mdp.py` for why the action is unit-by-unit). Pseudocode and a worked example are in the paper (Section "Pseudocode of the Benchmark Policies").
1. **FCFSPolicy:** the lowest-indexed customer still owed a unit.
2. **SlaGapPolicy:** the customer with the largest `B_{c,t-1}(T) − β_c`.
3. **CostGreedyPolicy:** among customers projected to exceed their allowance (`B_{c,t-1}(T) + o_c + w_c > β_c`), the one with the highest `p_c`; otherwise the lowest index.
4. **GreedyDynamicPolicy:** cost-greedy if some `B_{c,t-1}(T) > β_c`, SLA-gap if some `B_{c,t-1}(T) > 0.8 β_c`, otherwise FCFS.

The paper prescribes no allocation policy; these are benchmarks and the rollout policy of generation 0 of DCL.

## Deep Controlled Learning

The allocation policy is learned with [DynaPlex](https://dynaplex.github.io/DynaPlex/)'s Deep Controlled Learning (`dp.DCL`): starting from a rollout policy (`GreedyDynamicPolicy`), it samples candidate actions, evaluates them with lookahead rollouts, and trains a neural network on the best action at each decision point, over successive generations. See `train_and_compare.py` and `network.py` (`BlockMLP`).

## Repository Structure

```
.
├── docs/
│   ├── paper_improved_complete.tex   # the paper (model, policies, results)
│   ├── code_walkthrough.md           # section-by-section explanation of the code
│   ├── mdp_paper_alignment.md        # how the code relates to the paper
│   ├── literature_review_base_stock.md
│   └── backorder_timing_report.md    # superseded, kept for the record
├── python/multi_customer_sla/
│   ├── mdp.py                        # MDP, State, featurizer (any number of items and customers)
│   ├── policies.py                   # the four rule-based policies
│   ├── network.py                    # BlockMLP: custom PyTorch policy network
│   ├── validate.py                   # plain-CPython invariant checks against the paper's formulas
│   ├── example_policies.py           # small made-up example: period trace and cost of each policy
│   ├── compare_baselines.py          # the four policies at one constant base stock (no learning)
│   └── train_and_compare.py          # DCL training + PolicyComparer
└── README.md
```

## Running

`dynaplex` and `torch` live in a virtual environment (`.venv`, created once in the repository root). The scripts only work with that environment's Python, not with the system `python3` (and on macOS there is no `python`, only `python3`).

```bash
# once, from the repository root:
python3 -m venv .venv
.venv/bin/pip install dynaplex torch      # Python 3.11-3.14; macOS needs 15+ on Intel, 14+ on Apple Silicon

# every time:
source .venv/bin/activate                 # or call .venv/bin/python directly
cd python/multi_customer_sla
python validate.py              # invariant checks (any instance size)
python example_policies.py      # follow the period workflow; cost of each policy on a made-up instance
python compare_baselines.py     # the four policies on the paper's instance (compiles under DynaML)
python train_and_compare.py     # DCL training (slow: see --n --m --h --generations)
```
On Intel Macs the last available torch is 2.2.2, which needs `numpy<2` (`.venv/bin/pip install "numpy<2"`). In VS Code, select the interpreter `.venv/bin/python` (Command Palette: *Python: Select Interpreter*) before using the Run button.

## Status of verification

Checked on the current code: `validate.py` passes (it recomputes the order, the inventory position, the horizon statistics and the period cost from the paper's formulas, and fails when the MDP is deliberately broken), pyright accepts the `assert_mdp` / `assert_policy_for_mdp` / `assert_featurizer_for_mdp` calls, `compare_baselines.py` compiles and runs the MDP and all four policies under the DynaML engine, and the featurizer's output length matches its declared size and the network's block sizes for several instance sizes.

**Not yet re-run on the current code:** DCL training. The featurizer now includes the allocation-in-progress blocks (see `mdp.py`), so the training results reported in the paper's results section come from the earlier version and must be regenerated.

## Related Work

This research builds on:
- **METRIC** (Sherbrooke, 1968): Multi-echelon inventory optimization
- **Deep Reinforcement Learning** (Boute et al., 2022): Learning-based control policies
- **DCL Framework** (Temizoz et al., 2025): Deep Controlled Learning for inventory systems
- **Recurring finite-horizon SLAs** (Temizoz et al., 2026): Static and dynamic base-stock policies under repeating finite-horizon service targets

## References

1. Sherbrooke, C.C. (1968). "METRIC: A Multi-Echelon Technique for Recoverable Item Control"
2. Temizoz, T., et al. (2025). "Deep Controlled Learning for Inventory Control"
3. Boute, R.N., et al. (2022). "Deep Reinforcement Learning for Inventory Control: A Roadmap"
4. Temizoz, T., et al. (2026). "How to ace your next service level contract review?" 

## License

This project is released under the MIT License. See LICENSE file for details.

---

**Documentation:** See `docs/` folder for detailed technical papers and experiment results.
