# Dynamic Inventory Allocation in Multi-Customer Systems with Heterogeneous SLAs

## Overview

This repository implements a sophisticated inventory management system for a central warehouse serving multiple customers, each with distinct SLA requirements. The key innovation is the **Dynamic Allocation Policy** learned via Deep Controlled Learning, which decides how to allocate scarce inventory among customers to minimize long-run average cost while respecting per-customer backorder allowance constraints.

### Core Problem

**Multi-Customer Inventory Allocation Under SLAs:**
- **Static Ordering:** A fixed base-stock policy (per item) maintains target inventory positions
- **Dynamic Allocation:** A learned policy decides which customers receive priority when inventory is scarce
- **SLA Compliance:** Each customer has a cumulative backorder allowance (β_k) per review horizon, with linear penalties if exceeded

## Model Formulation

### System Components

#### 1. Central Warehouse
- Stocks a set of items **I** (**|I|** SKUs) for a set of customers **C** (**|C|** customers)
- Maintains per-item base-stock levels (S_i)
- Replenishment lead time (L) periods

#### 2. Per-Customer SLA Metrics
Each customer k has:
- **Backorder Allowance (β_k):** Maximum cumulative backorder units tolerated per review horizon
- **Review Horizon (T_k):** Period length over which SLA compliance is measured, independent per customer
- **Penalty Cost (p_k):** Linear cost per unit of excess backorders

#### 3. Demand Model
- Per-customer, per-item demand: D_{k,i}(t) ~ Poisson or Geometric, selected per (customer, item) via a high-variance flag

### Key State Variables

| Variable | Meaning | Size |
|----------|---------|------|
| **OH_i(t)** | On-hand inventory at the *start* of period t — before receiving Q_i(t-L) and before placing Q_i(t); always nonnegative (unmet demand is tracked via backorders, not negative on-hand) | \|I\| features |
| **Q_i(t-1...t-L)** | Pipeline orders in transit | \|I\|×L features |
| **BO_{k,i}(t)** | Backorder per customer-item | \|C\|×\|I\| features |
| **D_{k,i}(t)** | Realized demand (current period) | \|C\|×\|I\| features |
| **Σ̄BO_k(t)** | Cumulative backorder within the review horizon, inclusive of period t itself | \|C\| features |
| **T_rem_k(t)** | Time remaining in customer k's review horizon | \|C\| features |
| **β_k** | SLA backorder allowance per customer | \|C\| features |

**Total State Dimension:** \|I\| + \|I\|·L + \|C\|·\|I\| + \|C\|·\|I\| + \|C\| + \|C\| + \|C\| features

Available supply for allocation each period is OH_i(t) + Q_i(t-L) (on-hand plus the order arriving this period) — there is no separately-tracked "before allocation" state variable; this quantity is simply computed inline wherever it's needed.

### Decision Variables

#### Replenishment (Static)
Each period, for each item i:
```
Q_i(t) = max(0, S_i - IP_i(t))
```
where IP_i(t) = on-hand + pipeline - outstanding backorders

#### Allocation (Dynamic)
When demand exceeds inventory, the learned policy π(s_t) decides:
```
A_{k,i}(t) for each customer k and item i
```
subject to:
```
Σ_k A_{k,i}(t) ≤ OH_i(t) + Q_i(t-L)
```
i.e. total allocation of item i cannot exceed on-hand stock plus this period's arriving order. The policy may allocate less than what is available (a strategic hold) even when demand could be fully covered.

### Baseline Rationing Policies

`policies.py` implements four rule-based baselines the learned policy is compared against, each answering "who gets the next available unit?" one unit at a time (see `mdp.py`'s module docstring for why the action is unit-by-unit rather than a whole allocation vector):

1. **FCFSPolicy:** Serves the lowest-index customer still owed a unit
2. **SlaGapPolicy:** Serves whichever eligible customer is furthest past its SLA backorder allowance
3. **CostGreedyPolicy:** Serves whichever eligible customer has the largest projected SLA-penalty exposure if withheld
4. **GreedyDynamicPolicy:** Switches between the three above based on how close any customer is to breaching its allowance

None of these are prescribed by the paper itself — the paper leaves the allocation policy π to be optimized; these exist purely as benchmarks for the learned policy.

## Cost Structure

### Holding Cost (Every Period)
```
C_hold(t) = Σ_i h_i * OH_i(t+1)
```
Charged on inventory remaining after allocation, i.e. OH_i(t+1), not on-hand stock before allocation.

### SLA Penalty (At Horizon End, Per Customer)
```
C_penalty = Σ_k p_k * max(0, Σ̄BO_k(T_k) - β_k)
```
Assessed independently for each customer k at the end of *that customer's own* review horizon T_k. Σ̄BO_k(T_k) is inclusive of the horizon's last period's own backorder, not just the periods before it.

### Objective

The allocation policy π is chosen to minimize long-run average cost per period — taken as the horizon → ∞ limit over all (repeating) review horizons, not the cost of a single horizon.

## Period Workflow (8 Steps)

1. **Receive Orders:** Q_i(t-L) arrives and is added to on-hand inventory
2. **Observe IP:** Calculate inventory position: IP_i = OH_i + Σ_pipeline - Σ_BO
3. **Place Orders:** Q_i(t) = max(0, S_i - IP_i(t))
4. **Observe Demand:** D_{k,i}(t) realized per customer-item pair
5. **Check Rationing:** Is Σ_BO + Σ_D > OH? (per item)
6. **Allocate:** Policy π(s_t) decides allocation when rationing needed
7. **Update State:** OH, BO and cumulative backorder Σ̄BO_k updated based on allocations (Σ̄BO_k resets at each customer's own horizon end)
8. **Calculate Costs:** Holding costs every period, SLA penalties (vs. backorder allowance β_k) at each customer's horizon end

## Deep Controlled Learning

The allocation policy is learned using [DynaPlex](https://dynaplex.github.io/DynaPlex/)'s Deep Controlled Learning (`dp.DCL`): starting from a rollout policy (`GreedyDynamicPolicy` here), it samples candidate actions, evaluates them via lookahead rollouts, and trains a neural network on the best-performing action at each decision point, repeating over successive generations. See `train_and_compare.py` for the full training + comparison loop and `network.py` for the custom `BlockMLP` network used.

## Repository Structure

```
.
├── docs/
│   └── paper_improved_complete.tex   # Full paper with model formulation
├── python/multi_customer_sla/
│   ├── mdp.py                        # MDP, State, featurizer
│   ├── policies.py                   # FCFSPolicy, SlaGapPolicy, CostGreedyPolicy, GreedyDynamicPolicy
│   ├── network.py                    # BlockMLP: custom PyTorch policy network
│   ├── train_and_compare.py          # DCL training + PolicyComparer example
│   └── validate.py                   # Plain-CPython invariant checks
└── README.md (this file)
```

## Key Features

✅ **Multi-Customer Support:** Heterogeneous SLA constraints per customer, including independent per-customer review horizons
✅ **Cumulative Backorder Tracking:** Realistic SLA compliance measurement
✅ **Unit-by-Unit Allocation Decision:** The learned policy allocates each scarce unit directly, not from a fixed menu of rationing rules
✅ **Rule-Based Baselines:** FCFS, SLA-gap, cost-based-greedy, and a dynamic switch between them, for comparison against the learned policy
✅ **Deep Controlled Learning:** Policy optimization via neural networks

## Implementation (Official DynaPlex / DynaML)

`python/multi_customer_sla/` implements the model on the **official** `dynaplex` package (dynaplex.github.io/DynaPlex, `pip install dynaplex`), with models authored in **DynaML**, a compiled Python subset.

It implements `docs/paper_improved_complete.tex` directly: the action *is* the allocation decision A_{c,i}(t) itself (built from a sequence of scalar "who gets the next unit" decisions, since DynaML actions are always a single int — see the module docstring in `mdp.py`), not a choice among a fixed menu of rationing heuristics.

```bash
pip install dynaplex   # Python 3.11-3.14; see dynaplex.github.io for platforms
cd python/multi_customer_sla
python validate.py           # plain-CPython invariant checks
python train_and_compare.py  # DCL training + PolicyComparer (requires torch)
```

Scope: `mdp.py` implements the full MDP, its state, and a featurizer exposing the paper's declared state variables. `policies.py` has the four rule-based baseline policies described above, used as DCL's generation-0 rollout policy and as comparison baselines. `network.py` has `BlockMLP`, a custom PyTorch network with real autograd that encodes the featurizer's feature blocks separately before a shared trunk, wired into `DCL` via `dynaplex.nn.Net`'s importable-factory contract; `dp.MLP` also works against this model (tested) if a plain flat-vector MLP is preferred instead.

Verified against the real package: `assert_mdp` / `assert_policy_for_mdp` / `assert_featurizer_for_mdp` for the MDP and all four policies, a plain-CPython simulation checking the paper's invariants across all four policies (`validate.py`), DynaML compilation of the MDP and every policy (including `GreedyDynamicPolicy`'s composition of the other three) under the JIT `engine` backend, a real PyTorch backward pass through `BlockMLP` confirming every parameter receives a gradient, and end-to-end `DCL` training + `PolicyComparer` runs with both `BlockMLP` and `dp.MLP`.

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
