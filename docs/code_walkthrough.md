# Code walkthrough: `python/multi_customer_sla/`

A section-by-section explanation for presenting the code. Symbols follow the paper
(`docs/paper_improved_complete.tex`): lowercase `b`, `d`, `a`, `q`, `tau` for the lead time and `T` for the
review horizon. Customer-item pairs are flattened as `c * n_items + i`.

The pipeline in one sentence: `mdp.py` defines the inventory system, `policies.py` gives rule-based allocation
policies, `network.py` defines the neural network, `train_and_compare.py` trains it with DCL and compares it with the
rules, `validate.py` checks the MDP against the paper's formulas without any learning, and `example_policies.py` is a
small made-up example that prints a period trace and the cost of each policy.

---

## 0. The action space

Defined in `mdp.py` (`num_actions = |C| + 1`); `validate.py` and `train_and_compare.py` only use it.

    A = {0, ..., |C|-1, |C|}      action c: give the next unit of the scarce item to customer c
                                  action |C|: HOLD (keep the unit)

- **Why one integer.** DynaML actions are a single scalar. The paper's decision is the matrix `a_cit`, so it is built
  from a sequence of unit decisions; the tally is `a_cit`. Every allocation with `a_ci <= b_cit + d_cit` and
  `sum_c a_ci <= OH'_it` can be produced.
- **When the policy is asked.** Only for an item with `sum_c (b_cit + d_cit) > OH'_it` and `OH'_it > 0`: once per
  available unit, until the units run out or nobody is owed anything. Otherwise the code allocates in full (or
  nothing) without a decision.
- **Validity mask.** Customer `c` is valid iff it is still owed a unit of the item (`alloc_owed[c] > 0`); HOLD is
  always valid.
- **What DCL does with it.** At sampled states it tries each valid action, rolls the current policy forward `h`
  periods `m` times per action, and trains the network to classify the best action. The network has `|C| + 1` logits.

---

## 1. `mdp.py`

### 1.1 `State`
| Field | Paper | Meaning |
|---|---|---|
| `on_hand[i]` | `OH_it` | on-hand at the start of the period |
| `pipeline[i]` | `q_{i,t-tau..t-1}` | queue of `tau` orders, oldest first (`[0]` arrives this period) |
| `backorder[c*n_items+i]` | `b_cit` | owed at the beginning of the period |
| `demand[...]` | `d_cit` | demand observed in the period |
| `cum_backorder[c]` | `B_{c,t-1}(T)` | cumulative end-of-period backorders of the earlier periods of the horizon |
| `time_remaining` | `R_{t-1}(T)` | periods left in the horizon (one number, shared by all customers) |
| `order[i]`, `allocation[...]`, `alloc_item`, `alloc_owed[c]`, `alloc_units` | none | scratch for the period and for the unit-by-unit allocation |

The state variables keep their start-of-period values during the whole allocation; the period's results are written
in `_finalize_period`.

### 1.2 `MultiCustomerSlaMDP.__init__`
Takes `number_of_customers`, `number_of_items`, `lead_time` (`tau`), `review_horizon` (`T`), `holding_costs`,
`backorder_allowances`, `penalty_costs`, `demand_rates` (flattened) and `base_stock_level`. Demand is Poisson per pair,
or Geometric with the same mean where `high_demand_variance` is 1. Nothing is specific to an instance size.

### 1.3 `get_initial_state`
`OH_i1 = S_i`, `tau` zero orders in each pipeline, no backorders, `B = 0` and `R = T`.

### 1.4 `modify_state_with_event` (Steps 1 to 3)
Per item: `OH'_it = OH_it + pipeline[i][0]`, `IP_it = OH'_it + (other pipeline orders) - sum_c b_cit`, and the order
`q_it = (S_i - IP_it)^+` is stored in `state.order`. Then the demands are sampled, and the allocation starts.

### 1.5 `_advance_items` and `modify_state_with_action` (Step 4)
`_advance_items` goes item by item. For an item it computes what is owed (`b + d`) and takes one of three branches:
the stock covers everything owed (allocate in full), there is no stock (allocate nothing), or the item is scarce (stop
and ask the policy). `modify_state_with_action` applies one unit decision. When the units run out, or nobody is still
owed anything, it continues with the next item; after the last item it calls `_finalize_period`.

### 1.6 `_finalize_period` (Steps 5 and 6)
- For each item: the arriving order leaves the pipeline, the new order joins it, `OH_{i,t+1} = OH'_it - sum_c a_cit`,
  and the holding cost `h_i * OH_{i,t+1}` is charged.
- `b_{ci,t+1} = b_cit + d_cit - a_cit`.
- `B_ct = B_{c,t-1} + e_ct` where `e_ct` is the customer's total backorder at the **end** of the period; `R_t = R_{t-1} - 1`.
- If `R_t = 0` (the horizon's last period): the penalty `p_c (B_ct - beta_c)^+` is charged, `B` restarts at 0 and `R` at `T`.

The last period's own `e_ct` is the last term of the sum that is charged. At the boundary the individual backorders
`b` are not reset: units still owed are carried into the next period (they compete for stock there) and count again
in the new horizon only if they are still unserved at the end of its periods.

### 1.7 `MultiCustomerSlaFeaturizer`
The paper's state followed by four blocks about the allocation in progress:
`OH_it` and the `tau` pipeline orders per item; `b`; `d`; `B_{c,t-1}`; `R_{t-1}`; `beta`; then the item one-hot, the units
still owed per customer, the units left to decide, and the allocation so far. The last four blocks have no counterpart in
the paper; without them consecutive unit decisions would look identical to the policy.

---

## 2. `policies.py`
The paper prescribes no allocation policy; these are benchmarks and the generation-0 rollout policy. Each answers
"who gets the next unit?" and never holds. Pseudocode and a worked example are in the paper.
- **FCFS:** the lowest-indexed customer still owed a unit.
- **SLA-gap:** the customer with the largest `B_{c,t-1} - beta_c`.
- **Cost-greedy:** among customers with `B_{c,t-1} + o_c + w_c > beta_c`, the one with the highest `p_c`; else the
  lowest index. (`o_c`: units still owed of the current item; `w_c`: units still owed on the other items.)
- **Greedy-dynamic:** cost-greedy if some `B_{c,t-1} > beta_c`, SLA-gap if some `B_{c,t-1} > 0.8 beta_c`, else FCFS.

## 3. `network.py`
`BlockMLP` encodes each of the ten feature blocks with its own `Linear + ReLU` before a shared trunk to `|C| + 1`
logits. `block_mlp_factory` computes the block sizes from `|C|`, `|I|`, `tau` and checks that they sum to the
featurizer's declared size, so a mismatch fails loudly.

## 4. `train_and_compare.py`
Builds the paper's instance (2 customers, 3 items, `tau = 2`, `T = 20`), checks the interfaces, runs DCL (`--n`, `--m`,
`--h`, `--generations`; defaults 8000, 200, 100, 3) from `GreedyDynamicPolicy` with `BlockMLP`, and compares the
agents with the four rules using `PolicyComparer` on common random numbers (100 trajectories, 1000 periods after a
warm-up of 100). From generation 1 on, rollouts call the network and are roughly 13 to 19 times slower per simulated
period than rule-based rollouts.

## 5. `validate.py`
Plain CPython, so no DynaML compile. For several policies and instance sizes it checks, every period (however the
period ends): the order `(S - IP)^+` with `IP` recomputed from the start-of-period state; inventory-position
conservation (`OH + pipeline - b = max(S, IP) - d` after the period); the horizon statistics `B` and `R` against the
paper's sum formula; the period cost against holding + penalty; nonnegativity; and action validity (HOLD always valid).
`AlwaysHoldPolicy` exercises the HOLD branch. It does not check DynaML compilation; `compare_baselines.py` does.

## 6. `example_policies.py` and `compare_baselines.py`
`example_policies.py` is for learning the workflow: a made-up instance (3 customers A, B, C; 2 items X, Y; `tau = 2`,
`T = 5`), a printed trace of every step of each period on scripted demand, the cost of each policy on that demand,
the cost on random demand, and the one-item decision example of the paper. `compare_baselines.py` runs the four
policies on the paper's instance with `PolicyComparer`.
