# Code walkthrough: `python/multi_customer_sla/`

A section-by-section explanation for presenting the code. Symbols follow the
paper (`docs/paper_improved_complete.tex`). Line numbers refer to the files as
of this commit.

The pipeline in one sentence: `mdp.py` defines the inventory system,
`policies.py` gives rule-based allocation baselines, `network.py` defines the
neural network, `train_and_compare.py` trains the network with DCL and
compares it with the baselines, and `validate.py` checks the MDP's invariants
without any learning.

---

## 0. The action space (the part DCL needs defined)

**Where it is defined.** The action space lives in `mdp.py`, not in
`validate.py` or `train_and_compare.py`. Those two files only *use* it:
`validate.py` drives it with policies, `train_and_compare.py` hands it to DCL.

**Definition.**

    A = {0, 1, ..., |C|-1, |C|}          num_actions = |C| + 1   (mdp.py:248)

- Action `c` (0 <= c < |C|): give the next unit of the item currently being
  rationed to customer `c`.
- Action `|C|`: HOLD. Keep that unit in the warehouse.

**Why a single integer.** DynaML requires every action to be one scalar int.
The paper's decision is a whole matrix `A_{c,i}(t)` with
`sum_c A_{c,i}(t) <= OH'_i(t)`. So the matrix is built from a *sequence* of
scalar decisions, one per unit of scarce stock: "who gets this unit?". The
tally of those choices is exactly `A_{c,i}(t)`. This reaches every feasible
allocation in the paper, not a subset.

**When the policy is asked.** Only when an item is genuinely short:

- If `sum_c (BO_{c,i} + D_{c,i}) <= OH'_i`, everyone is served in full and no
  decision is asked (mdp.py:369-375). The module docstring argues this is
  lossless: withholding stock that could be given only raises holding cost and
  backorders.
- If `OH'_i <= 0`, nothing can be given, all owed units become backorder, and
  no decision is asked (mdp.py:376-386).
- Otherwise the state moves to `AWAIT_ACTION` and the policy is asked once per
  unit of `OH'_i`, until the units run out or nobody is owed anything
  (mdp.py:326).

**Action validity mask** (`write_action_validity`, mdp.py:340). Customer `c` is
a valid action iff `alloc_owed[c] > 0`, i.e. they are still owed a unit of this
item. This enforces `A_{c,i} <= BO_{c,i} + D_{c,i}`, which the paper implies
through `BO >= 0`. HOLD is always valid. DCL only samples and ranks valid
actions.

**What DCL does with it** (`train_and_compare.py`). At sampled states DCL
tries each valid action, rolls the current policy forward `h` periods `m`
times per action, picks the lowest-cost action, and trains the network to
classify "best action" from the state features. The network output has
`num_actions = |C| + 1` logits.

**What the paper does not say.** The paper leaves the policy `pi` open; this
encoding is an implementation choice forced by DynaML. The one feature with no
paper counterpart is the `alloc_item` one-hot (see section 1.7), which tells
the network which item a given unit-decision concerns.

---

## 1. `mdp.py` — the model

### 1.1 `State` (mdp.py:130)

| Field | Paper symbol | Meaning |
|---|---|---|
| `pipelines[i]` | `OH'_i(t)`, `Q_i(t-L+1..t-1)` | FIFO queue per item. Slot 0 is on-hand including this period's arriving order; slots 1..L-1 are the orders still in transit. |
| `backorder[c*|I|+i]` | `BO_{c,i}(t)` | Units of item `i` owed to customer `c`, flattened. |
| `current_demand` | `D_{c,i}(t)` | This period's sampled demand. |
| `current_allocation` | `A_{c,i}(t)` | Grows as unit-decisions are made. |
| `cumulative_backorder[c]` | `BObar_c(t)` | Sum of the beginning-of-period backorders of the earlier periods in the horizon (excludes period `t`). |
| `entering_backorder[c]` | `BO_c(t)` | Customer `c`'s total backlog entering the period, recorded before allocation changes `backorder`. |
| `time_remaining[c]` | `T_rem` | Periods left in customer `c`'s horizon. |
| `alloc_*` | none | Scratch state for the item currently being rationed. |
| `category` | none | `AWAIT_EVENT` (environment moves) or `AWAIT_ACTION` (policy moves). |

Design point to explain: the paper tracks `OH_i(t)` and `Q_i(t-L)` separately
but only ever uses their sum `OH'_i(t)`. The code stores the sum in slot 0, so
the queue has `L` slots instead of `L+1`. The same trick is used by DynaPlex's
own lost-sales model.

### 1.2 `MultiCustomerSlaMDP.__init__` (mdp.py:195)

Takes the paper's parameters: `lead_time` (L), `review_horizon` (T),
`holding_costs` (h_i), `backorder_allowances` (beta_c), `penalty_costs` (p_c),
`base_stock_level` (S_i), `demand_rates`. Each (customer, item) pair gets a
Poisson demand by default, or a Geometric one when `high_demand_variance` is 1
(a higher-variance demand with the same mean). `horizon_type = INFINITE`
encodes the paper's long-run average-cost objective. `__init__` is plain
Python; only the methods below are compiled by DynaML.

Two things differ from the paper's text and are worth stating when presenting:
`review_horizon` is a list (one `T_k` per customer) where the paper has a single
`T`, and `lead_time >= 1` is asserted where the paper allows `L >= 0`.

### 1.3 `get_initial_state` (mdp.py:253)

Each item starts with `S_i` units on hand and empty pipelines; backorders and
cumulative backorders are zero; every customer's `time_remaining` is its full
horizon.

### 1.4 `modify_state_with_event` (mdp.py:278) — paper Steps 1-3

This is the environment's move. In order:

1. **Ordering (Steps 1-2).** For each item compute
   `IP_i = OH'_i + sum(pipeline_rest) - sum_c BO_{c,i}` and push
   `Q_i = max(0, S_i - IP_i)` onto the queue. Ordering is action-independent,
   so it happens here before any decision.
2. **Demand (Step 3).** Sample `D_{c,i}(t)` for every pair and zero the
   allocations.
3. **Hand over to allocation.** Calls `_advance_items(state, context, 0)`.

On "IL vs IP": here `IP` is on-hand plus pipeline minus backorders, exactly the
paper's inventory-position formula. Inventory *level* is slot 0 (on-hand);
position adds the pipeline.

### 1.5 `_advance_items` and `modify_state_with_action` — Step 4

`_advance_items` (mdp.py:350) walks the items in order. For each item it pops
`OH'_i`, computes what each customer is owed (`BO + D`), and takes one of three
branches: full coverage, nothing to give, or genuine scarcity (stop and ask the
policy). After the last item it calls `_finalize_period`.

`modify_state_with_action` (mdp.py:311) applies one unit-decision: it
increments `current_allocation` for the chosen customer (or does nothing for
HOLD), decrements that customer's `alloc_owed` and the units still available.
When the units run out or nobody is owed anything, the item is finalized:
unserved owed units become `BO_{c,i}(t+1)` (paper eq. for `BO(t+1)`), the
remaining stock becomes `OH_i(t+1)`, and processing resumes at the next item.

### 1.6 `_finalize_item` and `_finalize_period` — Steps 5-6

`_finalize_item` (mdp.py:397) charges holding cost
`h_i * OH_i(t+1)` on the stock left after allocation and folds that leftover
into slot 0, which becomes next period's `OH'_i(t+1)`.

`_finalize_period` (mdp.py:407):

- adds each customer's entering backlog `BO_c(t)` (recorded at the start of the period) into `cumulative_backorder[c]`;
- decrements `time_remaining[c]`; at zero it charges
  `p_c * max(0, cumulative_backorder[c] - beta_c)` and resets both counters;
- returns to `AWAIT_EVENT`.

Each customer's horizon runs on its own clock. With equal `T_k` this is the
paper's single-`T` model.

**Backorder timing (settled).** Backorders are counted at the beginning of
each period. The cumulative sum adds the backlog that *entered* period `t` and
excludes the backorder created by period `t`'s own allocation, which counts from
the next period (and, for a horizon's last period, towards the next horizon).
`docs/backorder_timing_report.md` explains the decision and its effect.

### 1.7 `MultiCustomerSlaFeaturizer` (mdp.py:432)

Converts a `State` into the flat vector the network sees, in this order:

| Block | Size | Paper state variable |
|---|---|---|
| Per-item on-hand + pipeline | `|I|*L` | `OH`, `Q(t-1..t-L)` |
| `backorder[c,i]` | `|C||I|` | `BO_{c,i}` |
| `current_demand[c,i]` | `|C||I|` | `D_{c,i}` |
| `cumulative_backorder[c]` | `|C|` | `BObar_c` |
| `time_remaining[c]` | `|C|` | `T_rem` (one per customer here) |
| `backorder_allowances[c]` | `|C|` | `beta_c` |
| `alloc_item` one-hot | `|I|` | none (needed by the unit-by-unit encoding) |

For the item under rationing, slot 0 reports `alloc_on_hand` (the original
`OH'_i`) rather than the queue's slot 0, which has already been popped.

---

## 2. `policies.py` — rule-based baselines

The paper does not prescribe an allocation policy. These four exist as
benchmarks and as DCL's starting policy. Each answers one question: "who gets
the next unit of the item being rationed?"

- **`FCFSPolicy`** (policies.py:23): the lowest-index customer still owed a
  unit. Never holds.
- **`SlaGapPolicy`** (policies.py:36): the owed customer with the largest
  `cumulative_backorder - beta_c`, i.e. furthest past (or closest to) their
  allowance.
- **`CostGreedyPolicy`** (policies.py:57): projects each owed customer's
  backorder this horizon if the unit is withheld
  (`cumulative + this period's running total - beta_c`) and serves the one
  whose penalty `p_c` would be triggered and is highest. Falls back to the
  lowest index when nobody is projected to breach.
- **`GreedyDynamicPolicy`** (policies.py:83): switches rule by SLA pressure.
  If any customer already exceeds `beta_c` it uses CostGreedy; if any is above
  80% of `beta_c` it uses SlaGap; otherwise FCFS. The 80% threshold is
  carried over from the earlier C++ code, not from the paper.

None of them ever chooses HOLD. Only a learned policy (or `AlwaysHoldPolicy`
in `validate.py`) can.

---

## 3. `network.py` — the policy network

`BlockMLP` (network.py:55) encodes each of the seven feature blocks with its
own `Linear + ReLU`, concatenates the encodings, and runs them through a shared
trunk to `num_actions` logits. The point: a plain MLP sees one undifferentiated
vector, so it must learn from data that feature 4 and feature 41 belong to the
same customer. Block encoders bake that grouping in.

`block_mlp_factory` (network.py:97) is the contract DynaPlex calls to build the
network. It computes the block sizes from `|C|`, `|I|`, `L` and checks they sum
to the featurizer's declared size, so a mismatch between `write_features` and
the network fails loudly instead of silently misreading the tensor.

`dp.MLP` is a drop-in alternative if a plain network is preferred.

---

## 4. `train_and_compare.py` — DCL training and comparison

1. **Instance** (train_and_compare.py:22): 2 customers, 3 items, `L=2`,
   `T=20` for both, `beta=[6, 8]`, `p=[50, 60]`, `S=[6, 7, 5]`. The base stock
   is deliberately below demand so rationing, and hence the learned decision,
   actually occurs.
2. **Static checks**: `assert_mdp`, `assert_policy_for_mdp`,
   `assert_featurizer_for_mdp` are no-ops at runtime. They let pyright verify
   the code satisfies DynaPlex's interfaces.
3. **Network**: `dp.Net("network.block_mlp_factory", ...)` points DCL at
   `BlockMLP`.
4. **DCL** (train_and_compare.py:77):
   - `GreedyDynamicPolicy` is the generation-0 policy used for rollouts.
   - `n=8000` labelled samples per generation.
   - `m=200` rollouts per candidate action.
   - `h=100` periods per rollout.
   - `loss="ce"` (cross-entropy): the network is a classifier of the best
     action.
   - `generations=3`: each generation rolls out the previous generation's
     policy, so the policy improves iteratively.
5. **Comparison** (train_and_compare.py:93): `PolicyComparer` simulates 100
   trajectories with a 100-period warm-up and a 1000-period horizon, using the
   same random numbers for every policy, so cost differences are not noise from
   different demand draws. It compares the four baselines against each DCL
   generation.

The metric is long-run average cost per period, matching the paper's
objective.

---

## 5. `validate.py` — invariant checks without learning

It drives raw trajectories in plain CPython (no DynaML compile), so a bug in
the model is separated from a bug in training.

- **Instance** (validate.py:78): 3 customers, 2 items, deliberately unequal
  `T = [10, 15, 8]` to exercise independent horizon boundaries, with some
  Geometric demands and a tight base stock to force rationing.
- **`run`** (validate.py:21): steps the MDP and asserts, every period:
  no negative backorder, no negative pipeline or on-hand slot, cumulative
  backorder `>= 0`, `1 <= time_remaining[c] <= T_c`, cost `>= 0`.
- **At each decision** it asserts: `alloc_available > 0` on entering
  `AWAIT_ACTION`, at least one valid action, HOLD always valid, the policy's
  action is valid, and a unit-decision reduces `alloc_available` by exactly one
  and the chosen customer's `alloc_owed` by exactly one.
- **`AlwaysHoldPolicy`** (validate.py:71): never serves anyone under scarcity.
  It exercises the HOLD branch, which none of the four baselines ever takes.
- - **Backorder-accounting check:** after every period it recomputes `BObar_c` from the recorded history with the paper's sum formula and asserts it equals `cumulative_backorder[c]`.
- It runs all four baselines for 500 periods and AlwaysHold for 200, then
  prints cost, number of action calls, and the maximum decisions in one
  period.

What it does not check: that the model compiles under the DynaML JIT (a
compile-only restriction such as chained comparisons would slip through), so it
complements running `train_and_compare.py` rather than replacing it.

---

## 6. Status of what has been verified

Not run on this machine. The `dynaplex` wheels on PyPI need macOS 15 or later
(Intel) or macOS 14 or later (Apple Silicon); this Mac is Intel on macOS 13.6.1,
so `pip install dynaplex` finds no compatible file. The README's claims about
compilation and training results come from the repo's earlier sessions and are
not re-verified here. The explanations above come from reading the code, not
from running it.
