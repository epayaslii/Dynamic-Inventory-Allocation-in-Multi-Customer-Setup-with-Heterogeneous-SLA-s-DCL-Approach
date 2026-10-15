# `mdp.py` and the paper

How `python/multi_customer_sla/mdp.py` relates to `docs/paper_improved_complete.tex` (current
version: lowercase notation, a single review horizon `T`, lead time `tau >= 1`, backorders counted
at the end of each period). Validated by `validate.py`, which recomputes the quantities below
from the paper's formulas and compares them with the state, period by period.

## 1. Matches the paper

| Paper | Code |
|---|---|
| `OH_it`, pipeline `q_{i,t-tau..t-1}`, `b_cit`, `d_cit` kept as start-of-period state | `on_hand`, `pipeline` (a queue of `tau` orders, oldest first), `backorder`, `demand`; they keep their start-of-period values until the period is finalized |
| `OH'_it = OH_it + q_{i,t-tau}` is not stored | computed where needed (`pipeline[i][0]` is the arriving order) |
| `IP_it = OH'_it + sum_{j=t-tau+1}^{t-1} q_ij - sum_c b_cit` | `modify_state_with_event` |
| `q_it = (S_i - IP_it)^+` | `state.order[i]`; pushed to the pipeline in `_finalize_period` |
| feasible allocations: `a_ci <= b_cit + d_cit`, `sum_c a_ci <= OH'_it` | enforced by the validity mask (customer `c` is valid iff still owed a unit) and by the unit count |
| `OH_{i,t+1} = OH'_it - sum_c a_cit` | `_finalize_period` |
| `b_{ci,t+1} = b_cit + d_cit - a_cit` | `_finalize_period` |
| `B_ct = B_{c,t-1} + e_ct` with `e_ct = sum_i (b_cit + d_cit - a_cit)`, `R_t = R_{t-1} - 1` | `cum_backorder`, `time_remaining` |
| penalty `p_c (B_ct - beta_c)^+` at `R_t = 0`, then restart at `(0, ..., 0, T)` | `_finalize_period` |
| holding cost `sum_i h_i (OH'_it - sum_c a_cit)` | `_finalize_period` |
| initial state: `OH_i1 = S_i`, empty pipeline, no backorders, `P_1 = (0, ..., 0, T)` | `get_initial_state` |
| average-cost objective | `horizon_type = INFINITE`; the long-run average is computed by DynaPlex's `PolicyComparer` |
| `S_i` fixed, only the allocation is optimized | `base_stock_level` is a constructor argument that no method changes |

The sizes `|C|`, `|I|`, `tau` and `T` are constructor arguments; `validate.py` runs 3 x 4 items/customers
(`tau = 3`, `T = 7`) and 1 x 1 (`tau = 1`, `T = 1`) with the same checks.

## 2. Differences, and why

1. **Unit-by-unit action (DynaML constraint).** DynaML actions are a single scalar int, while the
   paper's decision is the matrix `a_cit`. The matrix is built from a sequence of "who gets the next
   unit?" decisions (action `c`, or `|C|` for HOLD), one per available unit of a scarce item. Every
   allocation in the paper's feasible set can be produced. Scratch variables (`order`, `allocation`,
   `alloc_item`, `alloc_owed`, `alloc_units`) exist only for this decomposition.
2. **No decision when stock covers everything owed, or when there is no stock.** The code then allocates
   in full (or nothing). This restricts the policies considered to those that allocate in full whenever
   supply covers everything owed. It is not lossless in general: holding a unit back that could serve one
   customer now can pay off when it is reserved for a customer close to its allowance. (An earlier
   version of this document argued it was lossless; that argument ignored this reservation effect.)
3. **Extra features.** The featurizer contains the paper's state (dimension `|I| + |I| tau + 2|C||I| +
   2|C| + 1`) followed by four blocks about the allocation in progress: the item one-hot, the units still
   owed per customer, the units left to decide, and the allocation so far. Without them consecutive unit
   decisions of the same item would be indistinguishable to the policy. They are not part of the paper's
   state.
4. **Integer allowances.** `backorder_allowances` are integers (unit-periods).

## 3. Not covered

- `tau = 0` (same-period replenishment) is not allowed; the paper also requires `tau >= 1`.
- Per-customer review horizons were removed; the paper has one `T` shared by all customers.
- DCL training with the current featurizer has not been re-run (see the README).
