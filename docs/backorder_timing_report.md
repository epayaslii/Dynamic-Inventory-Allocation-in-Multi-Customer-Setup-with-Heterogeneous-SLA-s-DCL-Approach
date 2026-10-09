# Backorder timing in the SLA penalty: decision, changes and effect on results

For: Willem van Jaarsveld, Tarkan Temizoz
From: Eliz Payasli
Status: model text, code and results updated; open points in section 6.

## 1. Summary

The paper was ambiguous about *which* backorder is added to a customer's
cumulative backorder in each period, and therefore about which periods' backorders
are charged in which review horizon. We settled it: **backorders are counted at
the beginning of each period.** The cumulative sum contains the backlog that
entered the earlier periods of the horizon and excludes the current period's own
backorder.

This changed the paper's equations, the simulation code, the validation checks and
the numerical results. With the new accounting, the learned (DCL) allocation policy
still beats all four rule-based policies. At a constant base stock of 6, the
cost-greedy rule costs 53.87 per period and the three DCL generations cost between
52.37 and 52.83, i.e. 1.9% to 2.8% less (details in section 5).

## 2. The ambiguity

Section 3.3.3 defines BO_{c,i}(t) as the backlog *entering* period t, from past
periods' unmet demand. But two other places treated the backorder as the one left
*after* period t's allocation:

- the definition of the cumulative sum said it runs "through period t itself" with
  BO_c(j) "at the end of period j";
- Step 6 added "this period's own backorder", BO_c(t+1), to the cumulative sum in the
  penalty, and described it as "the backorder just realized this period".

These two readings assign backorders to different horizons, so they are not
equivalent up to relabeling. The earlier code followed the second reading.

## 3. Decision and definitions

Backorders are counted at the beginning of each period. With review horizon T, r(t)
the index of the horizon containing t, and BO_c(j) = sum over items of BO_{c,i}(j),
the backlog owed to customer c entering period j:

- Cumulative sum (state variable):
  BObar_c(t) = sum_{j=(r-1)T+1}^{t-1} BO_c(j). It is 0 in the first period of a horizon.
- Update, inside a horizon (t < r(t)T): BObar_c(t+1) = BObar_c(t) + BO_c(t).
- Update, at a boundary (t = r(t)T): BObar_c(t+1) = 0.
- Penalty at the last period t = rT:
  p_c * max(0, BObar_c(t) + BO_c(t) - beta_c).

Consequences:

1. The backorder created by period t's own allocation, BO_c(t+1), is not charged in
   the horizon containing t. It is the next period's beginning-of-period backorder.
2. In particular, **the backorder created by a horizon's last period counts towards
   the next horizon**, as the backlog entering that horizon's first period.
3. Period 1's beginning-of-period backorder is always 0.
4. Individual backorders BO_{c,i} still persist across horizon boundaries; only the
   cumulative sum restarts, at 0.

### Worked example (one customer, one item, T = 3, beta = 1, p = 10)

Backlog entering periods 1 to 4 is BO(1) = 0, BO(2) = 2, BO(3) = 1, and period 3's
allocation creates BO(4) = 3.

| Convention | Quantity charged in horizon 1 (periods 1 to 3) | Excess over beta | Penalty |
|---|---|---|---|
| New: beginning of period | BO(1) + BO(2) + BO(3) = 0 + 2 + 1 = 3 | 2 | 20 |
| Old: after each period's allocation | BO(2) + BO(3) + BO(4) = 2 + 1 + 3 = 6 | 5 | 50 |

Under the new convention the 3 units created in period 3 are charged in horizon 2,
as part of the backlog entering its first period.

## 4. What changed

**Paper** (`docs/paper_improved_complete.tex`)
- The cumulative-sum definition (state variables) now sums periods up to t-1 and defines
  BO_c(j) at the beginning of period j.
- The Step 5 recursion: second term BO_c(t) instead of BO_c(t+1); the new-horizon case is 0 instead
  of BO_c(t+1). The sentence below it is rewritten.
- The Step 6 penalty uses BObar_c(t) + BO_c(t), with the explanation rewritten; the allowance
  paragraph in the state variables is adjusted to match.
- The numerical-results section is regenerated (section 5 below).

**Code** (`python/multi_customer_sla/`)
- `mdp.py`: a new state field `entering_backorder[c]` = BO_c(t), recorded in
  `modify_state_with_event` before the allocation changes `state.backorder`.
  `_finalize_period` adds it to `cumulative_backorder[c]` and then runs the unchanged
  per-customer boundary and penalty check. Adding first and then testing the boundary
  reproduces the two cases of the recursion.
- `validate.py`: a new check that recomputes BObar_c from the recorded history of
  entering backlogs using the paper's sum formula, and asserts it equals the state's
  value after every period. It runs with unequal horizons T = (10, 15, 8). I confirmed the
  check fails on the old accounting (assertion error at period 2).

**Documentation:** README, `docs/mdp_paper_alignment.md` (section 5, previously the open
question), `docs/code_walkthrough.md`.

## 5. Effect on the results

Setup (unchanged): 2 customers, 3 items, L = 2, T = 20, beta = (6, 8), p = (50, 60),
holding costs (1.0, 0.8, 1.2), Poisson demand, **constant base stock S = 6 for every
item**. Evaluation: 100 trajectories of 1000 periods after a warm-up of 100, common
random numbers. DCL: n = 2000, m = 100, h = 50, 3 generations (a reduced budget).
Numbers are average cost per period with standard errors of the mean in parentheses.

| Policy | Old accounting | New accounting (beginning of period) | New: difference to cost-greedy (paired) |
|---|---|---|---|
| FCFS | 65.17 (0.58) | 65.17 (0.59) | +11.30 (0.09) |
| SLA-gap | 55.26 (0.57) | 55.42 (0.57) | +1.55 (0.04) |
| Greedy-dynamic | 53.46 (0.54) | 54.37 (0.54) | +0.50 (0.02) |
| Cost-greedy | 53.01 (0.54) | 53.87 (0.54) | 0 |
| DCL generation 1 | 52.32 (0.53) | 52.71 (0.52) | -1.16 (0.04) |
| DCL generation 2 | 52.45 (0.52) | 52.37 (0.53) | -1.50 (0.04) |
| DCL generation 3 | 52.29 (0.53) | 52.83 (0.52) | -1.04 (0.04) |

What the comparison shows:

- FCFS is unchanged. The other rules cost 0.2 to 0.9 more under the new accounting, so
  the accounting matters for the level of the costs.
- The ordering of the rule-based policies is unchanged (FCFS worst, then SLA-gap,
  greedy-dynamic, cost-greedy).
- All three DCL generations still beat every rule-based policy; the gap to cost-greedy
  is larger than before in relative terms (1.9% to 2.8% versus 1.0% to 1.3%).
- The cost does not decrease steadily across DCL generations: generation 2 is best.
  Whether the differences between generations are significant has not been tested.

Caveats: the "old" and "new" DCL agents come from separate training runs on different
versions of the model, so their costs are not a like-for-like measure of the training
itself. Everything is a single instance, one base-stock level, one evaluation seed and a
reduced training budget, with no repeated training runs; the standard errors cover only
the demand sampling in the evaluation.

## 6. Open points for you

1. **Is the one-period lag intended?** Under the new convention, the shortfall of a
   horizon's last period is charged to the next horizon. Is that the intended contract
   semantics, or should the final period's own shortfall count in its own horizon?
2. **Rule-based policies were not re-derived.** SLA-gap and cost-greedy read
   `cumulative_backorder` and add a projection of the current period's new backorder.
   Under the new accounting that new backorder only enters the sum from the next period.
   They remain valid benchmarks (every policy is evaluated under the same cost), but
   their projection logic is a heuristic carried over from the earlier model.
3. **State features.** The network sees `cumulative_backorder` and the per-pair
   `backorder` features, but not `entering_backorder` as a separate frozen feature.
   Adding it may help DCL, at the cost of changing the network input.
4. **Compute.** Generation 2 and 3 rollouts use the network and are roughly 13 to 19 times
   slower per simulated period than rule-based rollouts (measured), so the full budget (n = 8000, m = 200,
   h = 100) has not been run to completion.
5. **Next step on the base stock** (RQ1): see `docs/literature_review_base_stock.md`.
