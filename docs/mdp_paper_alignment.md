# `mdp.py` ↔ paper alignment

Section-by-section check of `python/multi_customer_sla/mdp.py` against
`docs/paper_improved_complete.tex`. Written for discussion with Tarkan and
Willem: for every piece of the model, this says whether the code matches the
paper exactly, generalizes it deliberately, resolves a DynaML framework
constraint, or resolves a genuine ambiguity in the paper's own text.

Everything below was checked by tracing the actual arithmetic (not just
reading the code's own comments) and cross-checked against `validate.py`'s
passing invariant run and the real DynaML compile (`assert_mdp` /
`assert_policy_for_mdp` / `assert_featurizer_for_mdp`).

## 1. Exact matches

| Paper | Equation / location | Code |
|---|---|---|
| Sets, indices | Sec 3.1 | `number_of_items`, `number_of_customers` |
| Demand $D_{c,i}(t)$ | Sec 3.2 | `demand_samplers[c*n_i+i].sample(...)`, Step 3 of `modify_state_with_event` |
| Inventory position $\text{IP}_i(t)$ | eq. 243 | `mdp.py:261` — `on_hand_and_arriving + pipeline_rest - backorder_sum`, computed from the same three quantities in the same combination |
| Ordering $Q_i(t) = \max(0, S_i - \text{IP}_i(t))$ | eq. 251, Step 2 | `mdp.py:263-264` |
| Allocation constraint $\sum_c A_{c,i}(t) \le \text{OH}'_i(t)$ | eq. 258/279 | enforced structurally: unit-by-unit decisions can never exceed `alloc_on_hand` (`alloc_available` counts down to 0) |
| "Owed" $= \text{BO}_{c,i}(t) + D_{c,i}(t)$ | Step 4 prose | `mdp.py:336` — `owed[c] = state.backorder[idx] + state.current_demand[idx]` |
| $\text{OH}_i(t+1) = \text{OH}'_i(t) - \sum_c A_{c,i}(t)$ | eq. 287 | `_finalize_item`'s `leftover = alloc_on_hand - total_assigned`, folded into the FifoQueue (see §2 below) |
| $\text{BO}_{c,i}(t+1) = \text{BO}_{c,i}(t)+D_{c,i}(t)-A_{c,i}(t)$ | eq. 293 | full-coverage branch sets `backorder[idx] = owed[c] - current_allocation[idx]`(=0 there); scarcity branch sets it to `alloc_owed[c]` once every unit is decided — same formula, since `alloc_owed[c]` *is* `owed[c]` minus units assigned so far |
| $C_\text{hold}(t) = \sum_i h_i \cdot \text{OH}_i(t+1)$ | eq. 325 | `_finalize_item`: `context.cumulative_cost += leftover * holding_costs[i]`, and `leftover` is exactly $\text{OH}_i(t+1)$ (verified in §2) |
| Average-cost objective | eq. 350 | `horizon_type = HorizonType.INFINITE`; long-run averaging itself is DynaPlex's `PolicyComparer` (warmup + long horizon), not `mdp.py`'s job |
| $S_i$ fixed, only $\pi$ optimized | Sec 5 note | `base_stock_level` is a constructor arg, never touched by `modify_state_with_action` |

## 2. Verified derivations (not just asserted — traced through)

**FifoQueue ↔ $\text{OH}'_i(t) = \text{OH}_i(t) + Q_i(t-L)$.** Traced one full
period cycle: `pipelines[i]` holds $L$ slots $[\text{OH}'_i(t), Q_i(t-L+1),
\ldots, Q_i(t-1)]$. Steps 1–2 append the new order, making it
$L+1$ long. `_advance_items` pops the front (`on_hand = OH'_i(t)`,
consumed for allocation), leaving $[Q_i(t-L+1), \ldots, Q_i(t)]$ ($L$
slots). `_finalize_item` adds `leftover` ($=\text{OH}_i(t+1)$, eq. 287) into
the new slot 0, which already holds $Q_i(t-L+1) = Q_i((t+1)-L)$ — giving
slot 0 $= Q_i((t+1)-L) + \text{OH}_i(t+1) = \text{OH}'_i(t+1)$ by eq. 235,
exactly. The remaining slots are $Q_i(t-L+2), \ldots, Q_i(t)$, i.e. the
correct $L-1$ in-transit orders for period $t+1$. This confirms the
`On-hand representation` section of `mdp.py`'s own docstring is accurate,
not just asserted.

**Feasible action set under scarcity.** `write_action_validity` makes
customer $c$ eligible for a unit iff `alloc_owed[c] > 0`, i.e. the running
tally never lets $A_{c,i}(t)$ exceed $\text{BO}_{c,i}(t)+D_{c,i}(t)$. The
paper never states this as an explicit bound alongside eq. 258 — but it
is implied: $\text{BO}_{c,i}(t+1) \ge 0$ is asserted as a state-variable
domain constraint (Sec 3.3.3, "$\in \mathbb{Z}_{\ge 0}$"), and eq. 293 makes
that impossible unless $A_{c,i}(t) \le \text{BO}_{c,i}(t)+D_{c,i}(t)$. So the
code's restriction is a correct reading of an implicit paper constraint, not
an extra one invented by the port.

**Full-coverage shortcut is provably lossless, not just "typically" fine.**
When $\sum_c \text{owed}_c \le \text{OH}'_i(t)$, the code allocates in full
without ever entering `AWAIT_ACTION` — stronger than the paper's "the
policy *typically* allocates in full, but it is not required to" (Step 4).
Checked the dominance argument itself: ordering (Steps 1–2) never depends
on the action, so holding back a unit under full coverage changes nothing
about future orders. Withholding a unit from a customer who's owed it, when
it could be given, simultaneously (a) leaves that unit as on-hand $\Rightarrow$
strictly higher $\text{OH}_i(t+1)$ $\Rightarrow$ weakly higher $C_\text{hold}$, and
(b) raises that customer's $\text{BO}_{c,i}(t+1)$ by exactly the same unit
$\Rightarrow$ weakly higher (never lower) SLA exposure. No continuation can
benefit from either effect. So restricting the presented action space this
way changes nothing about the *reachable optimal behavior* — it only
removes decision points that are always weakly dominated to ask about.

## 3. Deliberate generalizations beyond the literal paper

**Per-customer review horizons ($T_k$).** Paper Sec 3.1 defines $T$ and
$r(t)$ as one shared quantity for all customers; "Time Remaining" is listed
as exactly 1 state feature. This port uses $|C|$ independent review
horizons (`review_horizon: list[int]`, `time_remaining: list[int]`), each
customer's SLA penalty and cumulative-backorder reset firing on its own
boundary. Motivation: the paper's own subject is heterogeneous,
customer-specific SLAs (already $\beta_c$, $p_c$ per customer), and this
repo's earlier C++ model already exposed per-customer `reviewHorizons`. The
paper's literal single-$T$ model is recovered exactly as the special case
$T_1 = \cdots = T_{|C|}$. (Session note: this was implemented and verified
— unequal $T_k$ across customers, both under `validate.py`'s plain-Python
invariant check and the real DynaML JIT compile.)

## 4. DynaML framework constraints, not paper deviations

**Unit-by-unit action decomposition.** DynaML's `MDPProtocol` requires a
single scalar int action per decision step; there is no vector/combinatorial
action type (verified against the installed `dynaplex` package). So a full
period allocation $A_{c,i}(t)$ is built from a sequence of "who gets the
next unit?" decisions (action $\in \{0,\ldots,|C|-1\}$ picks a customer,
action $=|C|$ HOLDs), following the same idiom DynaPlex's own bundled
`binpacking` model uses. The set of $A_{c,i}(t)$ vectors this can produce is
*exactly* the paper's feasible set (see §2) — the decomposition changes
how the decision is presented to a learning algorithm, not what decisions
are reachable. This machinery has no counterpart in the paper's state list,
which is why the featurizer has one extra block (`alloc_item` one-hot) the
paper doesn't declare — needed only to tell the policy which item's
in-progress rationing a given unit-decision concerns.

## 5. SLA penalty timing: resolved (backorders counted at the beginning of each period)

The earlier paper text was not internally consistent about what "BO_c(t)"
means in the penalty formula: Sec 3.3.3 defines it as the backlog *entering*
period t, while the old Step 6 prose called it "the backorder just realized
this period" (which is BO_c(t+1)). The two readings charge different periods
to different horizons.

**Decision:** backorders are counted at the beginning of each period, and the
cumulative sum excludes the current period's own backorder. The paper now says:

- BObar_c(t) = sum of BO_c(j) over j = (r-1)T+1, ..., t-1  (0 in a horizon's
  first period);
- BObar_c(t+1) = BObar_c(t) + BO_c(t) inside a horizon, and 0 when period t+1
  starts a new horizon;
- penalty at the last period t = rT: p_c * max(0, BObar_c(t) + BO_c(t) - beta_c).

Consequence: the backorder created by a horizon's last period is not charged
in that horizon; it is the beginning-of-period backorder of the next horizon's
first period. Period 1's beginning-of-period backorder is always 0.

**Code:** `modify_state_with_event` records `state.entering_backorder[c]`
(BO_c(t)) before the allocation changes `state.backorder`; `_finalize_period`
adds it to `cumulative_backorder[c]` and then runs the per-customer boundary
and penalty check. `validate.py` recomputes BObar_c from the recorded history
using the paper's sum formula and asserts it equals the state's value after
every period (checked with unequal T_k = 10, 15, 8; the check fails on the
earlier end-of-period accounting). See `docs/backorder_timing_report.md` for the
full write-up and the effect on the results.

## 6. Known scope limitation (not addressed in this pass)

**$L=0$ unsupported.** Paper Sec 3.1 allows $L \ge 0$; `mdp.py` asserts
`lead_time >= 1` because the current FifoQueue convention (slot 0 =
$\text{OH}'_i(t)$, slots $1..L-1$ = in-transit) has nowhere to represent
same-period replenishment. Left out of this alignment pass by explicit
decision — not implemented, not otherwise investigated here.
