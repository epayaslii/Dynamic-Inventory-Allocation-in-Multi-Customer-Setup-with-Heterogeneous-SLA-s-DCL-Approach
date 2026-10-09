# Literature review: base-stock level setting (RQ1) and its link to allocation (RQ2)

Scope: what exists on choosing base-stock / order-up-to levels when the stock is
then rationed among customers with different service requirements, and on
learning such policies with deep reinforcement learning (DRL).

## How this was done, and its limits

- About a dozen web searches (general web search, not Scopus / Web of Science /
  Google Scholar). Results are search-engine summaries of abstracts, listing
  pages and some PDFs.
- **I read abstracts and bibliographic records, not full texts.** Statements
  below are limited to what those abstracts say. Where a venue, year or detail
  was not confirmed by a search result, the entry says so.
- A statement that "no paper does X" means *I did not find one in these
  searches*. It is not evidence that none exists. See the recommended systematic
  search at the end.

## 1. The core question: why RQ1 and RQ2 cannot be fully separated

Across the rationing literature, the order-up-to level and the allocation
(rationing) parameters are chosen **together**, and the best level of one depends
on the other:

- Hung, Chew, Lee and Liu (2012), allowing backorders, derive dynamic critical
  levels (single period) and a periodic-review rationing policy (multi period);
  numerical tests find them close to optimal. A related thesis shows the optimal
  ordering rule is state dependent: the optimal base-stock level depends on the
  backorders per class and on the inventory positions across the lead time.
- A Cornell technical report (TR001450) combines a base-stock-type ordering rule
  with a threshold allocation and backorder-clearing policy, and minimizes total
  system stock subject to class-specific fill-rate constraints.
- Kranenburg and van Houtum (2008) report substantial savings from critical-level
  policies compared with base-stock policies with no critical levels. Deshpande,
  Cohen and Donohue (2003) report savings over "round-up" and "separate stock"
  policies under certain conditions.

**Implication for this project (my inference):** fixing S and optimizing the
allocation, then optimizing S, is a reasonable heuristic decomposition, but the
literature gives no reason to expect it to reach the joint optimum. The S that is
best under FCFS need not be best under a learned allocation policy, because a
good allocation can substitute for stock.

## 2. Findings by theme

### 2.1 Stock rationing with several demand classes (classical OR)

| Work | What it contributes | Verification |
|---|---|---|
| Veinott (1965) | Introduced critical-level policies for multiple demand classes | via secondary sources only |
| Topkis (1968) | Proved optimality of critical rationing levels, periodic review | via secondary sources only |
| Ha (1997), *Naval Research Logistics* 44(5), 457-472 | Make-to-stock, two priority classes, backordering; optimal production control and rationing characterized by one monotone switching curve (Poisson arrivals, exponential production) | abstract/listing only |
| Ha (1997), *Management Science* 43(8), 1093-1103 | Several classes, lost sales; sequence of monotone rationing levels | listing only |
| Deshpande, Cohen, Donohue (2003), *Management Science* 49(6), 683-703 | Static threshold rationing in a (Q, r) system, two classes; algorithm for Q, r and the threshold K; lower bound on the optimal cost | abstract read |
| Kranenburg and van Houtum (2008), *J. Operational Research Society* 59(7), 946-955 | Critical levels to differentiate service; minimizes holding and transport cost subject to a mean waiting-time constraint per customer group | abstract read |
| Kranenburg and van Houtum (2007), *Operations Research Letters* 35 | (S-1,S) lost-sales model with multiple demand classes; algorithms for the optimal critical level | title and secondary description only |
| Hung, Chew, Lee, Liu (2012) | Backorders allowed; dynamic critical levels | venue not confirmed |
| Escalona, Ordonez, Iturrieta (2017/2018) | Continuous-review (Q, r, C) with a constant critical level, two classes, convex cost; reported average benefit of 5.9% over a round-up policy and 33.5% over separate stocks | year and venue not confirmed |
| Arslan, Graves, Roemer (2007); Liu et al. (2015) | Named in the search summaries as relevant (single-product multi-class model; multi-class dynamic rationing with backordering) | **not read, not verified** |
| Zhong, Zheng, Chou, Teo (2018), *Management Science*, "Resource Pooling and Allocation Policies to Deliver Differentiated Service" | Links each customer's fill-rate requirement to the resources a pooled system needs (Blackwell approachability) | title only |

Common limits relative to this project: mostly one item, two classes, and
**class-level** targets (fill rate or waiting time). None of what I read has a
**cumulative backorder allowance per customer over a repeating finite review
horizon with a linear penalty**, as in the paper.

### 2.2 One warehouse, several retailers (allocation + base stock)

| Work | What it contributes | Verification |
|---|---|---|
| Federgruen and Zipkin (1984) | Classic balance-based allocation (raise retailers to a common fractile) | only secondary summaries; the summary pages are tool-generated, so I did not rely on their formulas |
| Dogru, De Kok, van Houtum (2009 numerical study) | Tests the balance assumption (it gives a lower bound); gaps small in some instances, **large in many practically relevant settings**; large gaps when retailers differ strongly in service requirements and demand | venue not confirmed |
| Dogru, De Kok, van Houtum (2013), *Central European J. Operations Research* 21(3), 541-559 | Newsvendor-type characterization of optimal base-stock levels under the balance assumption, discrete demand | listing only |
| Berling, Johansson, Marklund (2023) | Heuristics for one-warehouse multi-retailer systems with fill-rate constraints and highly variable order sizes | venue not confirmed |
| Wang and Wan (2023) | Non-identical retailers, Poisson demand, fixed-interval order-up-to policy with a myopic optimal allocation | venue not confirmed |

Relevance: our customers are deliberately heterogeneous, which is exactly where
the balance assumption is reported to be weakest. That supports learning the
allocation instead of assuming it.

### 2.3 Finite-horizon service-level contracts (closest to the paper's SLA)

| Work | What it contributes | Verification |
|---|---|---|
| Thomas (2005), *M&SOM*, "Measuring item fill-rate performance in a finite horizon" | Finite-horizon fill rate is a random variable; its expectation exceeds the long-run value in the zero-lead-time base-stock case | abstract/listing only |
| Katok, Thomas, Davis (2008), *M&SOM*, pp. 609-624, "Inventory service-level agreements as coordination mechanisms: the effect of review periods" | Review-period length and bonus size shift the supplier's base-stock level (laboratory study as well) | abstract read |
| Tan, Paul, Deng, Wei (2017), *POM* 26(11), 1971-1988 | The usual infinite-horizon order-up-to calculation **overstocks** for a finite-horizon fill-rate contract; simulation-based method to trim it; extension with a penalty for missing the target; gains grow when lead time is long relative to the horizon | abstract read |
| Heemskerk (Erasmus thesis, 2018) | Positive lead time in the finite-horizon fill-rate / order-up-to setting | thesis listing only |
| Temizoz et al. (2026), "How to ace your next service level contract review?" (SSRN 6075348) | The paper's own reference for recurring finite-horizon SLAs | **I could not find it with search; I cannot confirm its content** |

This is the most relevant line for RQ1: with recurring finite review horizons,
the classical infinite-horizon S is expected to be too high. Whether the same
holds with a cumulative-backorder allowance, rather than a fill rate, is not
something I found addressed.

### 2.4 Simulation optimization of base-stock levels

- Glasserman and Tayur (1995): simulation-based derivative estimators with
  respect to base-stock levels in capacitated multi-echelon systems (venue not
  confirmed).
- Suman and Ciarallo (2011): simulation search for base-stock levels with
  several demand sources; of four allocation schemes, no single one was best
  under all conditions (venue not confirmed).
- A Wright State dissertation: simulation optimization of cost subject to a
  customer service level under several allocation policies (title not
  confirmed).
- Gentinetta et al. (IBM, *EJOR*): joint capacity-allocation and base-stock
  optimization with service-level and budget constraints, in assembly systems.

These matter because they show the "search over S, evaluate by simulation under
a given allocation rule" approach has precedent, including the finding that the
best allocation rule depends on the setting.

### 2.5 Deep reinforcement learning for inventory

| Work | What it contributes | Verification |
|---|---|---|
| Boute, Gijsbrechts, van Jaarsveld, Vanvuchelen (2022), *EJOR* 298(2), 401-412 | Roadmap of DRL design choices for inventory problems | abstract read |
| Temizoz, Imdahl, Dijkman, Lamghari-Idrissi, van Jaarsveld (2025), *EJOR* 324(1), 104-117 | Deep Controlled Learning: approximate policy iteration with Sequential Halving and common random numbers; reported to beat state-of-the-art heuristics and DRL on lost sales, perishable and random-lead-time problems | abstract read (optimality gap reported as 0.2% in the published version) |
| Gijsbrechts et al. (2022) | General-purpose DRL often needs heavy tuning and can be dominated by specialized policies | seen via a search summary only |
| Kaynov et al. (2024), *Int. J. Production Economics* 267 | One-warehouse multi-retailer DRL; prior work pairs echelon base-stock with heuristic allocation; handles the exponentially large allocation space with a multi-discrete action distribution and a random rationing step; reported gains of roughly 1-3% (lost sales) and 12-20% (partial backordering) over benchmarks | summary only |
| Stranieri, Stella, Kouki (2024) | Adds a balanced allocation rule learned with PPO | summary only; venue not confirmed |
| Harsha et al. (2025, *M&SOM*) | RL for large combinatorial action spaces with state-dependent constraints, network replenishment with lost sales | summary only |
| arXiv 2304.08769 | GPU-parallel learning of a warehouse allocation sub-policy with replenishment, single product | summary only |
| Reinforcement-learning-for-optimal-inventory-rationing project (Univ. of Toronto) | Rationing under contractual penalties, Monte Carlo based | project page only |
| DeepStock (2026 preprint) | DRL regularized towards the base-stock structure, industrial deployment | unaudited preprint |

Kaynov et al. is the closest structural match to our current setup: static
base-stock ordering plus a **learned allocation**. The key difference is that
they work with lost sales or partial backordering and standard cost structures,
not per-customer cumulative backorder allowances.

## 3. Gap

I did not find, in these searches, any work that combines **all** of:

1. a cumulative-backorder (not fill-rate) allowance per customer,
2. over a **repeating finite review horizon**, with a linear penalty on excess,
3. heterogeneous allowances and demand across customers and several items,
4. a **learned allocation policy**, and
5. base-stock levels set in the light of that allocation policy.

Each element has precedent (sections 2.1 to 2.5). The combination appears to be
open, which is a reasonable motivation for RQ1 and RQ2, but this should be
confirmed with a systematic search before it is claimed in the paper.

## 4. What this suggests for the implementation (my suggestions, not from the papers)

All of these are options, to be decided with the supervisors.

1. **Start with the cheapest decomposition (current approach).** Hold S constant
   and learn the allocation with DCL. Already done for S = 6 (see the paper's
   results section, pending the backorder-timing rerun).
2. **Add an outer search over S.** Evaluate each candidate S by simulation on
   common random numbers. The rule-based policies take about 10 seconds per
   evaluation, so a grid over S for the baselines is cheap. DCL took about
   4 minutes per reduced training run, so retraining DCL at every S is more
   expensive but feasible for a handful of values.
3. **Check whether the best S moves with the policy.** Compare the cost-minimizing
   S under the best rule against that under the learned policy. If they differ,
   alternate: re-learn the allocation at the new S, re-search S, and stop when
   neither changes. This tests the section-1 concern directly.
4. **Initialize S with a finite-horizon correction.** Tan et al. suggest the
   infinite-horizon S is too high for finite review horizons; the search range
   for S should therefore include values below the classical choice.
5. **Per-item S** makes the search |I|-dimensional. Coordinate search is a
   simple option; derivative-based estimators (Glasserman and Tayur) would need
   adapting to this cost structure.
6. **Keep the baselines at every S.** Gijsbrechts et al. warn that DRL is often
   matched by good specialized policies, so a fair comparison needs strong
   benchmarks at the same S.

## 5. Recommended follow-up

- A systematic search in Google Scholar / Scopus / Web of Science with terms such
  as: "stock rationing" AND "base-stock" AND "service level agreement"; "critical
  level" AND "backorder allowance"; "one-warehouse multi-retailer" AND "deep
  reinforcement learning"; "finite horizon" AND "service level contract" AND
  "allocation"; "cumulative backorder" AND "penalty" AND "review period".
- Snowballing (backward and forward citations) from Kaynov et al. (2024), Tan et
  al. (2017), Hung et al. (2012), Deshpande et al. (2003) and Doğru et al.
- Reading in full the papers marked "not verified" above, and confirming the
  venues marked "not confirmed", before they go into the paper's bibliography.

## 6. Source links (as returned by the searches)

- Hung et al. thesis / dynamic rationing with backorders: https://ir.lib.nycu.edu.tw/handle/11536/16714
- Cornell TR001450: https://ecommons.cornell.edu/bitstream/handle/1813/9321/TR001450.pdf;sequence=1
- Convex backorders of a rationing inventory policy: https://www.numdam.org/articles/10.1051/ro/2016029/
- Kranenburg and van Houtum 2008 (JORS): https://ideas.repec.org/a/pal/jorsoc/v59y2008i7d10.1057_palgrave.jors.2602414.html
- Van Jaarsveld and Dekker working paper EI2009-14: https://repub.eur.nl/pub/16265/EI2009-14.pdf
- Deshpande, Cohen, Donohue 2003: https://ideas.repec.org/a/inm/ormnsc/v49y2003i6p683-703.html
- Ha 1997 (NRL): https://ideas.repec.org/a/wly/navres/v44y1997i5p457-472.html
- Ha 1997 (Management Science): https://ideas.repec.org/a/inm/ormnsc/v43y1997i8p1093-1103.html
- Zhong et al. 2018: https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.2016.2674
- Dogru et al. 2013 (CEJOR): https://ideas.repec.org/a/spr/cejnor/v21y2013i3p541-559.html
- Dogru et al. balance-assumption study: https://nokia.com/bell-labs/publications-and-media/publications/a-numerical-study-on-the-effect-of-the-balance-assumption-in-one-warehouse-multi-retailer-inventory-systems
- Berling et al.: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3109544
- Katok, Thomas, Davis 2008: https://pubsonline.informs.org/doi/fpi/10.1287/msom.1070.0188
- Tan et al. 2017: https://www.ceibs.edu/sites/portal.prod1.dpmgr.ceibs.edu/files/2017%20POMS%20Fill%20Rate.pdf
- Heemskerk thesis: https://thesis.eur.nl/pub/43258/Heemskerk.pdf
- Boute et al. 2022: https://research.tue.nl/en/publications/deep-reinforcement-learning-for-inventory-control-a-roadmap/
- Temizoz et al. 2025 (DCL): https://research.tue.nl/nl/publications/deep-controlled-learning-for-inventory-control/ and https://arxiv.org/html/2011.15122v7
- Kaynov et al. 2024: https://ideas.repec.org/a/eee/proeco/v267y2024ics0925527323003201.html
- Rationing with RL (Toronto): https://www.mie.utoronto.ca/wp-content/uploads/2019/09/Reinforcement-learning-for-optimal-inventory-rationing.pdf
- GPU-parallel RL allocation: https://awesomepapers.io/ai-agents/papers/2304.08769
