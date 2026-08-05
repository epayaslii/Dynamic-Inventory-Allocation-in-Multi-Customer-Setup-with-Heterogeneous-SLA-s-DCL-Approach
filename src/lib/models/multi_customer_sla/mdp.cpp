#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <cmath>

namespace DynaPlex::Models {
	namespace multi_customer_sla /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;
			vars.Add("valid_actions", totalRationingActions);
			return vars;
		}

		int64_t MDP::GetH(const State& state) const
		{
			int64_t maxRemaining = 0;
			for (int64_t k = 0; k < numberOfCustomers; k++)
				maxRemaining = std::max(maxRemaining, state.TimeRemaining[k]);
			return maxReviewHorizon + maxRemaining + 1;
		}

		MDP::MDP(const VarGroup& config)
		{
			config.Get("numberOfCustomers", numberOfCustomers);
			config.Get("numberOfItems", numberOfItems);
			config.Get("leadTimes", leadTimes);
			config.Get("holdingCosts", holdingCosts);
			config.Get("backorderAllowances", backorderAllowances);
			config.Get("reviewHorizons", reviewHorizons);
			config.Get("penaltyCosts", penaltyCosts);
			config.Get("customerDemandRates", customerDemandRates);
			config.Get("demandRates", demandRates);
			config.Get("highDemandVariance", highDemandVariance);
			config.GetOrDefault("unavoidableCostPerPeriod", unavoidableCostPerPeriod, 0.0);
			config.GetOrDefault("demandAmplitude", demandAmplitude, 0.0);
			config.GetOrDefault("demandPeriod", demandPeriod, static_cast<int64_t>(12));

			// === STATIC: Read fixed base-stock level ===
			config.Get("baseStockLevel", baseStockLevel);

			// === DYNAMIC: Read rationing action space ===
			config.GetOrDefault("totalRationingActions", totalRationingActions, static_cast<int64_t>(4));
			config.GetOrDefault("benchmarkRationingAction", benchmarkRationingAction, static_cast<int64_t>(1));

			if (benchmarkRationingAction >= totalRationingActions)
				benchmarkRationingAction = totalRationingActions - 1;

			maxReviewHorizon = 0;
			for (int64_t k = 0; k < numberOfCustomers; k++)
				maxReviewHorizon = std::max(maxReviewHorizon, reviewHorizons[k]);

			// Build the per (customer,item) demand distributions.
			demand_distributions.reserve(numberOfCustomers * numberOfItems);
			for (int64_t idx = 0; idx < numberOfCustomers * numberOfItems; idx++) {
				if (highDemandVariance[idx] == 1)
					demand_distributions.push_back(DynaPlex::DiscreteDist::GetGeometricDist(demandRates[idx]));
				else
					demand_distributions.push_back(DynaPlex::DiscreteDist::GetPoissonDist(demandRates[idx]));
			}
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const
		{
			// Any composite action is feasible: if the chosen central target is
			// below the current inventory position we simply do not order.
			return true;
		}

		double MDP::ModifyStateWithAction(State& state, int64_t action) const
		{
			double cost = 0.0;

			// === STATIC ORDERING ===
			// The action parameter (action) is the RATIONING RULE choice (0,1,2).
			// Ordering is STATIC and uses the FIXED baseStockLevel vector.
			//
			// For each item i (paper Step 2):
			//   IP[i] = OH*[i] + on_the_way[i] - sum_C BO_{C,i}(t)
			//   Q[i] = max(0, baseStockLevel[i] - IP[i])
			//   These orders arrive in (leadTime[i]) periods

			for (int64_t i = 0; i < numberOfItems; i++)
			{
				// Calculate Inventory Position = on-hand + on-the-way - backorders
				int64_t onHand = state.state_vector[i].front();
				int64_t onTheWay = 0;
				auto it = state.state_vector[i].begin();
				++it;  // Skip on-hand, start from pipeline orders
				while (it != state.state_vector[i].end()) {
					onTheWay += *it;
					++it;
				}
				int64_t backorderSum = 0;
				for (int64_t k = 0; k < numberOfCustomers; k++)
					backorderSum += state.backorder[k * numberOfItems + i];
				int64_t ip = onHand + onTheWay - backorderSum;

				// Order to maintain base stock level
				const int64_t toOrder = baseStockLevel[i] - ip;
				if (toOrder >= 0) {
					// Place order (will arrive later due to leadTime)
					if (leadTimes[i] == 0)
						state.state_vector[i].front() += toOrder;
					else
						state.state_vector[i].push_back(toOrder);
					state.inventory_position[i] = ip + toOrder;  // Update IP
				}
				else {
					// No order needed (inventory position >= base stock)
					if (leadTimes[i] != 0)
						state.state_vector[i].push_back(0);
					state.inventory_position[i] = ip;  // IP unchanged
				}
			}

			// Store the rationing action for the allocation phase
			if (action != state.LastRationingAction) {
				state.LastRationingAction = action;
				state.ChangeInAction++;
			}

			state.cat = StateCategory::AwaitEvent();
			return cost;
		}

		MDP::Event MDP::GetEvent(RNG& rng) const
		{
			std::vector<int64_t> events;
			events.reserve(numberOfCustomers * numberOfItems);
			for (int64_t idx = 0; idx < numberOfCustomers * numberOfItems; idx++)
				events.push_back(demand_distributions[idx].GetSample(rng));
			return events;
		}

		double MDP::GetTimeVaryingDemandRate(int64_t customerIdx, int64_t itemIdx, int64_t period) const
		{
			const int64_t flatIdx = customerIdx * numberOfItems + itemIdx;
			double baseRate = demandRates[flatIdx];
			if (demandAmplitude <= 0.0 || demandPeriod <= 0)
				return baseRate;
			// Apply sinusoidal variation: λ(t) = λ_base * (1 + A * sin(2π*t / P))
			const double phase = 2.0 * M_PI * period / demandPeriod;
			return baseRate * (1.0 + demandAmplitude * std::sin(phase));
		}

		double MDP::ModifyStateWithEvent(State& state, const MDP::Event& event) const
		{
			state.cat = StateCategory::AwaitAction();
			double cost = 0.0;
			state.aggregate_vector.clear();

			// === PERIOD WORKFLOW (per docs/paper_improved_complete.tex, Section "Period Workflow") ===
			// STEP 1: RECEIVE ORDERS from (leadTime) periods ago
			// STEP 2: OBSERVE CURRENT IP(t)
			// STEP 3: CALCULATE Q(t) [done in ModifyStateWithAction]
			// STEP 4: OBSERVE DEMANDS D_{k,i}(t) [parameter: event]
			// STEP 5: CHECK IF RATIONING NEEDED (Per Item)
			//   Rationing check: Σ_k BO_k^(i)(t-1) + D_i(t) > OH_i(t)?
			// STEP 6: ALLOCATIONS A_{k,i}(t) → CALL π (if rationing needed)
			// STEP 7: UPDATE INVENTORY & BACKORDERS
			// STEP 8: CALCULATE COST
			// STEP 9: UPDATE PER-CUSTOMER BACKORDER TRACKING (CUMULATIVE OVER REVIEW CYCLE)

			state.Period++;
			state.current_demand = event;
			state.current_allocation.assign(numberOfCustomers * numberOfItems, 0);
			state.current_stockouts.assign(numberOfCustomers * numberOfItems, 0);

			// STEP 5 (Precomputation): Determine allocation priority order based on action
			// Action 0: FCFS (customer-index order)
			// Action 1: SLA-gap myopic (prioritize customer furthest from target)
			// Action 2: Proportional (allocate by demand share)
			// Action 3: Cost-based greedy (allocate to customer with highest SLA-penalty risk)
			std::vector<int64_t> order(numberOfCustomers);
			for (int64_t k = 0; k < numberOfCustomers; k++)
				order[k] = k;

			if (state.LastRationingAction == 1) {
				std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
					int64_t exceedA = static_cast<int64_t>(std::max(0.0, state.cumulative_backorder[a] - backorderAllowances[a]));
					int64_t exceedB = static_cast<int64_t>(std::max(0.0, state.cumulative_backorder[b] - backorderAllowances[b]));
					return exceedA > exceedB;  // Prioritize customer with most excess backorder
					});
			}

			// Precompute SLA allowances for Action 3 (cost-based greedy)
			// For Action 3, we use the backorder allowances directly
			std::vector<int64_t> maxAllowed(numberOfCustomers, 0);
			if (state.LastRationingAction == 3) {
				for (int64_t k = 0; k < numberOfCustomers; k++) {
					maxAllowed[k] = backorderAllowances[k];
				}
			}

			std::vector<int64_t> periodStockouts(numberOfCustomers, 0);

			// Each customer's total backorder across items, as of the start of
			// this period (= BO_C(t) = sum_i BO_{C,i}(t), per the paper).
			// Used by Action 3's exploratory heuristic below.
			std::vector<int64_t> priorBackorderTotal(numberOfCustomers, 0);
			for (int64_t k = 0; k < numberOfCustomers; k++)
				for (int64_t i = 0; i < numberOfItems; i++)
					priorBackorderTotal[k] += state.backorder[k * numberOfItems + i];

			// === STEP 1-5: Process each item ===
			for (int64_t i = 0; i < numberOfItems; i++)
			{
				auto& currentState = state.state_vector[i];
				int64_t oh = currentState.pop_front();  // STEP 1-2: On-hand at start of period
				state.inventory_level_before_allocation[i] = oh;  // OH*_i(t): available supply before allocation

				// STEP 4-5: What each customer is owed for this item - prior
				// backorder BO_{k,i}(t) plus this period's realized demand
				// D_{k,i}(t). This is what Step 5 rations against, and what a
				// "no rationing" period allocates in full.
				std::vector<int64_t> owed(numberOfCustomers, 0);
				int64_t totalOwed = 0;
				for (int64_t k = 0; k < numberOfCustomers; k++) {
					const int64_t d_ki = event[k * numberOfItems + i];
					owed[k] = state.backorder[k * numberOfItems + i] + d_ki;
					totalOwed += owed[k];
				}

				// STEP 5: Check if rationing needed
				// Rationing check: Demand_i(t) = Σ_C [BO_{C,i}(t) + D_{C,i}(t)] > Available_i(t) = OH*_i(t)?
				bool needsRationing = totalOwed > oh;
				const int64_t available = oh > 0 ? oh : 0;

				// STEP 6: ALLOCATION based on rationing rule
				if (!needsRationing || available >= totalOwed) {
					// No rationing needed: full allocation clears all backlog + new demand
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						state.current_allocation[k * numberOfItems + i] = owed[k];
						state.current_stockouts[k * numberOfItems + i] = 0;
						state.backorder[k * numberOfItems + i] = 0;
					}
				}
				else if (state.LastRationingAction == 2) {
					// Action 2: Proportional rationing (against backlog + new demand)
					std::vector<int64_t> served(numberOfCustomers, 0);
					int64_t handedOut = 0;
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						if (totalOwed > 0)
							served[k] = (available * owed[k]) / totalOwed;
						handedOut += served[k];
					}
					int64_t remainder = available - handedOut;
					std::vector<int64_t> fracOrder(numberOfCustomers);
					for (int64_t k = 0; k < numberOfCustomers; k++)
						fracOrder[k] = k;
					std::sort(fracOrder.begin(), fracOrder.end(), [&](int64_t a, int64_t b) {
						return (available * owed[a]) % totalOwed > (available * owed[b]) % totalOwed;
						});
					for (int64_t pos = 0; pos < numberOfCustomers && remainder > 0; pos++) {
						const int64_t k = fracOrder[pos];
						if (served[k] < owed[k]) { served[k]++; remainder--; }
					}
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						state.current_allocation[k * numberOfItems + i] = served[k];
						state.current_stockouts[k * numberOfItems + i] = owed[k] - served[k];
						state.backorder[k * numberOfItems + i] = owed[k] - served[k];
						periodStockouts[k] += owed[k] - served[k];
					}
				}
				else if (state.LastRationingAction == 3) {
					// Action 3: Cost-based greedy (exploratory, outside paper scope)
					std::vector<int64_t> notServedSoFar(numberOfCustomers, 0);
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						notServedSoFar[k] = owed[k];
					}

					for (int64_t unit = 0; unit < available; unit++) {
						int64_t bestK = -1;
						double bestCost = -1.0;

						for (int64_t k = 0; k < numberOfCustomers; k++) {
							if (notServedSoFar[k] > 0) {
								// Backorder on OTHER items (excludes this item's stale
								// pre-period value, already folded into notServedSoFar)
								// plus already-finalized backorder from earlier items
								// processed this period.
								const int64_t priorC = (priorBackorderTotal[k] - state.backorder[k * numberOfItems + i]) + periodStockouts[k];
								const int64_t totalWithheld = priorC + notServedSoFar[k];
								const double marginalCost = (totalWithheld > maxAllowed[k]) ? penaltyCosts[k] : 0.0;
								if (marginalCost > bestCost) {
									bestCost = marginalCost;
									bestK = k;
								}
							}
						}

						if (bestK >= 0) {
							notServedSoFar[bestK]--;
							state.current_allocation[bestK * numberOfItems + i]++;
						}
					}

					for (int64_t k = 0; k < numberOfCustomers; k++) {
						state.current_stockouts[k * numberOfItems + i] = notServedSoFar[k];
						state.backorder[k * numberOfItems + i] = notServedSoFar[k];
						periodStockouts[k] += notServedSoFar[k];
					}
				}
				else {
					// Action 0 (FCFS) or Action 1 (SLA-gap): serve sequentially against backlog + new demand
					int64_t remaining = available;
					for (int64_t pos = 0; pos < numberOfCustomers; pos++) {
						const int64_t k = order[pos];
						const int64_t served = std::min(remaining, owed[k]);
						remaining -= served;
						state.current_allocation[k * numberOfItems + i] = served;
						state.current_stockouts[k * numberOfItems + i] = owed[k] - served;
						state.backorder[k * numberOfItems + i] = owed[k] - served;
						periodStockouts[k] += owed[k] - served;
					}
				}

				// STEP 7-8: UPDATE INVENTORY AND COSTS
				// OH_i(t+1) = OH*_i(t) - Σ_C A_{C,i}(t): allocation is always
				// bounded by available supply, so on-hand never goes negative -
				// unmet demand is tracked separately via state.backorder.
				int64_t periodAllocationThisItem = 0;
				for (int64_t k = 0; k < numberOfCustomers; k++)
					periodAllocationThisItem += state.current_allocation[k * numberOfItems + i];
				const int64_t newOnHand = oh - periodAllocationThisItem;
				state.inventory_level[i] = newOnHand;
				state.inventory_position[i] -= periodAllocationThisItem;

				// Holding cost (paper Step 8): h_i * OH*_i(t), charged on the
				// inventory available before allocation - the stock actually
				// carried this period - not on what's left after allocation.
				if (oh > 0) {
					cost += static_cast<double>(oh) * holdingCosts[i];
				}

				// Update queue for next period
				if (leadTimes[i] == 0)
					currentState.push_back(newOnHand);
				else
					currentState.front() += newOnHand;

				state.aggregate_vector.insert(state.aggregate_vector.end(), currentState.begin(), currentState.end());
			}

			// STEP 8-9: Per-customer SLA bookkeeping and costs
			for (int64_t k = 0; k < numberOfCustomers; k++)
			{
				// STEP 7/9: BO_C(t+1) = sum_i BO_{C,i}(t+1) (per-item backorder was
				// already updated to its new value in the item loop above).
				// cumulative_backorder[k] (= BO-bar_k) accumulates this over the
				// window [HorizonStartPeriod[k], state.Period] of the current
				// horizon r_k = ReviewHorizonIndex[k], and resets to 0 when that
				// horizon ends. We only ever track backorder accumulation here -
				// not raw demand or stockout accumulation.
				int64_t newBO_C = 0;
				for (int64_t i = 0; i < numberOfItems; i++)
					newBO_C += state.backorder[k * numberOfItems + i];
				state.cumulative_backorder[k] += static_cast<double>(newBO_C);

				state.TimeRemaining[k]--;
				if (state.TimeRemaining[k] == 0)
				{
					// STEP 8: Review cycle ends - apply SLA penalty based on cumulative backorder allowance
					state.NumReviewPeriodPassed[k]++;
					const int64_t exceededBackorders = static_cast<int64_t>(std::max(0.0, state.cumulative_backorder[k] - backorderAllowances[k]));
					const double n = static_cast<double>(state.NumReviewPeriodPassed[k]);
					if (exceededBackorders > 0) {
						cost += exceededBackorders * penaltyCosts[k];
						state.ShortfallPerReviewPeriod[k] = (state.ShortfallPerReviewPeriod[k] * (n - 1) + (double)exceededBackorders) / n;
						state.SuccessPerReviewPeriod[k] = (state.SuccessPerReviewPeriod[k] * (n - 1) + 0.0) / n;
					}
					else {
						state.ShortfallPerReviewPeriod[k] = (state.ShortfallPerReviewPeriod[k] * (n - 1) + 0.0) / n;
						state.SuccessPerReviewPeriod[k] = (state.SuccessPerReviewPeriod[k] * (n - 1) + 1.0) / n;
					}

					// Reset cumulative backorder sum and counters for next review cycle
					// (note: state.backorder itself is NOT reset - it persists)
					state.cumulative_backorder[k] = 0.0;
					state.TimeRemaining[k] = reviewHorizons[k];

					// Advance to the next review horizon: r_k(t) -> r_k(t)+1,
					// with the new horizon starting next period (t+1).
					state.ReviewHorizonIndex[k]++;
					state.HorizonStartPeriod[k] = state.Period + 1;
				}
			}

			return cost - unavoidableCostPerPeriod;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			// Add inventory level and position per item (for DCL feature learning)
			// OH(t) shows crisis (can be negative), OH*(t) is pre-allocation
			// available supply, IP(t) shows recovery (pipeline)
			for (int64_t i = 0; i < numberOfItems; i++) {
				features.Add(static_cast<float>(state.inventory_level[i]));
				features.Add(static_cast<float>(state.inventory_level_before_allocation[i]));
				features.Add(static_cast<float>(state.inventory_position[i]));
			}

			features.Add(state.aggregate_vector);

			// D_{C,i}(t): realized demand THIS period, per customer-item pair.
			// No accumulation - only this period's demand is exposed, so DCL
			// can react to it when choosing the next rationing action.
			for (int64_t k = 0; k < numberOfCustomers; k++)
				for (int64_t i = 0; i < numberOfItems; i++)
					features.Add(static_cast<float>(state.current_demand[k * numberOfItems + i]));

			for (int64_t k = 0; k < numberOfCustomers; k++)
			{
				// Progress toward the customer's numeric backorder target (beta_k):
				// cumulative backorder so far relative to the allowance.
				features.Add(static_cast<float>(state.cumulative_backorder[k]) / static_cast<float>(backorderAllowances[k]));
				features.Add(static_cast<float>(state.TimeRemaining[k]) / static_cast<float>(reviewHorizons[k]));
			}
		}

		std::vector<double> MDP::ReturnUsefulStatistics(const State& state) const
		{
			std::vector<double> statistics;
			statistics.reserve(2 * numberOfCustomers + 1);
			for (int64_t k = 0; k < numberOfCustomers; k++)
				statistics.push_back(state.ShortfallPerReviewPeriod[k]);
			for (int64_t k = 0; k < numberOfCustomers; k++)
				statistics.push_back(state.cumulative_backorder[k]);
			double avgSuccess = 0.0;
			for (int64_t k = 0; k < numberOfCustomers; k++)
				avgSuccess += state.SuccessPerReviewPeriod[k];
			statistics.push_back(avgSuccess / static_cast<double>(numberOfCustomers));
			return statistics;
		}

		void MDP::ResetHiddenStateVariables(State& state, RNG& rng) const
		{
			state.backorder.assign(numberOfCustomers * numberOfItems, 0);
			for (int64_t k = 0; k < numberOfCustomers; k++)
			{
				state.TimeRemaining[k] = reviewHorizons[k];
				state.NumReviewPeriodPassed[k] = 0;
				state.ShortfallPerReviewPeriod[k] = 0.0;
				state.SuccessPerReviewPeriod[k] = 1.0;
				state.cumulative_backorder[k] = 0.0;
				state.ReviewHorizonIndex[k] = 1;
				state.HorizonStartPeriod[k] = state.Period + 1;
			}
			state.ChangeInAction = 0;
		}

		MDP::State MDP::GetInitialState() const
		{
			State state{};
			state.cat = StateCategory::AwaitAction();
			state.state_vector.reserve(numberOfItems);
			state.inventory_position.reserve(numberOfItems);
			int64_t aggregateVectorLength = 0;

			// Initialize central inventory at base-stock level
			for (int64_t i = 0; i < numberOfItems; i++) {
				auto queue = Queue<int64_t>{};
				queue.reserve(leadTimes[i] + 1);
				queue.push_back(baseStockLevel[i]);  // initial IL[i] = base stock level
				for (int64_t j = 0; j < leadTimes[i] - 1; j++)
					queue.push_back(0);  // no orders in pipeline initially
				state.state_vector.push_back(queue);
				state.inventory_level.push_back(baseStockLevel[i]);
				state.inventory_level_before_allocation.push_back(baseStockLevel[i]);
				state.inventory_position.push_back(baseStockLevel[i]);
				aggregateVectorLength += leadTimes[i] == 0 ? 1 : leadTimes[i];
			}

			state.aggregate_vector.reserve(aggregateVectorLength);
			for (int64_t i = 0; i < numberOfItems; i++)
				state.aggregate_vector.insert(state.aggregate_vector.end(), state.state_vector[i].begin(), state.state_vector[i].end());

			// Sized (not empty): GetFeatures reads current_demand before any
			// event has been incorporated (on the very first AwaitAction state).
			state.current_demand.assign(numberOfCustomers * numberOfItems, 0);
			state.current_allocation.assign(numberOfCustomers * numberOfItems, 0);
			state.current_stockouts.assign(numberOfCustomers * numberOfItems, 0);
			state.Period = 0;

			// Initialize per-customer SLA tracking
			state.NumReviewPeriodPassed.assign(numberOfCustomers, 0);
			state.ShortfallPerReviewPeriod.assign(numberOfCustomers, 0.0);
			state.SuccessPerReviewPeriod.assign(numberOfCustomers, 1.0);
			state.TimeRemaining.assign(numberOfCustomers, 0);
			for (int64_t k = 0; k < numberOfCustomers; k++)
				state.TimeRemaining[k] = reviewHorizons[k];

			// Review-horizon indexing: every customer starts in horizon r=1,
			// which begins at period t=1 (state.Period is incremented to 1 on
			// the first call to ModifyStateWithEvent).
			state.ReviewHorizonIndex.assign(numberOfCustomers, 1);
			state.HorizonStartPeriod.assign(numberOfCustomers, 1);

			// Initialize backorder tracking: per-item balance BO_{C,i}, and the
			// horizon-scoped cumulative sum BO-bar_C.
			state.backorder.assign(numberOfCustomers * numberOfItems, 0);
			state.cumulative_backorder.assign(numberOfCustomers, 0.0);

			// Initialize rationing action tracking
			state.LastRationingAction = benchmarkRationingAction;
			state.ChangeInAction = 0;
			state.AllowedActions.assign(totalRationingActions, true);

			return state;
		}

		MDP::State MDP::GetState(const DynaPlex::VarGroup& vars) const
		{
			State state{};
			vars.Get("cat", state.cat);
			vars.Get("aggregate_vector", state.aggregate_vector);
			vars.Get("inventory_position", state.inventory_position);
			vars.Get("TimeRemaining", state.TimeRemaining);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("aggregate_vector", aggregate_vector);
			vars.Add("inventory_position", inventory_position);
			vars.Add("TimeRemaining", TimeRemaining);
			return vars;
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			return state.cat;
		}

		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{
			registry.Register<BaseStockPolicy>("base_stock",
				"Static composite action (fixed base-stock levels) for all customers.");
			registry.Register<GreedyDynamicPolicy>("greedy_dynamic",
				"Rule-based dynamic composite-action policy reacting to per-customer cumulative backorder relative to its allowance.");
		}

		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
				/*=id though which the MDP will be retrievable*/ "multi_customer_sla",
				/*description*/ "Multi-customer multi-item inventory management with a shared central warehouse under heterogeneous service level agreements. Central inventory is allocated to customers upon demand realization using a rationing rule when total demand exceeds available stock.",
				/*reference to passed registry*/registry);
		}
	}
}
