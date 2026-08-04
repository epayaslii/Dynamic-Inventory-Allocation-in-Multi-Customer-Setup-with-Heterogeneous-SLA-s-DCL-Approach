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
			config.GetOrDefault("backOrderCost", backOrderCost, 0.0);
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
			// For each item i:
			//   IP[i] = on_hand[i] + on_the_way[i]  (sum of state_vector[i])
			//   Q[i] = max(0, baseStockLevel[i] - IP[i])
			//   These orders arrive in (leadTime[i]) periods

			for (int64_t i = 0; i < numberOfItems; i++)
			{
				// Calculate Inventory Position = on-hand + on-the-way
				int64_t onHand = state.state_vector[i].front();
				int64_t onTheWay = 0;
				auto it = state.state_vector[i].begin();
				++it;  // Skip on-hand, start from pipeline orders
				while (it != state.state_vector[i].end()) {
					onTheWay += *it;
					++it;
				}
				int64_t ip = onHand + onTheWay;

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

			// === 9-STEP PERIOD WORKFLOW (Per work_summary.md Model) ===
			// STEP 1: RECEIVE ORDERS from (leadTime) periods ago
			// STEP 2: OBSERVE CURRENT IP(t)
			// STEP 3: CALCULATE Q(t) [done in ModifyStateWithAction]
			// STEP 4: OBSERVE DEMANDS D_{k,i}(t) [parameter: event]
			// STEP 5: CHECK IF RATIONING NEEDED (Per Item)
			//   Rationing check: Σ_k BO_k^(i)(t-1) + D_i(t) > OH_i(t)?
			// STEP 6: ALLOCATIONS A_{k,i}(t) → CALL π (if rationing needed)
			// STEP 7: UPDATE INVENTORY & BACKORDERS
			// STEP 8: CALCULATE COST
			// STEP 9: UPDATE AFR (CUMULATIVE OVER REVIEW CYCLE)

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
			std::vector<int64_t> periodDemand(numberOfCustomers, 0);

			// === STEP 1-5: Process each item ===
			for (int64_t i = 0; i < numberOfItems; i++)
			{
				auto& currentState = state.state_vector[i];
				int64_t oh = currentState.pop_front();  // STEP 1-2: On-hand at start of period

				// STEP 4: Total demand for this item across all customers
				int64_t totalDemand = 0;
				for (int64_t k = 0; k < numberOfCustomers; k++)
					totalDemand += event[k * numberOfItems + i];

				for (int64_t k = 0; k < numberOfCustomers; k++)
					periodDemand[k] += event[k * numberOfItems + i];

				// STEP 5: Check if rationing needed
				// Rationing check: Σ_k BO_k^(i)(t-1) + D_i(t) > OH_i(t)?
				// This checks if total backlog plus new demand exceeds available on-hand
				int64_t boSum = 0;
				for (int64_t k = 0; k < numberOfCustomers; k++) {
					// BO_k^(i) comes from cumulative_backorder, but we track per-customer total
					// For per-item backorder, we estimate from period stockouts
					// Approximate: use CumulativeStockouts as proxy for backorder level
					boSum += state.CumulativeStockouts[k];
				}

				bool needsRationing = (boSum + totalDemand) > oh;
				const int64_t available = oh > 0 ? oh : 0;

				// STEP 6: ALLOCATION based on rationing rule
				if (!needsRationing || available >= totalDemand) {
					// No rationing needed: full allocation
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						const int64_t demand_ki = event[k * numberOfItems + i];
						state.current_allocation[k * numberOfItems + i] = demand_ki;
						state.current_stockouts[k * numberOfItems + i] = 0;
					}
				}
				else if (state.LastRationingAction == 2) {
					// Action 2: Proportional rationing
					std::vector<int64_t> served(numberOfCustomers, 0);
					int64_t handedOut = 0;
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						const int64_t demand_ki = event[k * numberOfItems + i];
						if (totalDemand > 0)
							served[k] = (available * demand_ki) / totalDemand;
						handedOut += served[k];
					}
					int64_t remainder = available - handedOut;
					std::vector<int64_t> fracOrder(numberOfCustomers);
					for (int64_t k = 0; k < numberOfCustomers; k++)
						fracOrder[k] = k;
					std::sort(fracOrder.begin(), fracOrder.end(), [&](int64_t a, int64_t b) {
						const int64_t da = event[a * numberOfItems + i];
						const int64_t db = event[b * numberOfItems + i];
						return (available * da) % totalDemand > (available * db) % totalDemand;
						});
					for (int64_t pos = 0; pos < numberOfCustomers && remainder > 0; pos++) {
						const int64_t k = fracOrder[pos];
						const int64_t demand_ki = event[k * numberOfItems + i];
						if (served[k] < demand_ki) { served[k]++; remainder--; }
					}
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						const int64_t demand_ki = event[k * numberOfItems + i];
						state.current_allocation[k * numberOfItems + i] = served[k];
						state.current_stockouts[k * numberOfItems + i] = demand_ki - served[k];
						periodStockouts[k] += demand_ki - served[k];
					}
				}
				else if (state.LastRationingAction == 3) {
					// Action 3: Cost-based greedy
					std::vector<int64_t> notServedSoFar(numberOfCustomers, 0);
					for (int64_t k = 0; k < numberOfCustomers; k++) {
						notServedSoFar[k] = event[k * numberOfItems + i];
					}

					for (int64_t unit = 0; unit < available; unit++) {
						int64_t bestK = -1;
						double bestCost = -1.0;

						for (int64_t k = 0; k < numberOfCustomers; k++) {
							if (notServedSoFar[k] > 0) {
								const int64_t priorC = state.CumulativeStockouts[k] + periodStockouts[k];
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
						const int64_t demand_ki = event[k * numberOfItems + i];
						state.current_stockouts[k * numberOfItems + i] = notServedSoFar[k];
						periodStockouts[k] += notServedSoFar[k];
					}
				}
				else {
					// Action 0 (FCFS) or Action 1 (SLA-gap): serve sequentially
					int64_t remaining = available;
					for (int64_t pos = 0; pos < numberOfCustomers; pos++) {
						const int64_t k = order[pos];
						const int64_t demand_ki = event[k * numberOfItems + i];
						const int64_t served = std::min(remaining, demand_ki);
						remaining -= served;
						state.current_allocation[k * numberOfItems + i] = served;
						state.current_stockouts[k * numberOfItems + i] = demand_ki - served;
						periodStockouts[k] += demand_ki - served;
					}
				}

				// STEP 7-8: UPDATE INVENTORY AND COSTS
				const int64_t newOnHand = oh - totalDemand;
				state.inventory_level[i] = newOnHand;
				state.inventory_position[i] -= totalDemand;

				// Holding cost: h_i * OH_i(t+1) for leftover stock
				if (newOnHand > 0) {
					cost += newOnHand * holdingCosts[i];
				}

				// Backorder cost: b * max(0, BO_k(t) - BO_k(t-1)) for this item
				// This is tracked per-customer in the cost calculation below

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
				// Calculate period allocation and demand for this customer
				int64_t periodAllocation = 0;
				for (int64_t i = 0; i < numberOfItems; i++)
					periodAllocation += state.current_allocation[k * numberOfItems + i];

				// STEP 7: Update backorder: BO_k(t) = BO_k(t-1) + D_k(t) - A_k(t)
				double previousBO = state.cumulative_backorder[k];
				state.cumulative_backorder[k] += (periodDemand[k] - periodAllocation);
				int64_t newDebt = static_cast<int64_t>(std::max(0.0, state.cumulative_backorder[k] - previousBO));

				// STEP 8: Add backorder cost for new debt this period
				// Cost: b_k * max(0, BO_k(t) - BO_k(t-1))
				cost += backOrderCost * newDebt;

				// STEP 9: Update AFR tracking
				state.ObservedDemand[k] += periodDemand[k];
				state.CumulativeStockouts[k] += periodStockouts[k];
				if (state.ObservedDemand[k] > 0)
					state.AggregateFillRate[k] = static_cast<double>(state.ObservedDemand[k] - state.CumulativeStockouts[k]) / static_cast<double>(state.ObservedDemand[k]);
				else
					state.AggregateFillRate[k] = 1.0;

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
					state.AFRPerReviewPeriod[k] = (state.AFRPerReviewPeriod[k] * (n - 1) + state.AggregateFillRate[k]) / n;

					// Reset cumulative backorder and counters for next review cycle
					state.cumulative_backorder[k] = 0.0;
					state.ObservedDemand[k] = 0;
					state.CumulativeStockouts[k] = 0;
					state.AggregateFillRate[k] = 1.0;
					state.TimeRemaining[k] = reviewHorizons[k];
				}
			}

			// Per-customer SLA bookkeeping and end-of-horizon penalties.
			for (int64_t k = 0; k < numberOfCustomers; k++)
			{
				// Calculate period allocation for this customer (sum across items)
				int64_t periodAllocation = 0;
				for (int64_t i = 0; i < numberOfItems; i++)
					periodAllocation += state.current_allocation[k * numberOfItems + i];

				// Whiteboard: BO_i(t) = BO_i(t-1) + D_i(t) - a_i(t)
				// Accumulate backorder (unmet demand per customer)
				state.cumulative_backorder[k] += (periodDemand[k] - periodAllocation);

				state.ObservedDemand[k] += periodDemand[k];
				state.CumulativeStockouts[k] += periodStockouts[k];
				if (state.ObservedDemand[k] > 0)
					state.AggregateFillRate[k] = static_cast<double>(state.ObservedDemand[k] - state.CumulativeStockouts[k]) / static_cast<double>(state.ObservedDemand[k]);
				else
					state.AggregateFillRate[k] = 1.0;

				state.TimeRemaining[k]--;
				if (state.TimeRemaining[k] == 0)
				{
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
					state.AFRPerReviewPeriod[k] = (state.AFRPerReviewPeriod[k] * (n - 1) + state.AggregateFillRate[k]) / n;

					state.cumulative_backorder[k] = 0.0;
					state.ObservedDemand[k] = 0;
					state.CumulativeStockouts[k] = 0;
					state.AggregateFillRate[k] = 1.0;
					state.TimeRemaining[k] = reviewHorizons[k];
				}
			}

			return cost - unavoidableCostPerPeriod;
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			// Add inventory level and position per item (for DCL feature learning)
			// Whiteboard signals: IL(t) shows crisis (can be negative), IP(t) shows recovery (pipeline)
			for (int64_t i = 0; i < numberOfItems; i++) {
				features.Add(static_cast<float>(state.inventory_level[i]));
				features.Add(static_cast<float>(state.inventory_position[i]));
			}

			features.Add(state.aggregate_vector);
			for (int64_t k = 0; k < numberOfCustomers; k++)
			{
				const int64_t elapsed = reviewHorizons[k] - state.TimeRemaining[k];
				if (elapsed > 0) {
					features.Add(state.AggregateFillRate[k]);
					features.Add(static_cast<float>(state.CumulativeStockouts[k]) / static_cast<float>(customerDemandRates[k] * elapsed));
					features.Add(static_cast<float>(state.ObservedDemand[k]) / static_cast<float>(customerDemandRates[k] * elapsed));
				}
				else {
					features.Add(1.0);
					features.Add(0.0);
					features.Add(0.0);
				}
				features.Add(static_cast<float>(state.TimeRemaining[k]) / static_cast<float>(reviewHorizons[k]));
			}
		}

		std::vector<double> MDP::ReturnUsefulStatistics(const State& state) const
		{
			std::vector<double> statistics;
			statistics.reserve(3 * numberOfCustomers + 1);
			for (int64_t k = 0; k < numberOfCustomers; k++)
				statistics.push_back(state.AFRPerReviewPeriod[k]);
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
			for (int64_t k = 0; k < numberOfCustomers; k++)
			{
				state.ObservedDemand[k] = 0;
				state.CumulativeStockouts[k] = 0;
				state.AggregateFillRate[k] = 1.0;
				state.TimeRemaining[k] = reviewHorizons[k];
				state.NumReviewPeriodPassed[k] = 0;
				state.AFRPerReviewPeriod[k] = 1.0;
				state.ShortfallPerReviewPeriod[k] = 0.0;
				state.SuccessPerReviewPeriod[k] = 1.0;
				state.cumulative_backorder[k] = 0.0;
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
				state.inventory_position.push_back(baseStockLevel[i]);
				aggregateVectorLength += leadTimes[i] == 0 ? 1 : leadTimes[i];
			}

			state.aggregate_vector.reserve(aggregateVectorLength);
			for (int64_t i = 0; i < numberOfItems; i++)
				state.aggregate_vector.insert(state.aggregate_vector.end(), state.state_vector[i].begin(), state.state_vector[i].end());

			state.current_demand.clear();
			state.current_allocation.assign(numberOfCustomers * numberOfItems, 0);
			state.current_stockouts.assign(numberOfCustomers * numberOfItems, 0);
			state.Period = 0;

			// Initialize per-customer SLA tracking
			state.ObservedDemand.assign(numberOfCustomers, 0);
			state.CumulativeStockouts.assign(numberOfCustomers, 0);
			state.AggregateFillRate.assign(numberOfCustomers, 1.0);
			state.NumReviewPeriodPassed.assign(numberOfCustomers, 0);
			state.AFRPerReviewPeriod.assign(numberOfCustomers, 1.0);
			state.ShortfallPerReviewPeriod.assign(numberOfCustomers, 0.0);
			state.SuccessPerReviewPeriod.assign(numberOfCustomers, 1.0);
			state.TimeRemaining.assign(numberOfCustomers, 0);
			for (int64_t k = 0; k < numberOfCustomers; k++)
				state.TimeRemaining[k] = reviewHorizons[k];

			// Initialize cumulative backorder tracking
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
			vars.Get("ObservedDemand", state.ObservedDemand);
			vars.Get("CumulativeStockouts", state.CumulativeStockouts);
			vars.Get("AggregateFillRate", state.AggregateFillRate);
			vars.Get("TimeRemaining", state.TimeRemaining);
			return state;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("aggregate_vector", aggregate_vector);
			vars.Add("inventory_position", inventory_position);
			vars.Add("ObservedDemand", ObservedDemand);
			vars.Add("CumulativeStockouts", CumulativeStockouts);
			vars.Add("AggregateFillRate", AggregateFillRate);
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
				"Rule-based dynamic composite-action policy reacting to per-customer AFR.");
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
