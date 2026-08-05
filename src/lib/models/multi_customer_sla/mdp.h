#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/modelling/queue.h"

namespace DynaPlex::Models {
	namespace multi_customer_sla
	{
		// Multi-customer, multi-item inventory system with service level agreements.
		//
		// A single supplier serves K customers from ONE shared central warehouse
		// that stocks |I| items (SKUs). Customers do NOT exchange stock (no lateral
		// transshipment).
		//
		// STATIC ORDERING: The supplier maintains a FIXED base-stock level per item.
		// Each period:
		//   1. Receive orders from (leadTime) periods ago
		//   2. Calculate Inventory Position: IP[i] = on_hand[i] + on_the_way[i]
		//   3. Order: Q[i] = max(0, BSL[i] - IP[i])
		//   4. Orders placed now will arrive in (leadTime) periods
		//
		// DYNAMIC ALLOCATION: When total customer demand > available on-hand stock,
		// a rationing rule decides which customer gets served. DCL learns which
		// rationing rule (action) works best given the current state.
		//
		// Each customer k is bound by its own SLA: a cumulative backorder allowance β_k
		// (maximum tolerated backorder units per review horizon), a review-horizon length,
		// and a linear shortfall penalty. At the end of each customer's review horizon,
		// a penalty is charged if cumulative backorders exceed β_k.
		class MDP
		{
		public:
			int64_t numberOfCustomers;
			int64_t numberOfItems;

			// Per-item (central) data.
			std::vector<int64_t> leadTimes;          // size |I|
			std::vector<double> holdingCosts;        // size |I|

			// Per-customer SLA data (size K).
			std::vector<int64_t> backorderAllowances;  // β_k: max cumulative backorder units per customer per review horizon
			std::vector<int64_t> reviewHorizons;
			std::vector<double> penaltyCosts;
			std::vector<double> customerDemandRates;  // total demand rate of customer k
			int64_t maxReviewHorizon;

			// Per (customer,item) demand, flattened as [k * |I| + i].
			// demandRates[k*|I| + i] = base demand rate for customer k, item i
			std::vector<double> demandRates;
			std::vector<int64_t> highDemandVariance;
			std::vector<DynaPlex::DiscreteDist> demand_distributions;

			// Time-varying demand support: optional amplitude and period for sinusoidal variation
			// If demandAmplitude > 0: λ_{k,i}(t) = demandRates[...] * (1 + demandAmplitude * sin(2π*t/demandPeriod))
			double demandAmplitude;      // seasonal variation amplitude (e.g., 0.2 for ±20%)
			int64_t demandPeriod;        // period of seasonal cycle (e.g., 12 for monthly seasonality)

			double unavoidableCostPerPeriod;

			// === STATIC ORDERING: Fixed base-stock level per item ===
			// baseStockLevel[i] = target inventory position for item i
			// Order policy: every period, order = baseStockLevel[i] - inventory_position[i]
			std::vector<int64_t> baseStockLevel;

			// === DYNAMIC ALLOCATION: Actions are rationing rules ===
			// totalRationingActions = how many different rationing rules to try (4 actions)
			// Action 0: FCFS (first-come-first-served, customer index order)
			// Action 1: SLA-gap myopic (prioritize customer furthest from SLA target)
			// Action 2: Proportional (allocate proportional to demand, largest-remainder method)
			// Action 3: Cost-based greedy (assign each unit to customer with highest SLA-penalty risk)
			int64_t totalRationingActions;
			int64_t benchmarkRationingAction;

			struct State {
				DynaPlex::StateCategory cat;

				// === Central inventory tracking ===
				// INVENTORY LEVEL (IL) vs INVENTORY POSITION (IP):
				//
				// Inventory Level IL[i]:
				//   Physical on-hand stock at the warehouse (can be negative = backorder).
				//   Used for allocation decisions: how much is available to ration to customers.
				//   IL[i] = state_vector[i].front()
				//
				// Inventory Position IP[i]:
				//   IL[i] + sum of all orders in transit (pipeline).
				//   Used for replenishment: Q[i] = max(0, BSL[i] - IP[i]).
				//   Always ≥ IL[i] (pipeline quantities are non-negative).
				//
				// state_vector[i] = Queue with IL and pipeline per item i:
				//   - state_vector[i][0] = IL[i] (on-hand, can be negative)
				//   - state_vector[i][1..leadTime] = Q[i](t-1), Q[i](t-2), ... (arriving soon)
				//
				// aggregate_vector = flattened state_vector (for features/learning)
				std::vector<Queue<int64_t>> state_vector;
				std::vector<int64_t> inventory_level;
				// OH*_i(t): on-hand inventory before allocation (per the paper) -
				// available supply for rationing this period, after order arrival
				// but before allocations are subtracted.
				std::vector<int64_t> inventory_level_before_allocation;
				std::vector<int64_t> inventory_position;
				std::vector<int64_t> aggregate_vector;

				// Current period allocation and demand (for rationing rule transparency)
				// current_demand[k*|I| + i] = d_{k,i}(t) (realized demand)
				// current_allocation[k*|I| + i] = Alloc_{k,i}(t) (units allocated)
				// current_stockouts[k*|I| + i] = SO_{k,i}(t) = max(0, d_{k,i}(t) - Alloc_{k,i}(t))
				std::vector<int64_t> current_demand;
				std::vector<int64_t> current_allocation;
				std::vector<int64_t> current_stockouts;

				// Per-customer bookkeeping within the current review horizon (size K).
				// We only accumulate backorder here - not raw demand or stockouts.
				std::vector<int64_t> TimeRemaining;
				std::vector<int64_t> NumReviewPeriodPassed;
				std::vector<double> ShortfallPerReviewPeriod;
				std::vector<double> SuccessPerReviewPeriod;

				// === Review-horizon indexing (per the paper's t, T, r, r(t) notation) ===
				// r_k(t): index of the review horizon customer k is currently in
				// (r = 1, 2, 3, ...; increments each time customer k's horizon ends).
				std::vector<int64_t> ReviewHorizonIndex;
				// Period at which customer k's CURRENT horizon began, i.e.
				// (r_k(t) - 1) * T_k + 1. Together with Period (t) below, this
				// bounds the window [HorizonStartPeriod[k], t-1] that
				// cumulative_backorder[k] (= BO-bar_k(t)) sums over.
				std::vector<int64_t> HorizonStartPeriod;

				// === Per-customer-item backorder balance (per the paper's Step 7) ===
				// BO_{C,i}(t+1) = BO_{C,i}(t) + D_{C,i}(t) - A_{C,i}(t), flattened
				// as [k*|I|+i]. This is a running balance that persists across
				// review-horizon boundaries (only cleared by actual allocation) -
				// it is NOT reset when a horizon ends.
				std::vector<int64_t> backorder;

				// === Per-customer cumulative backorder SUM tracking ===
				// Per the paper: BO_C(t) = sum_i BO_{C,i}(t) (aggregate per
				// customer), and BO-bar_C(t) = sum_{j=HorizonStartPeriod[C]}^{t-1}
				// BO_C(j) - i.e. the sum of the (persisting) BO_C balance across
				// periods within the CURRENT horizon. Unlike backorder above,
				// this SUM is reset to 0 at the start of each new review horizon.
				std::vector<double> cumulative_backorder;

				int64_t LastRationingAction;
				int64_t ChangeInAction;
				int64_t Period;  // Current period t (absolute time index; also used for time-varying demand)

				std::vector<bool> AllowedActions;

				DynaPlex::VarGroup ToVarGroup() const;
			};

			using Event = std::vector<int64_t>; // flattened demand [k * |I| + i]

			std::vector<double> ReturnUsefulStatistics(const State&) const;
			void ResetHiddenStateVariables(State& state, DynaPlex::RNG&) const;
			int64_t GetH(const State&) const;
			double GetTimeVaryingDemandRate(int64_t customerIdx, int64_t itemIdx, int64_t period) const;

			// Remainder of the DynaPlex API.
			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&, const Event&) const;
			Event GetEvent(DynaPlex::RNG&) const;
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			bool IsAllowedAction(const State&, int64_t action) const;
			State GetInitialState() const;
			State GetState(const VarGroup&) const;
			explicit MDP(const DynaPlex::VarGroup&);
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
		};
	}
}
