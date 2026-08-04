#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
namespace DynaPlex::Models {
	namespace multi_customer_sla
	{
		BaseStockPolicy::BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.GetOrDefault("serviceLevelPolicy", serviceLevelPolicy, mdp->benchmarkRationingAction);
		}

		int64_t BaseStockPolicy::GetAction(const MDP::State& state) const
		{
			// Return the benchmark rationing rule (static allocation policy)
			return serviceLevelPolicy;
		}

		GreedyDynamicPolicy::GreedyDynamicPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			config.GetOrDefault("serviceLevelPolicy", serviceLevelPolicy, mdp->benchmarkRationingAction);
		}

		int64_t GreedyDynamicPolicy::GetAction(const MDP::State& state) const
		{
			// === DYNAMIC RATIONING RULE SELECTION ===
			// Ordering is STATIC (fixed base-stock level).
			// This policy decides which rationing rule to use for allocation:
			//   Action 0: FCFS
			//   Action 1: Backorder-gap myopic (prioritize customer closest to exceeding allowance)
			//   Action 2: Proportional (allocate by demand proportion)
			//   Action 3: Cost-based greedy (assign units to customer with highest SLA-penalty risk)

			// Check backorder status for all customers
			bool anyNearLimit = false;
			bool anyExceedsLimit = false;

			for (int64_t k = 0; k < mdp->numberOfCustomers; k++)
			{
				int64_t exceedAmount = static_cast<int64_t>(std::max(0.0, state.cumulative_backorder[k] - mdp->backorderAllowances[k]));
				if (exceedAmount > 0)
					anyExceedsLimit = true;
				else if (state.cumulative_backorder[k] > 0.8 * mdp->backorderAllowances[k])
					anyNearLimit = true;
			}

			// Choose rationing rule based on backorder status
			int64_t action = serviceLevelPolicy;  // default to benchmark

			if (anyExceedsLimit) {
				// At least one customer exceeds allowance: use cost-based greedy (action 3)
				// to minimize future penalty cost
				action = 3;
			}
			else if (anyNearLimit) {
				// At least one customer is near limit: use backorder-gap myopic (action 1)
				// to prioritize customer closest to exceeding allowance
				action = 1;
			}
			else {
				// All customers are safely below limits: use FCFS (action 0) for simplicity
				action = 0;
			}

			return action;
		}
	}
}
