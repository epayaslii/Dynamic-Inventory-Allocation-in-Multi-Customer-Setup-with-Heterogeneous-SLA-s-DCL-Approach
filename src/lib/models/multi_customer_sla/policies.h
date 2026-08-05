#pragma once
#include <cstdint>
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace multi_customer_sla
	{
		class MDP;

		// Static policy: always plays a single fixed composite action.
		class BaseStockPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			int64_t serviceLevelPolicy;
		public:
			BaseStockPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		// Rule-based dynamic policy: switches rationing rule based on how close
		// each customer's cumulative backorder is to its numeric backorder
		// allowance (beta_k) - cost-based greedy once exceeded, SLA-gap myopic
		// when near the limit, otherwise FCFS.
		class GreedyDynamicPolicy
		{
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			int64_t serviceLevelPolicy;
		public:
			GreedyDynamicPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};
	}
}
