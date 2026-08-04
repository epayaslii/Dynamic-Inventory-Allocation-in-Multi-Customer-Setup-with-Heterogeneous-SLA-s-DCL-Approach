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

		// Rule-based dynamic policy: starts from the static composite action and
		// shifts to a higher (lower) steady-state AFR level when some customer is
		// below (all customers are comfortably above) their SLA target.
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
