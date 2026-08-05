#include <gtest/gtest.h>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/trajectory.h"

using namespace DynaPlex;

namespace DynaPlex::Tests {

	// Build a minimal, hand-crafted multi_customer_sla instance (no Sequential
	// Halving preprocessing needed) so the test is fully self-contained.
	static DynaPlex::VarGroup MinimalConfig()
	{
		DynaPlex::VarGroup config;
		config.Add("id", "multi_customer_sla");
		config.Add("numberOfCustomers", static_cast<int64_t>(2));
		config.Add("numberOfItems", static_cast<int64_t>(2));
		config.Add("leadTimes", DynaPlex::VarGroup::Int64Vec{ 1, 1 });
		config.Add("holdingCosts", std::vector<double>{ 1.0, 1.0 });
		// Numeric backorder allowances (beta_k), per the paper's SLA formulation.
		config.Add("backorderAllowances", DynaPlex::VarGroup::Int64Vec{ 6, 6 });
		config.Add("reviewHorizons", DynaPlex::VarGroup::Int64Vec{ 6, 5 });
		config.Add("penaltyCosts", std::vector<double>{ 600.0, 400.0 });
		config.Add("customerDemandRates", std::vector<double>{ 3.0, 3.0 });
		// demand per (customer,item) flattened [k*I + i]
		config.Add("demandRates", std::vector<double>{ 2.0, 1.0, 1.0, 2.0 });
		config.Add("highDemandVariance", DynaPlex::VarGroup::Int64Vec{ 0, 0, 0, 0 });
		// Static base-stock level (same for all allocation actions)
		config.Add("baseStockLevel", DynaPlex::VarGroup::Int64Vec{ 7, 7 });
		// Dynamic allocation: 4 rationing rules to choose from (FCFS, SLA-gap, Proportional, Cost-greedy)
		config.Add("totalRationingActions", static_cast<int64_t>(4));
		config.Add("benchmarkRationingAction", static_cast<int64_t>(1));
		return config;
	}

	// Smoke test: the MDP can be built and simulated for a number of periods
	// under both the static and the rule-based dynamic policy.
	TEST(multi_customer_sla, mdp_basics) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::MDP mdp;
		ASSERT_NO_THROW(mdp = dp.GetMDP(MinimalConfig()));

		for (const std::string& policy_id : { std::string("base_stock"), std::string("greedy_dynamic") }) {
			DynaPlex::Policy policy;
			ASSERT_NO_THROW(policy = mdp->GetPolicy(policy_id));

			Trajectory trajectory{};
			ASSERT_NO_THROW(mdp->InitiateState({ &trajectory, 1 }));
			ASSERT_NO_THROW(trajectory.RNGProvider.SeedEventStreams(true, 123));

			int64_t max_period_count = 50;
			bool finalreached = false;
			while (trajectory.PeriodCount < max_period_count && !finalreached) {
				auto& cat = trajectory.Category;
				if (cat.IsAwaitEvent())
					ASSERT_NO_THROW(mdp->IncorporateEvent({ &trajectory, 1 }));
				else if (cat.IsAwaitAction()) {
					ASSERT_NO_THROW(policy->SetAction({ &trajectory, 1 }));
					ASSERT_NO_THROW(mdp->IncorporateAction({ &trajectory, 1 }));
				}
				else if (cat.IsFinal())
					finalreached = true;
			}
		}
	}

	// Smoke test for Action 3 (cost-based greedy rationing): verify it runs without error.
	TEST(multi_customer_sla, cost_based_greedy_smoke) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::MDP mdp;
		ASSERT_NO_THROW(mdp = dp.GetMDP(MinimalConfig()));

		// Force Action 3 (cost-based greedy) using base_stock policy with serviceLevelPolicy=3
		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		policy_config.Add("serviceLevelPolicy", static_cast<int64_t>(3));  // Action 3
		DynaPlex::Policy policy;
		ASSERT_NO_THROW(policy = mdp->GetPolicy(policy_config));

		Trajectory trajectory{};
		ASSERT_NO_THROW(mdp->InitiateState({ &trajectory, 1 }));
		ASSERT_NO_THROW(trajectory.RNGProvider.SeedEventStreams(true, 123));

		int64_t max_period_count = 50;
		bool finalreached = false;
		while (trajectory.PeriodCount < max_period_count && !finalreached) {
			auto& cat = trajectory.Category;
			if (cat.IsAwaitEvent())
				ASSERT_NO_THROW(mdp->IncorporateEvent({ &trajectory, 1 }));
			else if (cat.IsAwaitAction()) {
				ASSERT_NO_THROW(policy->SetAction({ &trajectory, 1 }));
				ASSERT_NO_THROW(mdp->IncorporateAction({ &trajectory, 1 }));
			}
			else if (cat.IsFinal())
				finalreached = true;
		}
	}

#if DP_TORCH_AVAILABLE
	// DCL smoke test: verifies the Deep Controlled Learning pipeline runs on the
	// multi-customer MDP. Deliberately tiny (this is NOT a convergence test).
	TEST(multi_customer_sla, dcl_smoke) {
		auto& dp = DynaPlexProvider::Get();
		DynaPlex::MDP mdp = dp.GetMDP(MinimalConfig());

		DynaPlex::VarGroup policy_config;
		policy_config.Add("id", "base_stock");
		policy_config.Add("serviceLevelPolicy", static_cast<int64_t>(1));
		auto policy = mdp->GetPolicy(policy_config);

		DynaPlex::VarGroup nn_architecture{ {"type","mlp"}, {"hidden_layers",DynaPlex::VarGroup::Int64Vec{32}} };
		DynaPlex::VarGroup nn_training{
			{"early_stopping_patience", 2},
			{"max_training_epochs", 2},
			{"mini_batch_size", 32},
			{"train_based_on_probs", false} };
		DynaPlex::VarGroup dcl_config{
			{"N", 256}, {"M", 10}, {"num_gens", 1}, {"H", 12}, {"L", 16},
			{"nn_architecture", nn_architecture},
			{"nn_training", nn_training},
			{"enable_sequential_halving", true} };

		ASSERT_NO_THROW({
			auto dcl = dp.GetDCL(mdp, policy, dcl_config);
			dcl.TrainPolicy();
			auto policies = dcl.GetPolicies();
			EXPECT_GT(policies.size(), 0u);
			});
	}
#endif
}
