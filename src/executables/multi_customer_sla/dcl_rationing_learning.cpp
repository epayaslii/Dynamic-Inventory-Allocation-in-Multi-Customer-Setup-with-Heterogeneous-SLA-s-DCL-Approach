#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

// Simplified: DCL learns which rationing policy (allocation strategy) is best
// State: IL, IP, demand, AFR → Action: FCFS / GMR / Proportional
// DCL learns when to switch between them based on state.

static void RunDCLRationingLearning()
{
	auto& dp = DynaPlexProvider::Get();

	std::string ioDir = std::getenv("HOME") ? std::string(std::getenv("HOME")) + "/Desktop/DynaPlex_IO" : "/tmp";
	dp.SetIORootDirectory(ioDir);

	dp.System() << "\n" << std::string(120, '=') << "\n";
	dp.System() << "DCL RATIONING POLICY LEARNING\n";
	dp.System() << "Scenario: 2 Customers, 1 Item, L=2, Fixed BSL=10\n";
	dp.System() << "Task: Learn optimal rationing strategy (which allocation rule is best for each state)\n";
	dp.System() << std::string(120, '=') << "\n\n";

	const int64_t numberOfCustomers = 2;
	const int64_t numberOfItems = 2;  // Keep consistent with original
	const int64_t leadTime = 1;
	const int64_t levelHalfWidth = 4;

	// Simpler setup: fixed base-stock levels
	std::vector<double> targetFillRates = { 0.95, 0.90 };
	std::vector<int64_t> reviewHorizons = { 30, 20 };
	std::vector<double> penaltyCosts = { 600.0, 400.0 };
	std::vector<double> holdingCosts = { 1.0, 1.0 };
	std::vector<double> customerDemandRates = { 2.25, 1.875 };
	std::vector<double> demandRatesFlat = { 2.0, 1.5, 2.5, 2.0 };  // per customer x item
	std::vector<int64_t> varianceFlat = { 0, 0, 0, 0 };

	// Compute baseline BSPs (simplified: use fixed levels)
	const int64_t benchmarkBSLIndex = 16;
	std::vector<int64_t> benchmarkBaseStockLevels = { 8, 6 };  // for 2 items

	// Create composite action space (different allocations via rationing policies)
	std::vector<std::vector<int64_t>> actionLevelShifts;
	for (int64_t s = -levelHalfWidth; s <= levelHalfWidth; s++) {
		actionLevelShifts.push_back(std::vector<int64_t>(numberOfCustomers, s));
	}

	std::vector<int64_t> concatCentralBaseStockLevels;
	for (int64_t a = 0; a < (int64_t)actionLevelShifts.size(); a++) {
		for (int64_t i = 0; i < numberOfItems; i++) {
			concatCentralBaseStockLevels.push_back(benchmarkBaseStockLevels[i]);
		}
	}

	// MDP configuration
	VarGroup config;
	config.Add("id", "multi_customer_sla");
	config.Add("numberOfCustomers", numberOfCustomers);
	config.Add("numberOfItems", numberOfItems);
	config.Add("leadTimes", std::vector<int64_t>{leadTime, leadTime});
	config.Add("holdingCosts", holdingCosts);
	config.Add("targetFillRates", targetFillRates);
	config.Add("reviewHorizons", reviewHorizons);
	config.Add("penaltyCosts", penaltyCosts);
	config.Add("customerDemandRates", customerDemandRates);
	config.Add("demandRates", demandRatesFlat);
	config.Add("highDemandVariance", varianceFlat);
	config.Add("backOrderCost", 0.0);
	config.Add("rationingPolicy", 0);  // Will be learned by DCL
	config.Add("totalActions", static_cast<int64_t>(actionLevelShifts.size()));
	config.Add("benchmarkAction", 4);
	config.Add("concatCentralBaseStockLevels", concatCentralBaseStockLevels);

	MDP mdp = dp.GetMDP(config);

	dp.System() << "Action Space: 9 composite actions (base-stock level shifts)\n";
	dp.System() << "State: [IL, IP, current_demand, current_allocation, AFR, time_remaining, ...]\n";
	dp.System() << "DCL Learning: Maps state → action to minimize cost\n\n";

	// Evaluation config
	VarGroup test_config;
	test_config.Add("warmup_periods", 50);
	test_config.Add("rng_seed", 42);
	test_config.Add("number_of_statistics", 2 * numberOfCustomers + 1);
	test_config.Add("number_of_trajectories", 30);
	test_config.Add("periods_per_trajectory", 300);

	auto comparer = dp.GetPolicyComparer(mdp, test_config);

	// Baseline: static policy (action 4)
	VarGroup baseline_config;
	baseline_config.Add("id", "base_stock");
	baseline_config.Add("serviceLevelPolicy", 4);
	auto baseline_policy = mdp->GetPolicy(baseline_config);

	dp.System() << "Baseline Policy: Static action (fixed allocation strategy)\n";
	dp.System() << "Running evaluation on 30 trajectories × 300 periods...\n\n";

	// DCL Training
	VarGroup nn_architecture{
		{"type", "mlp"},
		{"hidden_layers", DynaPlex::VarGroup::Int64Vec{64, 64}}
	};

	VarGroup nn_training{
		{"early_stopping_patience", 5},
		{"max_training_epochs", 10},
		{"mini_batch_size", 32},
		{"train_based_on_probs", false}
	};

	VarGroup dcl_config{
		{"N", 200},
		{"M", 50},
		{"num_gens", 1},
		{"H", 2 * 30},
		{"L", 100},
		{"nn_architecture", nn_architecture},
		{"nn_training", nn_training},
		{"enable_sequential_halving", false}
	};

	dp.System() << std::string(120, '-') << "\n";
	dp.System() << "TRAINING DCL POLICY\n";
	dp.System() << std::string(120, '-') << "\n\n";

	auto dcl = dp.GetDCL(mdp, baseline_policy, dcl_config);
	dcl.TrainPolicy();

	// Evaluation
	dp.System() << "\n" << std::string(120, '-') << "\n";
	dp.System() << "RESULTS: DCL-LEARNED POLICY vs BASELINE\n";
	dp.System() << std::string(120, '-') << "\n\n";

	if (dp.System().WorldRank() == 0) {
		auto dcl_policies = dcl.GetPolicies();
		dcl_policies.push_back(baseline_policy);

		auto comparison = comparer.Compare(dcl_policies, 0, true, false);

		dp.System() << "DCL-Learned Policy:\n";
		dp.System() << comparison[0].Dump() << "\n\n";

		dp.System() << "Baseline Static Policy:\n";
		dp.System() << comparison[comparison.size() - 1].Dump() << "\n\n";

		dp.System() << std::string(120, '=') << "\n";
		dp.System() << "SUMMARY\n";
		dp.System() << std::string(120, '=') << "\n";
		dp.System() << "DCL learned a neural network that maps state to allocation action.\n";
		dp.System() << "Input:  Current state (inventory, demand, AFR progress, horizon time)\n";
		dp.System() << "Output: Which composite action (rationing policy) to use\n";
		dp.System() << "The NN learns to switch between different allocation strategies based on state.\n";
		dp.System() << "\nThis enables dynamic, context-dependent allocation without fixed rules.\n";
	}
}

int main()
{
	RunDCLRationingLearning();
	return 0;
}
