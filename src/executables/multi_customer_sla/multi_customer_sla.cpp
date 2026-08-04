#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/discretedist.h"

using namespace DynaPlex;

// ---------------------------------------------------------------------------
// Multi-customer, multi-item inventory management under heterogeneous SLAs.
//
// A single supplier serves K customers from one shared central warehouse that
// stocks |I| items. Each customer has its OWN aggregate-fill-rate (AFR) target,
// review-horizon length, penalty cost and demand rates. There is no lateral
// transshipment between customers. When total demand exceeds central stock for
// an item, a rationing rule allocates the scarce units across customers.
//
// The MDP uses composite actions: a central base-stock level per item is selected
// to drive order-up-to behavior. Upon demand realization, available central stock
// is allocated to customers using the rationing rule. A Sequential Halving
// pre-processing step can generate a menu of candidate base-stock policies.
//
// This driver builds a small instance, prepares composite actions using the
// Greedy Heuristic (GH), builds the MDP and runs a SHORT Deep Controlled
// Learning (DCL) test - just enough to confirm the training/evaluation pipeline
// works end-to-end (a full training run would take much longer).
// ---------------------------------------------------------------------------

namespace {

	// Greedy Heuristic for a single customer: returns the base-stock sequence
	// {S^0, S^1, ...} (each a vector over items) together with the per-step
	// steady-state AFR. holdingCosts and leadTimes are shared (central);
	// demandRates are the customer's per-item demand rates.
	struct GHResult {
		std::vector<std::vector<int64_t>> baseStockLevels; // [step][item]
		std::vector<double> fillRates;                     // [step]
	};

	std::vector<double> CalculateItemStatistics(
		const DiscreteDist& demandDist,
		const DiscreteDist& demandOverLeadtimeDist,
		int64_t stock_level)
	{
		// {expected fill-rate gain, ..., expected on-hand change} - we only need
		// indices [1] (marginal fill rate) and [2] (marginal on-hand) here, plus
		// [0] (cumulative fill-rate contribution).
		std::vector<double> statistics(4, 0.0);
		for (int64_t j = 0; j <= stock_level; j++) {
			const double leadTimeDemandProb = demandOverLeadtimeDist.ProbabilityAt(j);
			double probSums = 0.0;
			for (int64_t k = 0; k <= stock_level - j; k++) {
				const double prob = demandDist.ProbabilityAt(k);
				probSums += prob;
				statistics[0] += leadTimeDemandProb * prob * k;
				statistics[3] += leadTimeDemandProb * prob * (stock_level - j - k);
			}
			const double factor = leadTimeDemandProb * (1.0 - probSums);
			statistics[0] += factor * (stock_level - j);
			statistics[1] += factor;
			statistics[2] += leadTimeDemandProb * probSums;
		}
		return statistics;
	}

	GHResult RunGreedyHeuristic(
		int64_t numberOfItems,
		const std::vector<double>& demandRates,
		const std::vector<int64_t>& highVariance,
		const std::vector<int64_t>& leadTimes,
		const std::vector<double>& holdingCosts,
		double totalDemandRate,
		int64_t max_iter)
	{
		std::vector<DiscreteDist> demandDist;
		std::vector<DiscreteDist> demandOverLeadtime;
		demandDist.reserve(numberOfItems);
		demandOverLeadtime.reserve(numberOfItems);
		for (int64_t i = 0; i < numberOfItems; i++) {
			if (highVariance[i] == 1)
				demandDist.push_back(DiscreteDist::GetGeometricDist(demandRates[i]));
			else
				demandDist.push_back(DiscreteDist::GetPoissonDist(demandRates[i]));
			if (leadTimes[i] > 0) {
				auto d = DiscreteDist::GetZeroDist();
				for (int64_t l = 0; l < leadTimes[i]; l++)
					d = d.Add(demandDist[i]);
				demandOverLeadtime.push_back(d);
			}
			else {
				demandOverLeadtime.push_back(DiscreteDist::GetZeroDist());
			}
		}

		GHResult result;
		std::vector<int64_t> stockLevels(numberOfItems, 0);
		double aggFillRate = 0.0;
		std::vector<double> changeFR(numberOfItems, 0.0);
		std::vector<double> changeH(numberOfItems, 0.0);
		std::vector<double> ratio(numberOfItems, 0.0);
		for (int64_t i = 0; i < numberOfItems; i++) {
			const auto stats = CalculateItemStatistics(demandDist[i], demandOverLeadtime[i], stockLevels[i]);
			aggFillRate += stats[0];
			changeFR[i] = stats[1];
			changeH[i] = stats[2] * holdingCosts[i];
			ratio[i] = changeFR[i] / changeH[i];
		}
		aggFillRate /= totalDemandRate;

		auto bestSKU = [&]() {
			int64_t best = 0;
			double limit = -std::numeric_limits<double>::infinity();
			for (int64_t i = 0; i < numberOfItems; i++)
				if (ratio[i] > limit) { best = i; limit = ratio[i]; }
			return best;
			};

		int64_t bestRatioSKU = bestSKU();
		for (int64_t k = 0; k < max_iter; k++) {
			result.baseStockLevels.push_back(stockLevels);
			result.fillRates.push_back(aggFillRate);

			stockLevels[bestRatioSKU]++;
			const auto stats = CalculateItemStatistics(demandDist[bestRatioSKU], demandOverLeadtime[bestRatioSKU], stockLevels[bestRatioSKU]);
			aggFillRate += changeFR[bestRatioSKU] / totalDemandRate;
			changeFR[bestRatioSKU] = stats[1];
			changeH[bestRatioSKU] = stats[2] * holdingCosts[bestRatioSKU];
			ratio[bestRatioSKU] = changeFR[bestRatioSKU] / changeH[bestRatioSKU];
			bestRatioSKU = bestSKU();
		}
		return result;
	}

	// First GH step whose steady-state AFR reaches the target.
	int64_t BenchmarkIndex(const std::vector<double>& fillRates, double target)
	{
		for (int64_t i = 0; i < (int64_t)fillRates.size(); i++)
			if (fillRates[i] >= target)
				return i;
		return (int64_t)fillRates.size() - 1;
	}
}

// rationingPolicy : 0 = FCFS, 1 = GMR (SLA-gap), 2 = proportional.
// independentBSP  : false = one shared composite action for all companies;
//                   true  = each customer picks its own base-stock policy.
// runDCL          : run the short DCL training test at the end.
static void RunMultiCustomerTest(int64_t rationingPolicy, bool independentBSP, bool runDCL)
{
	auto& dp = DynaPlexProvider::Get();

	// --- Instance definition: two heterogeneous customers, two shared SKUs. --
	// This is the simplest prototype (2 customers x 2 items). Scale up by
	// editing the vectors below.
	const int64_t numberOfCustomers = 2;
	const int64_t numberOfItems = 2;

	// Per-item (central) data: shared lead time and holding cost per SKU.
	std::vector<int64_t> leadTimes = { 1, 1 };
	std::vector<double> holdingCosts = { 1.0, 1.0 };

	// Per-customer SLA parameters: AFR target, review horizon, shortfall penalty.
	std::vector<double> targetFillRates = { 0.95, 0.90 };
	std::vector<int64_t> reviewHorizons = { 30, 20 };
	std::vector<double> penaltyCosts = { 600.0, 400.0 };

	// Per (customer,item) mean demand rates and variance flags.
	// (variance flag 0 = Poisson, 1 = geometric / high variance.)
	std::vector<std::vector<double>> demandPerCustomer = {
		{ 3.0, 2.0 },   // customer A
		{ 2.0, 3.0 }    // customer B
	};
	std::vector<std::vector<int64_t>> variancePerCustomer = {
		{ 0, 0 },       // customer A: both items Poisson
		{ 0, 0 }        // customer B: both items Poisson
	};

	std::vector<double> demandRatesFlat;
	std::vector<int64_t> varianceFlat;
	std::vector<double> customerDemandRates(numberOfCustomers, 0.0);
	for (int64_t k = 0; k < numberOfCustomers; k++) {
		for (int64_t i = 0; i < numberOfItems; i++) {
			demandRatesFlat.push_back(demandPerCustomer[k][i]);
			varianceFlat.push_back(variancePerCustomer[k][i]);
			customerDemandRates[k] += demandPerCustomer[k][i];
		}
	}

	// --- Composite actions = a finite set of base-stock policies (BSPs). -----
	// Each customer k has a GH-generated SEQUENCE of base-stock policies ordered
	// by steady-state AFR; customerBenchmark[k] is the policy that meets its SLA
	// target (current practice). A composite action selects a central base-stock
	// level per item to drive order-up-to behavior. Central orders are summed
	// from the per-customer contributions.
	//
	//   independentBSP = false : ONE shared level shift for all customers
	//                            (flat action set of size 2h+1; easiest).
	//   independentBSP = true  : EACH customer gets its own shift; the action set
	//                            is the cartesian product (size (2h+1)^K).
	const int64_t levelHalfWidth = independentBSP ? 2 : 4;

	std::vector<int64_t> customerBenchmark(numberOfCustomers);
	std::vector<std::vector<std::vector<int64_t>>> customerSequences(numberOfCustomers);

	for (int64_t k = 0; k < numberOfCustomers; k++) {
		const int64_t maxIter = 4000;
		auto gh = RunGreedyHeuristic(numberOfItems, demandPerCustomer[k], variancePerCustomer[k],
			leadTimes, holdingCosts, customerDemandRates[k], maxIter);
		customerBenchmark[k] = BenchmarkIndex(gh.fillRates, targetFillRates[k]);
		customerSequences[k] = gh.baseStockLevels;
		dp.System() << "Customer " << k << ": benchmark BSP index " << customerBenchmark[k]
			<< " (steady-state AFR " << gh.fillRates[customerBenchmark[k]] << ")" << std::endl;
	}

	// Enumerate the per-customer level shifts that define the composite actions.
	// Each entry is a vector of K shifts in [-h, h]; the benchmark action is the
	// all-zero shift (every customer at its GH benchmark BSP).
	std::vector<std::vector<int64_t>> actionLevelShifts;
	if (independentBSP) {
		std::vector<int64_t> combo(numberOfCustomers, -levelHalfWidth);
		while (true) {
			actionLevelShifts.push_back(combo);
			// Odometer increment over customers.
			int64_t pos = numberOfCustomers - 1;
			while (pos >= 0) {
				combo[pos]++;
				if (combo[pos] <= levelHalfWidth) break;
				combo[pos] = -levelHalfWidth;
				pos--;
			}
			if (pos < 0) break;
		}
	}
	else {
		for (int64_t s = -levelHalfWidth; s <= levelHalfWidth; s++)
			actionLevelShifts.push_back(std::vector<int64_t>(numberOfCustomers, s));
	}

	const int64_t totalActions = (int64_t)actionLevelShifts.size();
	int64_t benchmarkAction = 0;
	for (int64_t a = 0; a < totalActions; a++) {
		bool allZero = true;
		for (int64_t k = 0; k < numberOfCustomers; k++)
			if (actionLevelShifts[a][k] != 0) { allZero = false; break; }
		if (allZero) { benchmarkAction = a; break; }
	}
	dp.System() << "Composite actions (base-stock policies): " << totalActions
		<< (independentBSP ? " (independent per-customer BSP)" : " (shared BSP level)")
		<< ", benchmark action index " << benchmarkAction << std::endl;

	// For each composite action, the central order-up-to level per item is the
	// sum over customers of that customer's selected BSP (shared central pool).
	std::vector<int64_t> concatCentralBaseStockLevels;
	concatCentralBaseStockLevels.reserve(totalActions * numberOfItems);
	for (int64_t a = 0; a < totalActions; a++) {
		std::vector<int64_t> central(numberOfItems, 0);
		for (int64_t k = 0; k < numberOfCustomers; k++) {
			int64_t idx = customerBenchmark[k] + actionLevelShifts[a][k];
			idx = std::max<int64_t>(0, std::min<int64_t>(idx, (int64_t)customerSequences[k].size() - 1));
			for (int64_t i = 0; i < numberOfItems; i++)
				central[i] += customerSequences[k][idx][i];
		}
		for (int64_t i = 0; i < numberOfItems; i++)
			concatCentralBaseStockLevels.push_back(central[i]);
	}

	// --- Assemble the MDP configuration. ------------------------------------
	DynaPlex::VarGroup config;
	config.Add("id", "multi_customer_sla");
	config.Add("numberOfCustomers", numberOfCustomers);
	config.Add("numberOfItems", numberOfItems);
	config.Add("leadTimes", leadTimes);
	config.Add("holdingCosts", holdingCosts);
	config.Add("targetFillRates", targetFillRates);
	config.Add("reviewHorizons", reviewHorizons);
	config.Add("penaltyCosts", penaltyCosts);
	config.Add("customerDemandRates", customerDemandRates);
	config.Add("demandRates", demandRatesFlat);
	config.Add("highDemandVariance", varianceFlat);
	config.Add("backOrderCost", 0.0);
	config.Add("rationingPolicy", rationingPolicy); // 0=FCFS, 1=GMR, 2=proportional
	config.Add("totalActions", totalActions);
	config.Add("benchmarkAction", benchmarkAction);
	config.Add("concatCentralBaseStockLevels", concatCentralBaseStockLevels);
	// Time-varying demand: enable sinusoidal seasonal variation
	config.Add("demandAmplitude", 0.2);  // ±20% variation
	config.Add("demandPeriod", static_cast<int64_t>(12));  // 12-period cycle

	// --- Evaluation settings (kept small for a quick test). -----------------
	DynaPlex::VarGroup test_config;
	test_config.Add("warmup_periods", 100);
	test_config.Add("rng_seed", 10061994);
	test_config.Add("number_of_statistics", 2 * numberOfCustomers + 1);
	test_config.Add("number_of_trajectories", 100);
	test_config.Add("periods_per_trajectory", 1000);

	DynaPlex::MDP mdp = dp.GetMDP(config);

	DynaPlex::VarGroup static_policy_config;
	static_policy_config.Add("id", "base_stock");
	static_policy_config.Add("serviceLevelPolicy", benchmarkAction);
	auto static_policy = mdp->GetPolicy(static_policy_config);

	DynaPlex::VarGroup greedy_policy_config;
	greedy_policy_config.Add("id", "greedy_dynamic");
	auto greedy_policy = mdp->GetPolicy(greedy_policy_config);

	// Compare the static benchmark with the rule-based dynamic policy.
	const char* rationingName = rationingPolicy == 0 ? "FCFS"
		: (rationingPolicy == 1 ? "GMR (SLA-gap)" : "Proportional");
	dp.System() << "----- rationing = " << rationingName
		<< " : static vs rule-based dynamic -----" << std::endl;
	dp.System() << "(mean = avg cost; mean_stat_1 = AFR customer A; mean_stat_2 = AFR customer B)" << std::endl;
	auto comparer = dp.GetPolicyComparer(mdp, test_config);
	{
		std::vector<DynaPlex::Policy> policies = { static_policy, greedy_policy };
		auto comparison = comparer.Compare(policies, 0, true, false);
		for (auto& results : comparison)
			dp.System() << results.Dump() << std::endl;
	}

	if (!runDCL)
		return;

	// --- SHORT DCL training test. -------------------------------------------
	// Deliberately tiny so this finishes quickly; this only verifies that the
	// DCL pipeline runs on the multi-customer MDP, not that it converges.
	DynaPlex::VarGroup nn_architecture{
		{"type","mlp"},
		{"hidden_layers",DynaPlex::VarGroup::Int64Vec{64,64}}
	};
	DynaPlex::VarGroup nn_training{
		{"early_stopping_patience",5},
		{"max_training_epochs", 5},
		{"mini_batch_size", 64},
		{"train_based_on_probs", false}
	};
	DynaPlex::VarGroup dcl_config{
		{"N", 200},
		{"M", 50},
		{"num_gens", 1},
		{"H", 2 * 30},
		{"L", 100},
		{"nn_architecture", nn_architecture},
		{"nn_training", nn_training},
		{"enable_sequential_halving", true}
	};

	dp.System() << "----- short DCL training test -----" << std::endl;
	auto dcl = dp.GetDCL(mdp, static_policy, dcl_config);
	dcl.TrainPolicy();

	if (dp.System().WorldRank() == 0) {
		auto dcl_policies = dcl.GetPolicies();
		dcl_policies.push_back(greedy_policy);
		auto comparison = comparer.Compare(dcl_policies, 0, true, false);
		for (auto& results : comparison)
			dp.System() << results.Dump() << std::endl;
		dp.System() << "DCL pipeline ran successfully on the multi-customer MDP." << std::endl;
	}
}

int main()
{
	auto& dp = DynaPlexProvider::Get();

	// (a) Compare the three rationing rules (fast: no DCL training). Uses the
	//     shared-BSP composite actions and the static benchmark + rule-based
	//     dynamic policy, so differences are driven by the rationing rule.
	dp.System() << "========== RATIONING-RULE COMPARISON ==========" << std::endl;
	for (int64_t rationingPolicy = 0; rationingPolicy <= 2; rationingPolicy++) {
		RunMultiCustomerTest(rationingPolicy, /*independentBSP=*/false, /*runDCL=*/false);
		dp.System() << std::endl;
	}

	// (b) One short DCL training test (GMR rationing) to confirm the learning
	//     pipeline still runs end-to-end on the multi-customer MDP.
	dp.System() << "========== SHORT DCL TEST (GMR) ==========" << std::endl;
	RunMultiCustomerTest(/*rationingPolicy=*/1, /*independentBSP=*/false, /*runDCL=*/true);
	return 0;
}
