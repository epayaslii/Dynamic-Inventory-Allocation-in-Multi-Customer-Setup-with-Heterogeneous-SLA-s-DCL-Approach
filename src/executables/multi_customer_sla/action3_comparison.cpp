#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

static void RunAction3Comparison()
{
    auto& dp = DynaPlexProvider::Get();

    std::string ioDir = std::getenv("HOME") ? std::string(std::getenv("HOME")) + "/Desktop/DynaPlex_IO" : "/tmp";
    dp.SetIORootDirectory(ioDir);

    dp.System() << "\n" << std::string(120, '=') << "\n";
    dp.System() << "ACTION 3 DEMONSTRATION: Cost-Based Greedy vs SLA-Gap Myopic\n";
    dp.System() << "Configuration: 2 Customers, 2 Items, Lead Time 2\n";
    dp.System() << std::string(120, '=') << "\n\n";

    // ========== Setup MDP ==========
    const int64_t numberOfCustomers = 2;
    const int64_t numberOfItems = 2;
    const int64_t leadTime = 2;
    const int64_t reviewHorizon = 10;

    dp.System() << "MDP Configuration:\n";
    dp.System() << "  Customers: " << numberOfCustomers << "\n";
    dp.System() << "  Items: " << numberOfItems << "\n";
    dp.System() << "  Lead Time: " << leadTime << "\n";
    dp.System() << "  Review Horizon: " << reviewHorizon << "\n";
    dp.System() << "  Base-Stock Level: [10, 10]\n";
    dp.System() << "  Customer 0: targetFR=0.95, penalty=600.0\n";
    dp.System() << "  Customer 1: targetFR=0.90, penalty=400.0\n\n";

    // Configure MDP
    VarGroup config;
    config.Add("id", "multi_customer_sla");
    config.Add("numberOfCustomers", numberOfCustomers);
    config.Add("numberOfItems", numberOfItems);
    config.Add("leadTimes", std::vector<int64_t>{leadTime, leadTime});
    config.Add("holdingCosts", std::vector<double>{1.0, 1.0});
    config.Add("targetFillRates", std::vector<double>{0.95, 0.90});
    config.Add("reviewHorizons", std::vector<int64_t>{reviewHorizon, reviewHorizon});
    config.Add("penaltyCosts", std::vector<double>{600.0, 400.0});
    config.Add("customerDemandRates", std::vector<double>{3.0, 3.0});
    config.Add("demandRates", std::vector<double>{2.0, 1.0, 1.0, 2.0});
    config.Add("highDemandVariance", std::vector<int64_t>{0, 0, 0, 0});
    config.Add("backOrderCost", 0.5);
    config.Add("baseStockLevel", std::vector<int64_t>{10, 10});
    config.Add("totalRationingActions", static_cast<int64_t>(4));
    config.Add("benchmarkRationingAction", static_cast<int64_t>(1));

    MDP mdp = dp.GetMDP(config);

    dp.System() << "MDP created successfully.\n";
    dp.System() << "Action Space:\n";
    dp.System() << "  Action 0: FCFS (First-Come-First-Served)\n";
    dp.System() << "  Action 1: SLA-Gap Myopic (prioritize furthest from target)\n";
    dp.System() << "  Action 2: Proportional (by demand share)\n";
    dp.System() << "  Action 3: Cost-Based Greedy (exploratory, outside paper scope)\n\n";

    // ========== Setup Evaluation Configuration ==========
    VarGroup test_config;
    test_config.Add("warmup_periods", 50);
    test_config.Add("rng_seed", 42);
    test_config.Add("number_of_statistics", static_cast<int64_t>(7));
    test_config.Add("number_of_trajectories", 50);   // 50 trajectories for demo
    test_config.Add("periods_per_trajectory", 200);

    auto comparer = dp.GetPolicyComparer(mdp, test_config);

    // ========== ACTION 1: SLA-Gap Myopic BENCHMARK ==========
    dp.System() << std::string(120, '-') << "\n";
    dp.System() << "BENCHMARK: SLA-Gap Myopic (Action 1)\n";
    dp.System() << std::string(120, '-') << "\n\n";

    VarGroup action1_config;
    action1_config.Add("id", "base_stock");
    action1_config.Add("serviceLevelPolicy", static_cast<int64_t>(1));
    auto action1_policy = mdp->GetPolicy(action1_config);

    dp.System() << "Running 50 trajectories × 200 periods...\n";
    std::vector<DynaPlex::Policy> action1_policies = {action1_policy};
    auto action1_comparison = comparer.Compare(action1_policies, 0, true, false);

    dp.System() << "\nAction 1 (SLA-Gap) Results:\n";
    dp.System() << action1_comparison[0].Dump() << "\n\n";

    // ========== ACTION 3: Cost-Based Greedy ==========
    dp.System() << std::string(120, '-') << "\n";
    dp.System() << "NEW RULE: Cost-Based Greedy (Action 3)\n";
    dp.System() << std::string(120, '-') << "\n\n";

    VarGroup action3_config;
    action3_config.Add("id", "base_stock");
    action3_config.Add("serviceLevelPolicy", static_cast<int64_t>(3));
    auto action3_policy = mdp->GetPolicy(action3_config);

    dp.System() << "Running 50 trajectories × 200 periods...\n";
    std::vector<DynaPlex::Policy> action3_policies = {action3_policy};
    auto action3_comparison = comparer.Compare(action3_policies, 0, true, false);

    dp.System() << "\nAction 3 (Cost-Greedy) Results:\n";
    dp.System() << action3_comparison[0].Dump() << "\n\n";

    // ========== COMPARISON & INTERPRETATION ==========
    dp.System() << std::string(120, '=') << "\n";
    dp.System() << "COMPARISON ANALYSIS\n";
    dp.System() << std::string(120, '=') << "\n\n";

    double cost_a1 = 0.0, cost_a3 = 0.0;
    try {
        action1_comparison[0].Get("mean", cost_a1);
        action3_comparison[0].Get("mean", cost_a3);

        double cost_diff = cost_a1 - cost_a3;
        double pct_diff = (std::abs(cost_a1) > 0.001) ? (cost_diff / cost_a1) * 100.0 : 0.0;

        dp.System() << std::fixed << std::setprecision(4);
        dp.System() << "Average Cost Comparison:\n";
        dp.System() << "  Action 1 (SLA-Gap):     " << cost_a1 << "\n";
        dp.System() << "  Action 3 (Cost-Greedy): " << cost_a3 << "\n";
        dp.System() << "  Difference:            " << cost_diff << " (" << pct_diff << "%)\n\n";

        if (std::abs(cost_diff) < 0.01) {
            dp.System() << "INTERPRETATION:\n";
            dp.System() << "  With high base-stock levels (S=10), allocation rules converge to similar cost.\n";
            dp.System() << "  This is expected: shortfalls are rare, so allocation method matters less.\n";
            dp.System() << "  Action 3's advantage emerges under resource scarcity (lower S, higher demand variance).\n\n";
        } else if (cost_diff > 0) {
            dp.System() << "INTERPRETATION:\n";
            dp.System() << "  Action 3 (Cost-Greedy) OUTPERFORMS Action 1 (SLA-Gap) by " << pct_diff << "%.\n";
            dp.System() << "  Cost-based allocation better protects high-penalty customers at risk.\n\n";
        } else {
            dp.System() << "INTERPRETATION:\n";
            dp.System() << "  Action 1 (SLA-Gap) performs better in this scenario.\n";
            dp.System() << "  This suggests fill-rate-gap prioritization is effective here.\n\n";
        }
    }
    catch (...) {
        dp.System() << "Note: Raw results available:\n\n";
        dp.System() << "Action 1 (SLA-Gap):\n" << action1_comparison[0].Dump() << "\n\n";
        dp.System() << "Action 3 (Cost-Greedy):\n" << action3_comparison[0].Dump() << "\n\n";
    }

    dp.System() << std::string(120, '=') << "\n";
    dp.System() << "CONCLUSION\n";
    dp.System() << std::string(120, '=') << "\n";
    dp.System() << "Action 3 (Cost-Based Greedy) has been successfully implemented and tested.\n";
    dp.System() << "It is available as an exploratory rationing rule for comparison and learning.\n";
    dp.System() << "The rule is NOT part of the paper's core 3-rule scope (FCFS/SLA-Gap/Proportional).\n";
    dp.System() << std::string(120, '=') << "\n\n";
}

int main()
{
    try {
        RunAction3Comparison();
        std::cout << "\nDONE: Action 3 demonstration complete.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
