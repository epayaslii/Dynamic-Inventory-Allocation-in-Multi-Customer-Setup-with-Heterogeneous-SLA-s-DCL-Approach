#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

// ============================================================================
// SIMPLIFIED MULTI-CUSTOMER, MULTI-SKU INVENTORY SYSTEM
//
// Educational version showing: STATE → ACTION → DEMAND → COST → NEXT STATE
//
// Setup:
//   - 2 companies (A, B) sharing a central warehouse
//   - 2 SKUs (item 0, item 1)
//   - No composite actions: each customer picks a "level" (0-4) directly
//   - Level 0 = lowest stock (cheap), Level 4 = highest stock (safe)
// ============================================================================

struct Item {
    int64_t sku_id;
    double holding_cost;
    int64_t lead_time;
};

struct Customer {
    int64_t id;
    string name;
    double demand_rate[2];  // demand rate per SKU
    double target_afr;
    int64_t review_horizon;
    double penalty_cost;

    // Current state
    int64_t total_demand_seen = 0;
    int64_t total_satisfied = 0;
    int64_t periods_in_horizon = 0;
};

struct State {
    int64_t on_hand[2];                  // on-hand stock per SKU
    int64_t in_pipeline[2];              // units in transit (lead time 1)
    double customer_afr[2];               // AFR so far for each customer
    int64_t customer_time_remaining[2];   // time left in review horizon
};

// ============================================================================
// GREEDY HEURISTIC: Find sequence of base-stock policies for a customer
// ============================================================================
struct GHResult {
    vector<vector<int64_t>> baseStockSequence;  // [step][sku]
    vector<double> fillRates;                    // [step]
};

GHResult RunGreedyHeuristic(
    const Customer& customer,
    const Item items[2],
    int64_t max_steps)
{
    GHResult result;
    vector<int64_t> stock(2, 0);
    double aggFillRate = 0.0;
    double totalDemand =  customer.demand_rate[0] +  customer.demand_rate[1];

    // Simple approximation: marginal fill rate gain and cost per SKU
    auto getMarginalsRatio = [&](int64_t sku) {
        // Higher variance → higher fill rate gain per unit
        // Higher cost → lower appeal
        double gain =  customer.demand_rate[sku] * 0.15;  // arbitrary: 15% fill rate gain per unit
        double cost = items[sku].holding_cost;
        return gain / cost;
    };

    for (int64_t step = 0; step < max_steps; step++) {
        result.baseStockSequence.push_back(stock);
        result.fillRates.push_back(aggFillRate);

        // Greedy step: add 1 unit to best SKU (highest gain/cost ratio)
        int64_t best_sku = 0;
        double best_ratio = getMarginalsRatio(0);
        for (int64_t i = 1; i < 2; i++) {
            double ratio = getMarginalsRatio(i);
            if (ratio > best_ratio) {
                best_sku = i;
                best_ratio = ratio;
            }
        }

        stock[best_sku]++;
        aggFillRate += 0.01;  // arbitrary: 1% AFR gain per step
        aggFillRate = min(aggFillRate, 0.99);
    }

    return result;
}

// ============================================================================
// MAIN SIMULATION
// ============================================================================
int main() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    cout << "║  SIMPLIFIED MULTI-CUSTOMER INVENTORY MDP (Educational Version)      ║\n";
    cout << "║  2 Customers × 2 SKUs | Level-based Actions (0=cheap, 4=safe)      ║\n";
    cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    cout << "\n";

    // ========== SETUP ==========
    Item items[2] = {
        {0, 1.0, 1},   // SKU 0: holding cost = 1.0, lead time = 1 period
        {1, 1.0, 1}    // SKU 1: holding cost = 1.0, lead time = 1 period
    };

    Customer customers[2] = {
        {0, "Customer A", {3.0, 2.0}, 0.95, 10, 600.0},  // demand=[3,2], AFR target=95%, horizon=10, penalty=600
        {1, "Customer B", {2.0, 3.0}, 0.90, 10, 400.0}   // demand=[2,3], AFR target=90%, horizon=10, penalty=400
    };

    cout << "━━ CUSTOMERS ━━\n";
    for (auto& c : companies) {
        cout << c.name << " (id=" << c.id << "):\n"
             << "  Demand rates: SKU0=" << c.demand_rate[0] << "/period, "
             << "SKU1=" << c.demand_rate[1] << "/period\n"
             << "  AFR target: " << (c.target_afr * 100.0) << "%\n"
             << "  Review horizon: " << c.review_horizon << " periods\n"
             << "  Shortfall penalty: " << c.penalty_cost << "\n\n";
    }

    cout << "━━ ITEMS ━━\n";
    for (auto& item : items) {
        cout << "SKU " << item.sku_id << ": holding_cost=" << item.holding_cost
             << ", lead_time=" << item.lead_time << " period(s)\n";
    }
    cout << "\n";

    // ========== GREEDY HEURISTIC PREPROCESSING ==========
    cout << "━━ GREEDY HEURISTIC (GH) - Preprocessing ━━\n";
    cout << "(Finding base-stock sequences for each customer)\n\n";

    GHResult gh_A = RunGreedyHeuristic(customers[0], items, 5);
    GHResult gh_B = RunGreedyHeuristic(customers[1], items, 5);

    cout << "Customer A base-stock sequence:\n";
    for (int64_t level = 0; level < (int64_t)gh_A.baseStockSequence.size(); level++) {
        cout << "  Level " << level << ": [" << gh_A.baseStockSequence[level][0]
             << ", " << gh_A.baseStockSequence[level][1] << "]"
             << " → AFR=" << fixed << setprecision(2) << (gh_A.fillRates[level] * 100.0) << "%\n";
    }
    cout << "\nCustomer B base-stock sequence:\n";
    for (int64_t level = 0; level < (int64_t)gh_B.baseStockSequence.size(); level++) {
        cout << "  Level " << level << ": [" << gh_B.baseStockSequence[level][0]
             << ", " << gh_B.baseStockSequence[level][1] << "]"
             << " → AFR=" << fixed << setprecision(2) << (gh_B.fillRates[level] * 100.0) << "%\n";
    }
    cout << "\n";

    // ========== INITIAL STATE ==========
    State state;
    state.on_hand[0] = 5;
    state.on_hand[1] = 5;
    state.in_pipeline[0] = 0;
    state.in_pipeline[1] = 0;
    state.customer_afr[0] = 0.0;
    state.customer_afr[1] = 0.0;
    state.customer_time_remaining[0] = customers[0].review_horizon;
    state.customer_time_remaining[1] = customers[1].review_horizon;

    cout << "━━ INITIAL STATE ━━\n";
    cout << "On-hand stock: [" << state.on_hand[0] << ", " << state.on_hand[1] << "]\n";
    cout << "In pipeline: [" << state.in_pipeline[0] << ", " << state.in_pipeline[1] << "]\n";
    cout << "\n";

    // ========== SIMULATION LOOP ==========
    const int64_t NUM_PERIODS = 12;
    double total_cost = 0.0;

    for (int64_t period = 0; period < NUM_PERIODS; period++) {
        cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
        cout << "║ PERIOD " << setw(2) << period << "                                                                ║\n";
        cout << "╚═══════════════════════════════════════════════════════════════════╝\n";

        // ─── OBSERVE STATE ───
        cout << "\n[STATE]\n";
        cout << "  On-hand stock: [" << state.on_hand[0] << ", " << state.on_hand[1] << "]\n";
        cout << "  In pipeline: [" << state.in_pipeline[0] << ", " << state.in_pipeline[1] << "]\n";
        cout << "  Inventory position: [" << (state.on_hand[0] + state.in_pipeline[0])
             << ", " << (state.on_hand[1] + state.in_pipeline[1]) << "]\n";
        for (int64_t k = 0; k < 2; k++) {
            cout << "  " << customers[k].name << " → " << state.customer_time_remaining[k]
                 << " periods left, AFR so far: " << fixed << setprecision(1)
                 << (state.customer_afr[k] * 100.0) << "%\n";
        }

        // ─── CHOOSE ACTION ───
        // Test: Customer A picks level 3, Customer B picks level 3 (middle ground)
        int64_t action_level_A = 3;
        int64_t action_level_B = 3;

        cout << "\n[ACTION]\n";
        cout << "  Customer A chooses level " << action_level_A
             << " → target stock = [" << gh_A.baseStockSequence[action_level_A][0]
             << ", " << gh_A.baseStockSequence[action_level_A][1] << "]\n";
        cout << "  Customer B chooses level " << action_level_B
             << " → target stock = [" << gh_B.baseStockSequence[action_level_B][0]
             << ", " << gh_B.baseStockSequence[action_level_B][1] << "]\n";

        // Calculate central order-up-to (sum of customer targets)
        int64_t central_target[2];
        central_target[0] = gh_A.baseStockSequence[action_level_A][0] +
                           gh_B.baseStockSequence[action_level_B][0];
        central_target[1] = gh_A.baseStockSequence[action_level_A][1] +
                           gh_B.baseStockSequence[action_level_B][1];

        cout << "  Central order-up-to: [" << central_target[0] << ", " << central_target[1] << "]\n";

        // Place orders (base-stock rule: order = target - inventory_position)
        int64_t to_order[2];
        int64_t inv_position[2];
        inv_position[0] = state.on_hand[0] + state.in_pipeline[0];
        inv_position[1] = state.on_hand[1] + state.in_pipeline[1];

        to_order[0] = max(0LL, central_target[0] - inv_position[0]);
        to_order[1] = max(0LL, central_target[1] - inv_position[1]);

        cout << "  Orders to place: [" << to_order[0] << ", " << to_order[1] << "]\n";

        // Orders enter pipeline (will arrive next period due to lead time = 1)
        state.in_pipeline[0] += to_order[0];
        state.in_pipeline[1] += to_order[1];

        // ─── LEAD TIME PASSES ───
        // Orders from PREVIOUS period arrive
        state.on_hand[0] += state.in_pipeline[0];
        state.on_hand[1] += state.in_pipeline[1];
        state.in_pipeline[0] = 0;
        state.in_pipeline[1] = 0;

        // (Note: In a real system with lead_time > 1, you'd have a queue)

        // ─── RANDOM DEMAND EVENT ───
        // Simplification: use expected demand (Poisson mean)
        double demand_A[2] = {customers[0].demand_rate[0], customers[0].demand_rate[1]};
        double demand_B[2] = {customers[1].demand_rate[0], customers[1].demand_rate[1]};

        cout << "\n[DEMAND EVENT]\n";
        cout << "  Customer A demand: [" << fixed << setprecision(1) << demand_A[0]
             << ", " << demand_A[1] << "]\n";
        cout << "  Customer B demand: [" << fixed << setprecision(1) << demand_B[0]
             << ", " << demand_B[1] << "]\n";

        // Total demand
        double total_demand[2];
        total_demand[0] = demand_A[0] + demand_B[0];
        total_demand[1] = demand_A[1] + demand_B[1];

        cout << "  Total demand: [" << fixed << setprecision(1) << total_demand[0]
             << ", " << total_demand[1] << "]\n";

        // ─── ALLOCATION: Choose rationing rule ───
        // 0 = FCFS, 1 = GMR (Gap-based Myopic Rationing)
        int64_t rationing_rule = 0;  // ← Change to 0 for FCFS, 1 for GMR

        int64_t satisfied_A[2] = {0, 0};
        int64_t satisfied_B[2] = {0, 0};

        if (rationing_rule == 0) {
            // FCFS: Customer A gets first pick
            cout << "\n[ALLOCATION - FCFS Rationing]\n";
            for (int64_t i = 0; i < 2; i++) {
                satisfied_A[i] = (int64_t)min((double)state.on_hand[i], demand_A[i]);
                state.on_hand[i] -= satisfied_A[i];
            }
            for (int64_t i = 0; i < 2; i++) {
                satisfied_B[i] = (int64_t)min((double)state.on_hand[i], demand_B[i]);
                state.on_hand[i] -= satisfied_B[i];
            }
        }
        else {
            // GMR: Allocate based on SLA gap (who needs it more?)
            cout << "\n[ALLOCATION - GMR (Gap-based Myopic Rationing)]\n";

            // Calculate current AFR for each customer (running average)
            double current_afr_A = (customers[0].total_demand_seen > 0) ?
                (double)customers[0].total_satisfied / customers[0].total_demand_seen : 0.0;
            double current_afr_B = (customers[1].total_demand_seen > 0) ?
                (double)customers[1].total_satisfied / customers[1].total_demand_seen : 0.0;

            // Calculate gap to target
            double gap_A = max(0.0, customers[0].target_afr - current_afr_A);
            double gap_B = max(0.0, customers[1].target_afr - current_afr_B);

            cout << "  Current AFR: A=" << fixed << setprecision(1) << (current_afr_A * 100.0)
                 << "% (gap=" << (gap_A * 100.0) << "%)" << ", B="
                 << (current_afr_B * 100.0) << "% (gap=" << (gap_B * 100.0) << "%)\n";

            // Allocate iteratively: give to customer with higher gap
            int64_t total_stock[2] = {state.on_hand[0], state.on_hand[1]};
            int64_t remaining_stock[2] = {state.on_hand[0], state.on_hand[1]};

            // For each SKU, allocate to the customer with higher gap
            for (int64_t i = 0; i < 2; i++) {
                // Total demand for this SKU
                double sku_demand_A = demand_A[i];
                double sku_demand_B = demand_B[i];
                double total_sku_demand = sku_demand_A + sku_demand_B;

                // Allocate based on gap ratio
                double gap_ratio_A = (total_sku_demand > 0) ? (gap_A * sku_demand_A / total_sku_demand) : 0;
                double gap_ratio_B = (total_sku_demand > 0) ? (gap_B * sku_demand_B / total_sku_demand) : 0;
                double total_gap_ratio = gap_ratio_A + gap_ratio_B;

                int64_t allocate_A = (total_gap_ratio > 0) ?
                    (int64_t)(remaining_stock[i] * gap_ratio_A / total_gap_ratio) :
                    remaining_stock[i] / 2;  // Split evenly if no gap

                int64_t allocate_B = remaining_stock[i] - allocate_A;

                // Actually satisfy demand up to allocation
                satisfied_A[i] = (int64_t)min((double)allocate_A, sku_demand_A);
                satisfied_B[i] = (int64_t)min((double)allocate_B, sku_demand_B);

                // If A didn't use all, B can have the rest
                int64_t unused_A = allocate_A - satisfied_A[i];
                satisfied_B[i] = (int64_t)min((double)(allocate_B + unused_A), sku_demand_B);

                remaining_stock[i] = 0;
            }

            state.on_hand[0] = 0;
            state.on_hand[1] = 0;
        }

        cout << "  Customer A satisfied: [" << satisfied_A[0] << ", " << satisfied_A[1] << "] / ["
             << (int64_t)demand_A[0] << ", " << (int64_t)demand_B[0] << "]\n";
        cout << "  Customer B satisfied: [" << satisfied_B[0] << ", " << satisfied_B[1] << "] / ["
             << (int64_t)demand_B[0] << ", " << (int64_t)demand_B[1] << "]\n";
        cout << "  Remaining on-hand: [" << state.on_hand[0] << ", " << state.on_hand[1] << "]\n";

        // ─── UPDATE CUSTOMER METRICS ───
        customers[0].total_demand_seen += (int64_t)(demand_A[0] + demand_A[1]);
        customers[0].total_satisfied += satisfied_A[0] + satisfied_A[1];
        customers[1].total_demand_seen += (int64_t)(demand_B[0] + demand_B[1]);
        customers[1].total_satisfied += satisfied_B[0] + satisfied_B[1];

        customers[0].periods_in_horizon++;
        customers[1].periods_in_horizon++;

        // ─── COST CALCULATION ───
        cout << "\n[COST CALCULATION]\n";

        double holding_cost = (state.on_hand[0] * items[0].holding_cost) +
                             (state.on_hand[1] * items[1].holding_cost);

        cout << "  Holding cost: " << fixed << setprecision(2) << holding_cost << "\n";

        double period_cost = holding_cost;

        // Penalty if review horizon ended
        double penalty = 0.0;
        if (state.customer_time_remaining[0] == 1) {
            double actual_afr_A = (customers[0].total_demand_seen > 0) ?
                (double)customers[0].total_satisfied / customers[0].total_demand_seen : 0.0;
            state.customer_afr[0] = actual_afr_A;

            if (actual_afr_A < customers[0].target_afr) {
                penalty += customers[0].penalty_cost * (customers[0].target_afr - actual_afr_A);
                cout << "  ⚠️  Customer A review period ended: AFR = " << (actual_afr_A * 100.0)
                     << "% (target " << (customers[0].target_afr * 100.0)
                     << "%) → Penalty = " << fixed << setprecision(2) << penalty << "\n";
            } else {
                cout << "  ✓ Customer A review period ended: AFR = " << (actual_afr_A * 100.0)
                     << "% ≥ target " << (customers[0].target_afr * 100.0) << "% → No penalty\n";
            }
        }

        if (state.customer_time_remaining[1] == 1) {
            double actual_afr_B = (customers[1].total_demand_seen > 0) ?
                (double)customers[1].total_satisfied / customers[1].total_demand_seen : 0.0;
            state.customer_afr[1] = actual_afr_B;

            if (actual_afr_B < customers[1].target_afr) {
                penalty += customers[1].penalty_cost * (customers[1].target_afr - actual_afr_B);
                cout << "  ⚠️  Customer B review period ended: AFR = " << (actual_afr_B * 100.0)
                     << "% (target " << (customers[1].target_afr * 100.0)
                     << "%) → Penalty = " << fixed << setprecision(2) << penalty << "\n";
            } else {
                cout << "  ✓ Customer B review period ended: AFR = " << (actual_afr_B * 100.0)
                     << "% ≥ target " << (customers[1].target_afr * 100.0) << "% → No penalty\n";
            }
        }

        period_cost += penalty;

        cout << "  TOTAL PERIOD COST: " << fixed << setprecision(2) << period_cost << "\n";
        total_cost += period_cost;

        // ─── UPDATE STATE ───
        state.customer_time_remaining[0]--;
        state.customer_time_remaining[1]--;

        // Reset at end of horizon
        if (state.customer_time_remaining[0] <= 0) {
            state.customer_time_remaining[0] = customers[0].review_horizon;
            customers[0].total_demand_seen = 0;
            customers[0].total_satisfied = 0;
        }
        if (state.customer_time_remaining[1] <= 0) {
            state.customer_time_remaining[1] = customers[1].review_horizon;
            customers[1].total_demand_seen = 0;
            customers[1].total_satisfied = 0;
        }

        cout << "\n";
    }

    // ========== SUMMARY ==========
    cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    cout << "║ SIMULATION SUMMARY                                                ║\n";
    cout << "╚═══════════════════════════════════════════════════════════════════╝\n";
    cout << "\nTotal cost over " << NUM_PERIODS << " periods: " << fixed << setprecision(2)
         << total_cost << "\n";
    cout << "Average cost per period: " << (total_cost / NUM_PERIODS) << "\n\n";

    return 0;
}
