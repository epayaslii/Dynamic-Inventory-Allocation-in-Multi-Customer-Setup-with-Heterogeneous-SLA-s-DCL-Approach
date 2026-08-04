# Dynamic Inventory Allocation in Multi-Customer Systems with Heterogeneous SLAs

## Overview

This repository implements a sophisticated inventory management system for a central warehouse serving multiple customers, each with distinct SLA requirements. The key innovation is the **Dynamic Allocation Policy** learned via Deep Controlled Learning, which decides how to ration scarce inventory among customers to minimize total expected cost while respecting per-customer backorder allowance constraints.

### Core Problem

**Multi-Customer Inventory Allocation Under SLAs:**
- **Static Ordering:** A fixed base-stock policy (per item) maintains target inventory positions
- **Dynamic Allocation:** A learned policy decides which customers receive priority when inventory is scarce
- **SLA Compliance:** Each customer has a cumulative backorder allowance (β_k) per review horizon, with linear penalties if exceeded

## Model Formulation

### System Components

#### 1. Central Warehouse
- Stocks **I** items (SKUs) for **C** customers
- Maintains per-item base-stock levels (S_i)
- Replenishment lead time (L) periods

#### 2. Per-Customer SLA Metrics
Each customer k has:
- **Backorder Allowance (β_k):** Maximum cumulative backorder units tolerated per review horizon
- **Review Horizon (T_k):** Period length over which SLA compliance is measured
- **Penalty Cost (p_k):** Linear cost per unit of excess backorders

#### 3. Demand Model
- Per-customer, per-item demand: D_{k,i}(t) ~ distribution
- Optional time-varying (seasonal) demand: λ(t) = λ_base * (1 + A * sin(2π*t / P))

### Key State Variables

| Variable | Meaning | Size |
|----------|---------|------|
| **OH_i(t)** | On-hand inventory (can be negative) | I features |
| **Q_i(t-1...t-L)** | Pipeline orders in transit | I×L features |
| **BO_{k,i}(t)** | Backorder per customer-item | C×I features |
| **D_{k,i}(t)** | Realized demand (current period) | C×I features |
| **Σ̄BO_k(t)** | Cumulative backorder within review horizon | C features |
| **T_rem(t)** | Time remaining in review horizon | 1 feature |

**Total State Dimension:** I + I·L + C·I + C·I + C + 1 + C features

### Decision Variables

#### Replenishment (Static)
Each period, for each item i:
```
Q_i(t) = max(0, S_i - IP_i(t))
```
where IP_i(t) = on-hand + pipeline - outstanding backorders

#### Allocation (Dynamic)
When demand exceeds inventory, the learned policy π(s_t) decides:
```
A_{k,i}(t) for each customer k and item i
```
subject to:
```
Σ_k A_{k,i}(t) ≤ OH_i(t)
```

### Rationing Actions

Four built-in rationing strategies (extensible):

1. **Action 0 - FCFS (First-Come-First-Served):** Serve customers in index order
2. **Action 1 - SLA-Gap Myopic:** Prioritize customer furthest from SLA target
3. **Action 2 - Proportional:** Allocate proportional to demand (largest-remainder method)
4. **Action 3 - Cost-Based Greedy:** Assign each unit to customer with highest SLA-penalty risk

## Cost Structure

### Holding Cost (Every Period)
```
C_hold(t) = Σ_i h_i * OH_i(t+1)
```

### SLA Penalty (At Horizon End)
```
C_penalty = Σ_k p_k * max(0, Σ̄BO_k(T_k) - β_k)
```

## Period Workflow (9 Steps)

1. **Receive Orders:** Q_i(t-L) arrives and is added to on-hand inventory
2. **Observe IP:** Calculate inventory position: IP_i = OH_i + Σ_pipeline - Σ_BO
3. **Place Orders:** Q_i(t) = max(0, S_i - IP_i(t))
4. **Observe Demand:** D_{k,i}(t) realized per customer-item pair
5. **Check Rationing:** Is Σ_BO + Σ_D > OH? (per item)
6. **Allocate:** Policy π(s_t) decides allocation when rationing needed
7. **Update State:** OH and BO updated based on allocations
8. **Calculate Costs:** Holding costs every period, SLA penalties at horizon end
9. **Track Metrics:** Cumulative backorder, fill rate per customer

## Deep Controlled Learning

The system uses DCL to learn an optimal allocation policy by:

1. **State Features:** Extract relevant inventory, demand, and SLA metrics from state
2. **Action Sampling:** Evaluate different rationing strategies on sampled trajectories
3. **Progressive Refinement:** Sequential Halving algorithm identifies promising actions
4. **Neural Network Learning:** GC-LSN network learns features → action mapping

### DCL Components

- **Sequential Halving:** Eliminates unpromising actions early
- **Sample Generator:** Creates realistic demand scenarios
- **Uniform Action Selector:** Fallback strategy for exploration
- **Neural Network Weights:** Pre-trained GC-LSN weights in `gc-lsn-weights/`

## Repository Structure

```
.
├── src/
│   ├── lib/models/multi_customer_sla/
│   │   ├── mdp.h / mdp.cpp          # Core MDP model
│   │   ├── policies.h / policies.cpp # Policy implementations
│   │   └── allocation_network.h/cpp  # Allocation decision network
│   ├── algorithms/dcl/
│   │   ├── dcl.h / dcl.cpp           # Deep Controlled Learning
│   │   ├── sequentialhalving.*       # Sequential Halving algorithm
│   │   ├── uniformactionselector.*   # Uniform action selector
│   │   └── samplegenerator.cpp       # Sample generation
│   ├── executables/multi_customer_sla/
│   │   ├── multi_customer_sla.cpp           # Main simulation entry point
│   │   ├── multi_customer_sla_simple.cpp    # Simplified example
│   │   ├── dcl_allocation_learn_full.cpp    # Full DCL learning
│   │   ├── dcl_rationing_learning.cpp       # Rationing action learning
│   │   ├── bsl_sensitivity_analysis.cpp     # Base-stock level sensitivity
│   │   └── action3_comparison.cpp           # Cost-based greedy comparison
│   └── tests/
│       └── t_multi_customer_sla.cpp  # Unit tests
├── docs/
│   ├── WHITEBOARD_EQUATIONS_EXPLAINED.md
│   ├── DCL_ALLOCATION_LEARNING_RESULTS.md
│   ├── COST_COMPONENT_ANALYSIS.md
│   └── DCL_SIMULATION_DETAILED_WALKTHROUGH.md
├── gc-lsn-weights/
│   ├── GC-LSN.json                  # Network architecture
│   ├── GC-LSN.pth                   # Pre-trained weights
│   └── All-GC-LSN-Results.out       # Benchmark results
└── README.md (this file)
```

## Key Features

✅ **Multi-Customer Support:** Heterogeneous SLA constraints per customer  
✅ **Cumulative Backorder Tracking:** Realistic SLA compliance measurement  
✅ **Time-Varying Demand:** Optional seasonal/cyclic demand patterns  
✅ **Multiple Rationing Actions:** FCFS, SLA-gap, proportional, cost-based  
✅ **Deep Controlled Learning:** Policy optimization via neural networks  
✅ **Flexible Architecture:** Extensible for additional rationing strategies  

## Build & Compile

This code integrates with the DynaPlex framework. To build:

```bash
# Requires DynaPlex framework setup
# See https://github.com/tarkantemizoz/DynaPlex for complete setup

cd /path/to/DynaPlex
cmake -B build
cmake --build build
```

## Configuration Example

### Multi-Customer Multi-Item Setup

```cpp
VarGroup config;
config.Add("numberOfCustomers", 2);           // 2 customers
config.Add("numberOfItems", 3);               // 3 items

// Lead times (per item)
config.Add("leadTimes", std::vector<int64_t>{2, 3, 1});

// Holding costs (per item)
config.Add("holdingCosts", std::vector<double>{1.0, 0.8, 1.2});

// Per-customer SLA
config.Add("backorderAllowances", std::vector<int64_t>{6, 8});    // β_k
config.Add("reviewHorizons", std::vector<int64_t>{20, 20});       // T_k
config.Add("penaltyCosts", std::vector<double>{50.0, 60.0});      // p_k

// Base-stock levels (per item)
config.Add("baseStockLevel", std::vector<int64_t>{25, 30, 20});

// Demand (per customer, per item)
std::vector<double> demandRates;
// Customer 0: item 0,1,2
demandRates.push_back(1.0);   // λ_{0,0}
demandRates.push_back(0.8);   // λ_{0,1}
demandRates.push_back(0.5);   // λ_{0,2}
// Customer 1: item 0,1,2
demandRates.push_back(0.7);   // λ_{1,0}
demandRates.push_back(1.2);   // λ_{1,1}
demandRates.push_back(0.6);   // λ_{1,2}
config.Add("demandRates", demandRates);

// Demand distributions
config.Add("highDemandVariance", std::vector<int64_t>(6, 0));  // 0=Poisson, 1=Geometric

// Rationing actions
config.Add("totalRationingActions", 4);
config.Add("benchmarkRationingAction", 1);  // SLA-gap myopic

MDP mdp(config);
```

## Example: Simple Simulation

See `src/executables/multi_customer_sla/multi_customer_sla_simple.cpp` for a complete working example that:
1. Creates a 2-customer, 2-item inventory system
2. Runs a base-stock policy with proportional rationing
3. Tracks cumulative backorders and SLA penalties
4. Reports total cost and fill rates

## Learning Experiments

### DCL Allocation Learning (`dcl_allocation_learn_full.cpp`)
- Compares 4 rationing actions on 1,000+ samples
- Uses Sequential Halving for action elimination
- Measures average cost per action
- Outputs best-performing policy

### Sensitivity Analysis (`bsl_sensitivity_analysis.cpp`)
- Varies base-stock levels (S_i) from low to high
- Measures impact on holding costs and stockouts
- Helps optimize inventory investment

### Action Comparison (`action3_comparison.cpp`)
- Detailed cost breakdown per rationing action
- SLA compliance rates
- Identifies cost-benefit of each strategy

## Related Work

This research builds on:
- **METRIC** (Sherbrooke, 1968): Multi-echelon inventory optimization
- **Deep Reinforcement Learning** (Boute et al., 2022): Learning-based control policies
- **DCL Framework** (Temizoz et al., 2025): Deep Controlled Learning for inventory systems

## Authors

- **Eliz Payasli** (Eindhoven University of Technology)
- **Supervisors:** Willem van Jaarsveld, Tarkan Temizoz

## References

1. Sherbrooke, C.C. (1968). "METRIC: A Multi-Echelon Technique for Recoverable Item Control"
2. Temizoz, T., et al. (2025). "Deep Controlled Learning for Inventory Control"
3. Boute, R.N., et al. (2022). "Deep Reinforcement Learning for Inventory Control: A Roadmap"

## License

This project is released under the MIT License. See LICENSE file for details.

---

**Documentation:** See `docs/` folder for detailed technical papers and experiment results.  
**Pre-trained Models:** GC-LSN neural network weights available in `gc-lsn-weights/`.  
**Questions?** File an issue or contact the authors.
