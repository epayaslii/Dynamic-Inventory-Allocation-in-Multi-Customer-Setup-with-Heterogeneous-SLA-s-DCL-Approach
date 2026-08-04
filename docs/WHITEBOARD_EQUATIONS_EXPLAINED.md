# Whiteboard Equations: Multi-Period SLA Model Formulation

**Source**: Whiteboard notes  
**Purpose**: Understand the mathematical formulation and how to implement in DynaPlex mdp.cpp/policies.cpp

---

## 1. Inventory Position (IP) - The Core Signal

### Equation (from whiteboard):
```
IP(t) = OH(t) + Σ Q(j)  for j = t-L to t-1
```

### Explanation:
- **OH(t)** = On-Hand inventory (physical stock at warehouse, can be negative)
- **Q(j)** = Order placed j periods ago
- **L** = Lead time (how many periods for order to arrive)
- **Σ Q(j) for j=t-L to t-1** = All orders in transit (will arrive in next L periods)

### Example (L=2):
```
Period 2:
  OH(2) = 1 (on-hand)
  Q(0) = 3.5 (arriving this period from order placed 2 periods ago)
  Q(1) = 8.0 (arriving next period from order placed 1 period ago)
  
  IP(2) = 1 + 3.5 + 8.0 = 12.5
  
Interpretation: We have 1 unit now, but 11.5 more coming, so position is 12.5
```

### Why It Matters:
- **IP > 0 even when OH < 0** means: Crisis now (backorder), Recovery planned (orders incoming)
- **This is the KEY INSIGHT**: When OH becomes negative, check IP for recovery signal

---

## 2. Order Quantity (Q) - Static Base-Stock Rule

### Equation (from whiteboard):
```
Q(t) = max(0, BS_L - IP(t))
```

Where **BS_L** = Base-Stock Level (target, FIXED in your current work)

### Explanation:
- At each period, observe current IP
- If IP < BS_L, order enough to bring IP back to target
- If IP ≥ BS_L, don't order (Q=0)

### Example (BS_L = 10):
```
Period 1: IP = 6.5, Q(1) = max(0, 10 - 6.5) = 3.5 units
Period 2: IP = 2.0, Q(2) = max(0, 10 - 2.0) = 8.0 units
Period 3: IP = -2.5 (negative!), Q(3) = max(0, 10 - (-2.5)) = 12.5 units (strong recovery order)
```

### Code Location:
- **mdp.cpp lines 86-100**: Currently implements this
- Orders are placed each period deterministically

---

## 3. Backorder Accumulation - Multi-Period SLA Tracking

### Equation (from whiteboard):
```
BO_i^(w)(t) = BO_i^(w)(t-1) + D_i^(w)(t) - a_i^(w)(t)
```

Where:
- **BO_i^(w)(t)** = Backorder for customer i at period t (cumulative unmet demand)
- **D_i^(w)(t)** = Demand from customer i at period t
- **a_i^(w)(t)** = Allocation to customer i at period t (what they actually receive)
- **w** = Index for "window" or review period

### Explanation:
- Backorders accumulate: each period, add new demand and subtract what we allocate
- **BO can grow negative** (we owe customers more and more units)
- This tracks the SLA compliance over the review horizon

### Example (2 customers, 1 item):
```
Period 1:
  BO_A(1) = 0 + 2.0 - 2.0 = 0 (A's demand fully met)
  BO_B(1) = 0 + 1.5 - 1.5 = 0 (B's demand fully met)

Period 2:
  BO_A(2) = 0 + 2.5 - 2.5 = 0 (still satisfied)
  BO_B(2) = 0 + 2.0 - 2.0 = 0 (still satisfied)

Period 3 (SHORTAGE):
  Available stock = 1 unit
  Allocation A gets all: a_A = 1, a_B = 0
  
  BO_A(3) = 0 + 2.0 - 1.0 = 1.0 (A now has 1 unit backorder)
  BO_B(3) = 0 + 2.5 - 0.0 = 2.5 (B now has 2.5 unit backorder)

Period 4:
  BO_A(4) = 1.0 + 3.0 - 0.0 = 4.0 (A's backorder grows)
  BO_B(4) = 2.5 + 1.5 - 0.0 = 4.0 (B's backorder grows)
```

### Code Location:
- **mdp.cpp**: Not explicitly tracking cumulative backorders per customer
- **Need to add**: Vector tracking BO_i^(w)(t) for each customer over horizon

---

## 4. Allocation Constraint - Can't Over-Allocate

### Equation (from whiteboard):
```
a_i^(w)(t) ≤ BO_i^(w)(t)
```

Wait, this seems off. Should be:

```
a_i^(w)(t) ≤ min(D_i^(w)(t), Available(t))
```

### Explanation:
- You can't allocate more than:
  1. What the customer demands, OR
  2. What's physically available

### Example:
```
Period 3:
  Available stock = 1 unit
  Demand_A = 2.0, Demand_B = 2.5
  
  Constraint: a_A + a_B ≤ 1
  
  FCFS (Priority A):
    a_A = min(2.0, 1) = 1.0
    a_B = min(2.5, 1-1) = 0
```

### Code Location:
- **policies.cpp**: Implements different allocation strategies
- **Need to add**: Validation that sum(allocations) ≤ available

---

## 5. State Definition - What the System Needs to Track

### Equation (from whiteboard):
```
State = {BO_i^(t)(BO_i^(1)(t), BO_II^(t), ...), OH(t), Q^(t-1), ..., Q^(t-L)}
```

### Explanation:
- **BO_i^(t)** = Backorder vector for each customer (cumulative unmet demand)
- **OH(t)** = On-hand inventory vector for each item
- **Q^(t-1), ..., Q^(t-L)** = All orders currently in transit (L periods of orders)
- **Optional**: Time remaining in horizon, current AFR, etc.

### Example (2 customers, 1 item, L=2):
```
State at period 3:
  BO = [1.0, 2.5]        (Customer A owes 1, B owes 2.5)
  OH = [−2.5]            (On-hand is negative = backorder state)
  Q_pipeline = [3.5, 8.0] (Order from period 0 arriving now, order from period 1 arriving next)
  time_remaining = [27, 27] (27 periods left for each customer's SLA)
  AFR = [1.0, 1.0]       (Aggregate fill rates so far)
```

### Code Location:
- **mdp.h**: State structure definition (around line 30-60)
- **Current**: `state_vector` is std::vector<std::deque<int64_t>>
- **Need to add**: Explicit BO tracking, AFR tracking per customer

---

## 6. Cost Function - Period Cost Breakdown

### Equations (from whiteboard, multiple formulations):

**Simple Version**:
```
Cost(t) = Σ h_i · max(0, OH_i(t)) + Σ b_i · max(0, -OH_i(t))
```

**With SLA Penalties**:
```
Cost(t) = Σ h_i · max(0, OH_i(t)) 
        + Σ b_i · max(0, -OH_i(t)) 
        + Σ p_k · PenaltyIfSLAMissed_k(t)
```

**At Horizon End (t = T_k for customer k)**:
```
PenaltyIfSLAMissed_k(T_k) = p_k · max(0, β_k* · D_k_total - Satisfied_k_total)
```

### Explanation:

**Holding Cost**: `h_i · max(0, OH_i(t))`
- Cost per unit of inventory held
- Only when OH > 0 (can't hold negative inventory)
- Example: h=1.0, OH=6.5 → holding cost = 6.5

**Backorder Cost**: `b_i · max(0, -OH_i(t))`
- Cost per unit of backorder (owe customers)
- Only when OH < 0 (negative inventory)
- Example: b=0.5, OH=-2.5 → backorder cost = 1.25

**SLA Penalty (at horizon end)**:
- If AFR_k < β_k* (target), incur penalty
- Penalty = cost_per_unit × (units short of target fill rate)

### Example (Periods 1-2):
```
Period 1:
  OH = 6.5
  Holding Cost = 1.0 × 6.5 = 6.5
  Backorder Cost = 0.5 × max(0, -6.5) = 0
  Total = 6.5

Period 3 (SHORTAGE):
  OH = -2.5
  Holding Cost = 1.0 × max(0, -2.5) = 0
  Backorder Cost = 0.5 × max(0, 2.5) = 1.25
  Total = 1.25
```

### Code Location:
- **mdp.cpp lines 150-200**: Cost calculation in ModifyStateWithEvent
- **Current**: Calculates holding + backorder correctly
- **Missing**: Explicit SLA penalty tracking and accumulation

---

## 7. Aggregate Fill Rate (AFR) - SLA Compliance Metric

### Equation (from whiteboard):
```
AFR_k = Σ Satisfied_k(t) / Σ Demand_k(t)  for t in review horizon
```

### Explanation:
- Cumulative: sum of all satisfied demand for customer k, divided by total demand
- Measured over review horizon T_k (e.g., 30 periods)
- Target: β_k* (e.g., 0.95 for 95% fill rate)

### Example (6 periods):
```
Customer A:
  Total Demand = 2 + 2.5 + 2 + 3 + 2.5 + 2 = 14
  Total Satisfied = 2 + 2.5 + 1 + 0 + 0 + 0 = 5.5
  AFR_A = 5.5 / 14 = 0.393 (39.3%)
  Target = 0.95 (95%)
  → MISS SLA by 55.7%
```

### Code Location:
- **mdp.cpp**: Implicitly tracked in state, but not explicitly exposed
- **Need to add**: AFR vector per customer, tracked throughout horizon

---

## Summary Table: Where Equations Live in Code

| Equation | Current Location | Status | What's Missing |
|----------|------------------|--------|-----------------|
| IP(t) = OH + Σ Q(j) | mdp.cpp:88-96 | ✅ Implemented | Could be more explicit |
| Q(t) = max(0, BS_L - IP) | mdp.cpp:99-100 | ✅ Implemented | Works correctly |
| BO_i(t) = BO_i(t-1) + D_i - a_i | mdp.cpp:150+ | ⚠️ Implicit | Need explicit per-customer tracking |
| a_i ≤ Available | policies.cpp | ✅ Implemented | Allocation logic exists |
| State def | mdp.h:30-60 | ⚠️ Basic | Missing explicit BO, AFR per customer |
| Cost = h·OH + b·(-OH) + penalties | mdp.cpp:150-200 | ✅ Mostly | SLA penalties could be clearer |
| AFR_k = Σ Satisfied / Σ Demand | mdp.cpp | ⚠️ Implicit | Need explicit AFR tracking vector |

---

## Next Steps: Implementation Roadmap

### Phase 1: Clarify State Tracking
Add to `mdp.h` State struct:
```cpp
struct State {
    std::vector<double> backorder_cumulative;  // BO_i(t) for each customer
    std::vector<double> afr_current;           // Current AFR for each customer
    std::vector<int64_t> time_remaining;       // Periods left in horizon for each
    // ... existing fields ...
};
```

### Phase 2: Enhance Cost Calculation
In `mdp.cpp` ModifyStateWithEvent:
```cpp
// Explicit backorder accumulation
for (int k = 0; k < numberOfCustomers; k++) {
    backorder_cumulative[k] = backorder_cumulative[k] + demand[k] - allocation[k];
    double backorder_cost = penaltyCosts[k] * max(0.0, backorder_cumulative[k]);
}

// AFR tracking
cumulative_satisfied[k] += allocation[k];
cumulative_demand[k] += demand[k];
afr_current[k] = cumulative_satisfied[k] / cumulative_demand[k];

// SLA penalty at horizon end
if (time_remaining[k] == 0) {
    double shortage = max(0.0, targetFillRates[k] * cumulative_demand[k] - cumulative_satisfied[k]);
    penalty_cost += penaltyCosts[k] * shortage;
}
```

### Phase 3: Validate Allocation Constraints
In `policies.cpp`:
```cpp
// Ensure a_i <= available
double total_allocated = 0;
for (int k = 0; k < numberOfCustomers; k++) {
    allocation[k] = min(demand[k], available - total_allocated);
    total_allocated += allocation[k];
}
```

### Phase 4: Add Tests
Create `dcl_multi_period_trace.cpp` to validate:
- IP calculation
- Q ordering
- BO accumulation
- AFR tracking
- Cost calculation

---

## Key Insight: Why This Formulation Matters

The whiteboard equations formalize a **multi-period, multi-customer** problem where:
1. **Orders take time** (lead time L)
2. **Demand is stochastic** and varies per customer
3. **Stock is shared** (allocation decision each period)
4. **SLAs are time-windowed** (reviewed over horizon T_k)
5. **Fairness and efficiency conflict** (high penalty customer vs. low penalty customer)

Your DCL work learns to solve: **What allocation strategy minimizes cost while respecting SLA targets?**

The whiteboard equations give the mathematical foundation for evaluating that tradeoff.
