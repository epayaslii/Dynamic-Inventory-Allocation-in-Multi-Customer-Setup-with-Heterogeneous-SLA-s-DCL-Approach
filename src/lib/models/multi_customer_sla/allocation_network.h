#ifndef DYNAPLEX_ALLOCATION_NETWORK_H
#define DYNAPLEX_ALLOCATION_NETWORK_H

#include <vector>
#include <cstdint>
#include <string>

namespace DynaPlex::Models::multi_customer_sla {

/**
 * AllocationNetwork: Neural Network policy for direct allocation under SLA constraints
 *
 * Architecture:
 *   Input:  36-dim state features
 *           [OH(1), OH(2), IP(1), IP(2),
 *            Q_pipeline(1), Q_pipeline(2), Q_pipeline(1), Q_pipeline(2),
 *            D_I, D_II,
 *            BO_I, BO_II,
 *            X_I, X_II, D̄_I, D̄_II, T_rem_I, T_rem_II, a_I, a_II,
 *            r_I, r_II, τ_I, τ_II,
 *            ρ_1, ρ_2,
 *            ā_1, ā_2]
 *
 *   Hidden:  256 → 256 → 128 units with ReLU
 *
 *   Output:  4 logits [logit_I1, logit_I2, logit_II1, logit_II2]
 *            (one per customer-item pair)
 *
 * Allocation derivation:
 *   For each item i:
 *     weights_i = softmax([logit_Ci] for C ∈ {I, II})
 *     A_{C,i} = weights_i[C] × available_i
 */
class AllocationNetwork {
public:
  struct Config {
    int64_t numberOfCustomers = 2;
    int64_t numberOfItems = 2;
    int64_t inputDim = 36;
    int64_t hiddenSize1 = 256;
    int64_t hiddenSize2 = 256;
    int64_t hiddenSize3 = 128;
    int64_t outputDim = 4;
    double learningRate = 0.001;
    bool useRandomInitialization = true;
  };

  /**
   * Constructor: initialize network with random or zero weights
   */
  explicit AllocationNetwork(const Config& config = Config());

  /**
   * Forward pass: state features → output logits
   *
   * @param state_features: 36-dim state vector
   * @return: 4-dim logits (one per customer-item pair)
   */
  std::vector<double> Forward(const std::vector<double>& state_features);

  /**
   * Get softmax-derived allocation matrix from logits
   *
   * Per item i, compute softmax over customers and scale by availability:
   *   weights_i = softmax([logit_I,i, logit_II,i])
   *   A_{C,i} = weights_i[C] × available_i
   *
   * Guarantees:
   *   - Σ_C A_{C,i}(t) = available_i(t) ∀ i
   *   - A_{C,i}(t) ≥ 0 ∀ C, i
   *
   * @param logits: 4-dim output from Forward()
   * @param availablePerItem: [available_1, available_2]
   * @return: 2×2 allocation matrix A[customer][item]
   */
  std::vector<std::vector<int64_t>> GetAllocationFromLogits(
    const std::vector<double>& logits,
    const std::vector<int64_t>& availablePerItem
  );

  /**
   * Backward pass: update weights using gradient descent
   *
   * Simplified version for manual weight updates.
   * Production code would use automatic differentiation.
   *
   * @param trajectory_cost: cumulative cost over trajectory
   * @param learning_rate: step size (can override config)
   */
  void Backward(double trajectory_cost, double learning_rate = -1.0);

  /**
   * Initialize weights from scratch
   *
   * Uses small random values ~ N(0, 0.01^2) for stability
   */
  void InitializeWeights();

  /**
   * Access raw weights (for inspection, testing)
   */
  const std::vector<std::vector<double>>& GetWeights1() const { return W1_; }
  const std::vector<std::vector<double>>& GetWeights4() const { return W4_; }

  /**
   * Save network weights to file
   *
   * Format: binary dump of all weight matrices
   */
  void SaveWeights(const std::string& filepath) const;

  /**
   * Load network weights from file
   */
  void LoadWeights(const std::string& filepath);

  /**
   * Get network configuration
   */
  const Config& GetConfig() const { return config_; }

private:
  Config config_;

  // Network weights (dense layers)
  std::vector<std::vector<double>> W1_;  // 36 × 256
  std::vector<double> b1_;               // 256

  std::vector<std::vector<double>> W2_;  // 256 × 256
  std::vector<double> b2_;               // 256

  std::vector<std::vector<double>> W3_;  // 256 × 128
  std::vector<double> b3_;               // 128

  std::vector<std::vector<double>> W4_;  // 128 × 4
  std::vector<double> b4_;               // 4 (output logits)

  // Cached activations for backprop (populated during Forward)
  std::vector<double> h1_, h2_, h3_;     // Hidden layer outputs
  std::vector<double> z4_;               // Final logits
  std::vector<double> input_cached_;     // Input (for gradient computation)

  // Helper: numerically stable softmax
  static std::vector<double> ComputeSoftmax(const std::vector<double>& logits);

  // Helper: matrix-vector multiplication with bias and ReLU
  static std::vector<double> DenseReLU(
    const std::vector<double>& input,
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>& bias
  );

  // Helper: linear layer (no activation)
  static std::vector<double> DenseLinear(
    const std::vector<double>& input,
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>& bias
  );
};

}  // namespace DynaPlex::Models::multi_customer_sla

#endif  // DYNAPLEX_ALLOCATION_NETWORK_H
