#include "allocation_network.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <random>
#include <fstream>
#include <numeric>

namespace DynaPlex::Models::multi_customer_sla {

AllocationNetwork::AllocationNetwork(const Config& config)
		: config_(config)
{
	if (config_.useRandomInitialization) {
		InitializeWeights();
	} else {
		// Zero initialization for testing
		W1_.assign(config_.inputDim, std::vector<double>(config_.hiddenSize1, 0.0));
		b1_.assign(config_.hiddenSize1, 0.0);

		W2_.assign(config_.hiddenSize1, std::vector<double>(config_.hiddenSize2, 0.0));
		b2_.assign(config_.hiddenSize2, 0.0);

		W3_.assign(config_.hiddenSize2, std::vector<double>(config_.hiddenSize3, 0.0));
		b3_.assign(config_.hiddenSize3, 0.0);

		W4_.assign(config_.hiddenSize3, std::vector<double>(config_.outputDim, 0.0));
		b4_.assign(config_.outputDim, 0.0);
	}
}

void AllocationNetwork::InitializeWeights()
{
	// Xavier initialization: weights ~ N(0, sqrt(2 / (fan_in + fan_out)))
	// Simplified: use small random values

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dist(0.0, 0.01);

	auto init_matrix = [&](std::vector<std::vector<double>>& W) {
		for (auto& row : W)
			for (auto& w : row)
				w = dist(gen);
	};

	auto init_bias = [&](std::vector<double>& b) {
		for (auto& v : b)
			v = dist(gen);
	};

	init_matrix(W1_);
	init_bias(b1_);

	init_matrix(W2_);
	init_bias(b2_);

	init_matrix(W3_);
	init_bias(b3_);

	init_matrix(W4_);
	init_bias(b4_);
}

std::vector<double> AllocationNetwork::Forward(const std::vector<double>& state_features)
{
	assert(state_features.size() == static_cast<size_t>(config_.inputDim));

	input_cached_ = state_features;

	// Layer 1: input (36) → h1 (256) with ReLU
	h1_ = DenseReLU(state_features, W1_, b1_);
	assert(h1_.size() == static_cast<size_t>(config_.hiddenSize1));

	// Layer 2: h1 (256) → h2 (256) with ReLU
	h2_ = DenseReLU(h1_, W2_, b2_);
	assert(h2_.size() == static_cast<size_t>(config_.hiddenSize2));

	// Layer 3: h2 (256) → h3 (128) with ReLU
	h3_ = DenseReLU(h2_, W3_, b3_);
	assert(h3_.size() == static_cast<size_t>(config_.hiddenSize3));

	// Layer 4: h3 (128) → z4 (4) linear (no activation)
	z4_ = DenseLinear(h3_, W4_, b4_);
	assert(z4_.size() == static_cast<size_t>(config_.outputDim));

	return z4_;
}

std::vector<double> AllocationNetwork::DenseReLU(
		const std::vector<double>& input,
		const std::vector<std::vector<double>>& weights,
		const std::vector<double>& bias)
{
	assert(input.size() == weights.size());
	assert(weights.size() > 0);
	assert(bias.size() == weights[0].size());

	std::vector<double> output(bias.size());
	std::copy(bias.begin(), bias.end(), output.begin());

	for (size_t i = 0; i < input.size(); i++) {
		for (size_t j = 0; j < output.size(); j++) {
			output[j] += weights[i][j] * input[i];
		}
	}

	// Apply ReLU
	for (auto& val : output)
		val = std::max(0.0, val);

	return output;
}

std::vector<double> AllocationNetwork::DenseLinear(
		const std::vector<double>& input,
		const std::vector<std::vector<double>>& weights,
		const std::vector<double>& bias)
{
	assert(input.size() == weights.size());
	assert(weights.size() > 0);
	assert(bias.size() == weights[0].size());

	std::vector<double> output(bias.size());
	std::copy(bias.begin(), bias.end(), output.begin());

	for (size_t i = 0; i < input.size(); i++) {
		for (size_t j = 0; j < output.size(); j++) {
			output[j] += weights[i][j] * input[i];
		}
	}

	return output;
}

std::vector<double> AllocationNetwork::ComputeSoftmax(const std::vector<double>& logits)
{
	// Numerically stable softmax: subtract max before exp
	double max_logit = *std::max_element(logits.begin(), logits.end());

	std::vector<double> exp_logits(logits.size());
	double sum_exp = 0.0;
	for (size_t i = 0; i < logits.size(); i++) {
		exp_logits[i] = std::exp(logits[i] - max_logit);
		sum_exp += exp_logits[i];
	}

	std::vector<double> softmax(logits.size());
	for (size_t i = 0; i < logits.size(); i++) {
		softmax[i] = exp_logits[i] / sum_exp;
	}

	return softmax;
}

std::vector<std::vector<int64_t>> AllocationNetwork::GetAllocationFromLogits(
		const std::vector<double>& logits,
		const std::vector<int64_t>& availablePerItem)
{
	assert(logits.size() == static_cast<size_t>(config_.outputDim));
	assert(availablePerItem.size() == static_cast<size_t>(config_.numberOfItems));

	std::vector<std::vector<int64_t>> allocation(
			config_.numberOfCustomers,
			std::vector<int64_t>(config_.numberOfItems, 0));

	// For each item, compute softmax over customers and allocate
	for (int64_t i = 0; i < config_.numberOfItems; i++) {
		// Extract logits for this item across all customers
		std::vector<double> item_logits(config_.numberOfCustomers);
		for (int64_t c = 0; c < config_.numberOfCustomers; c++) {
			item_logits[c] = logits[c * config_.numberOfItems + i];
		}

		// Compute softmax weights
		std::vector<double> weights = ComputeSoftmax(item_logits);

		// Allocate stock based on weights
		int64_t total_allocated = 0;
		for (int64_t c = 0; c < config_.numberOfCustomers - 1; c++) {
			allocation[c][i] = static_cast<int64_t>(weights[c] * availablePerItem[i]);
			total_allocated += allocation[c][i];
		}

		// Last customer gets remainder (ensures exact allocation)
		allocation[config_.numberOfCustomers - 1][i] = availablePerItem[i] - total_allocated;
	}

	return allocation;
}

void AllocationNetwork::Backward(double trajectory_cost, double learning_rate)
{
	// Stub for gradient-based update
	// In production, would compute full backpropagation through all layers
	// For now, simple policy: reduce logits slightly if cost is high

	if (learning_rate < 0.0)
		learning_rate = config_.learningRate;

	// Penalty: if cost is high, push logits toward balanced allocation
	// This is a placeholder; real implementation uses autograd
	const double cost_penalty_factor = 0.001 * trajectory_cost;

	for (auto& row : W4_) {
		for (auto& w : row) {
			w -= learning_rate * cost_penalty_factor;
		}
	}

	for (auto& b : b4_) {
		b -= learning_rate * cost_penalty_factor * 0.1;
	}
}

void AllocationNetwork::SaveWeights(const std::string& filepath) const
{
	std::ofstream file(filepath, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Failed to open file for saving weights: " + filepath);

	// Save dimensions
	int64_t dims[5] = {
			static_cast<int64_t>(W1_.size()),
			static_cast<int64_t>(W1_[0].size()),
			static_cast<int64_t>(W2_[0].size()),
			static_cast<int64_t>(W3_[0].size()),
			static_cast<int64_t>(W4_[0].size())};
	file.write(reinterpret_cast<char*>(dims), sizeof(dims));

	// Save W1
	for (const auto& row : W1_)
		file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
	file.write(reinterpret_cast<const char*>(b1_.data()), b1_.size() * sizeof(double));

	// Save W2
	for (const auto& row : W2_)
		file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
	file.write(reinterpret_cast<const char*>(b2_.data()), b2_.size() * sizeof(double));

	// Save W3
	for (const auto& row : W3_)
		file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
	file.write(reinterpret_cast<const char*>(b3_.data()), b3_.size() * sizeof(double));

	// Save W4
	for (const auto& row : W4_)
		file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
	file.write(reinterpret_cast<const char*>(b4_.data()), b4_.size() * sizeof(double));

	file.close();
}

void AllocationNetwork::LoadWeights(const std::string& filepath)
{
	std::ifstream file(filepath, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Failed to open file for loading weights: " + filepath);

	// Read dimensions
	int64_t dims[5];
	file.read(reinterpret_cast<char*>(dims), sizeof(dims));

	// Verify dimensions match current network
	if (dims[0] != config_.inputDim || dims[1] != config_.hiddenSize1 ||
			dims[2] != config_.hiddenSize2 || dims[3] != config_.hiddenSize3 ||
			dims[4] != config_.outputDim) {
		throw std::runtime_error("Weight file dimensions do not match network configuration");
	}

	// Load W1 and b1
	for (auto& row : W1_)
		file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
	file.read(reinterpret_cast<char*>(b1_.data()), b1_.size() * sizeof(double));

	// Load W2 and b2
	for (auto& row : W2_)
		file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
	file.read(reinterpret_cast<char*>(b2_.data()), b2_.size() * sizeof(double));

	// Load W3 and b3
	for (auto& row : W3_)
		file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
	file.read(reinterpret_cast<char*>(b3_.data()), b3_.size() * sizeof(double));

	// Load W4 and b4
	for (auto& row : W4_)
		file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
	file.read(reinterpret_cast<char*>(b4_.data()), b4_.size() * sizeof(double));

	file.close();
}

}  // namespace DynaPlex::Models::multi_customer_sla
