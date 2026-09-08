"""
A custom PyTorch policy network for MultiCustomerSlaMDP -- an actual, working
replacement for src/lib/models/multi_customer_sla/allocation_network.{h,cpp}.

That C++ class is not a working trained network: its Backward() is an
explicit placeholder ("Stub for gradient-based update ... real implementation
uses autograd") that nudges only the last layer's weights by a crude
heuristic, never touches the cached activations it stores, and never
implements the chain rule through the earlier layers -- it could not have
learned anything as written. It was also hardcoded to exactly 2 customers x
2 items and bypassed DynaPlex's action/policy abstractions entirely (it
returns a full allocation matrix from one forward pass via softmax, rather
than choosing a single action). This module is not a port of that code; it
is a new, general (any |C|, |I|, lead_time), genuinely trainable network
using real PyTorch autograd, wired into the actual DCL training loop via
dynaplex.nn.Net.

Why a custom network at all, given dp.MLP already works
----------------------------------------------------------
dynaplex.nn.mlp_factory (what dp.MLP builds) only requires a single flat
input tensor -- which MultiCustomerSlaFeaturizer already produces, so
nothing here is FORCED by a format mismatch, unlike dynaplex's own docs
example of a shaped/multi-tensor featurizer needing a custom factory.
Confirmed working via train_and_compare.py (a real DCL training run,
verified end-to-end against the installed dynaplex package, producing a
trained NNAgent that beats the FCFS baseline). This module exists because a
plain MLP treats MultiCustomerSlaFeaturizer's flat vector as one
undifferentiated blob, ignoring that it is actually several semantically
distinct blocks (paper Sec. 3.3): per-item inventory, per-(customer,item)
backorder, per-(customer,item) demand, per-customer cumulative backorder,
time remaining, per-customer SLA allowance, and the alloc-item pointer (see
mdp.py's MultiCustomerSlaFeaturizer.write_features for the exact order/
sizes). BlockMLP encodes each block separately before combining them, so
the network doesn't have to learn from scratch, purely from data, that (say)
feature 4 and feature 41 both pertain to the same customer.

Usage: pass a Net referencing this factory to dp.DCL's `network=` argument,
e.g.:
    dp.DCL(mdp, policy_0, features=MultiCustomerSlaFeaturizer,
           network=dp.Net("network.block_mlp_factory",
                           number_of_customers=mdp.number_of_customers,
                           number_of_items=mdp.number_of_items,
                           lead_time=mdp.lead_time),
           ...)
"""
import torch
import torch.nn as nn


class BlockMLP(nn.Module):
    """Per-block encoder + shared trunk. Each of the 7 feature blocks listed
    above is passed through its own small Linear+ReLU encoder; the encoded
    blocks are concatenated and fed through a shared MLP trunk down to
    num_actions logits. A size-0 block (e.g. lead_time == 1, so there are no
    in-transit-pipeline features) is skipped entirely."""

    def __init__(
        self,
        name: str,
        block_sizes: list[int],
        block_hidden: int,
        trunk_hidden: list[int],
        num_actions: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.block_sizes = block_sizes
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(size, block_hidden), nn.ReLU()) if size > 0 else nn.Identity()
            for size in block_sizes
        )

        trunk_in = block_hidden * sum(1 for size in block_sizes if size > 0)
        layers: list[nn.Module] = []
        d = trunk_in
        for width in trunk_hidden:
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, num_actions))
        self.trunk = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch[self.name]
        offset = 0
        parts: list[torch.Tensor] = []
        for size, encoder in zip(self.block_sizes, self.encoders):
            if size > 0:
                parts.append(encoder(x[:, offset:offset + size]))
            offset += size
        return self.trunk(torch.cat(parts, dim=-1))


def block_mlp_factory(
    spec: dict,
    num_actions: int,
    number_of_customers: int,
    number_of_items: int,
    lead_time: int,
    block_hidden: int = 64,
    trunk_hidden: list[int] | None = None,
) -> nn.Module:
    """Net factory (dynaplex.nn.Net's contract: `fn(spec, num_actions,
    **kwargs) -> torch.nn.Module`). `spec` is MultiCustomerSlaFeaturizer's
    declared TensorSpec dict -- {"v": TensorSpec(..., (size,))}. The block
    sizes below must match write_features' emission order exactly (mdp.py);
    the assertion catches a drift between the two rather than silently
    misreading the tensor."""
    if len(spec) != 1:
        raise ValueError("block_mlp_factory needs a single-tensor spec (MultiCustomerSlaFeaturizer "
                         "always produces one, named 'v')")
    (name, tensor_spec), = spec.items()
    if len(tensor_spec.sample_shape) != 1:
        raise ValueError(f"block_mlp_factory needs a flat tensor, got shape {tensor_spec.sample_shape}")

    n_c, n_i, lt = number_of_customers, number_of_items, lead_time
    block_sizes = [
        n_i * lt,   # per-item on-hand + pipeline
        n_c * n_i,  # backorder[c, i]
        n_c * n_i,  # current_demand[c, i]
        n_c,        # cumulative_backorder[c]
        1,          # time_remaining
        n_c,        # backorder_allowances[c]
        n_i,        # alloc_item one-hot
    ]
    declared_size = tensor_spec.sample_shape[0]
    if sum(block_sizes) != declared_size:
        raise ValueError(
            f"block_mlp_factory's block sizes {block_sizes} sum to {sum(block_sizes)}, "
            f"but the featurizer's spec declares {declared_size} -- "
            "MultiCustomerSlaFeaturizer.write_features and this factory have drifted apart"
        )

    hidden = list(trunk_hidden) if trunk_hidden is not None else [128, 128]
    return BlockMLP(name, block_sizes, block_hidden, hidden, num_actions)
