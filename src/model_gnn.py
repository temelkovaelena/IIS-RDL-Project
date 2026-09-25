"""Heterogeneous GraphSAGE over the PK-FK graph.

The building blocks come from relbench.modeling.nn, so the architecture matches the one
behind the published numbers. Three parts feed the message passing:

- HeteroEncoder turns each row into a vector, one column encoder per column type.
- HeteroTemporalEncoder adds how old a node is relative to the seed, in days.
- HeteroGraphSAGE runs one SAGEConv per edge type, num_layers times. Layers equal hops:
  two layers is where constructors and races first become visible from a driver.
"""

from __future__ import annotations

import torch
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP


class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: dict,
        num_layers: int = 2,
        channels: int = 128,
        out_channels: int = 1,
        aggr: str = "sum",
        norm: str = "batch_norm",
    ) -> None:
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        # Only node types that carry a time column get a temporal encoding. The static
        # ones (drivers, constructors, circuits) have no timestamp to be relative to.
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[node_type for node_type in data.node_types if "time" in data[node_type]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(self, batch: HeteroData, entity_table: str) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        # The first seed_time.size(0) rows are the seed nodes; the rest are sampled
        # neighbours that happen to share the entity type.
        return self.head(x_dict[entity_table][: seed_time.size(0)])
