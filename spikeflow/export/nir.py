"""NIR (Neural Intermediate Representation) export for neuromorphic deployment.

NIR is the open standard for exchanging SNN models between frameworks.
Export your trained SpikeFlow model to NIR format for deployment on:
    - Intel Loihi (via Lava)
    - SpiNNaker (via Nengo)
    - custom neuromorphic hardware

Based on: https://github.com/INL/NeuromorphicIntermediateRepresentation
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


class NIRExporter:
    """Export SpikeFlow models to NIR format.

    NIR is a graph-based intermediate representation that captures:
        - Neuron dynamics (LIF parameters)
        - Connectivity (weights, biases)
        - Time constants (tau, threshold)

    Usage:
        exporter = NIRExporter()
        nir_graph = exporter.export(model, input_shape=(1, 3, 224, 224))
        exporter.save(nir_graph, "model.nir")
    """

    def __init__(self):
        self.nodes = []
        self.edges = []

    def export(self, model: nn.Module, input_shape: tuple[int, ...] = (1, 3, 224, 224)) -> dict[str, Any]:
        """Export model to NIR format.

        Returns a dictionary describing the neural network graph.
        """
        self.nodes = []
        self.edges = []

        node_id = 0
        prev_node = None

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                node = {
                    "id": node_id,
                    "name": name,
                    "type": "linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "weight": module.weight.data.cpu().numpy(),
                    "bias": module.bias.data.cpu().numpy() if module.bias is not None else None,
                }
                self.nodes.append(node)
                if prev_node is not None:
                    self.edges.append({"from": prev_node, "to": node_id})
                prev_node = node_id
                node_id += 1

            elif isinstance(module, nn.Conv2d):
                node = {
                    "id": node_id,
                    "name": name,
                    "type": "conv2d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding,
                    "weight": module.weight.data.cpu().numpy(),
                    "bias": module.bias.data.cpu().numpy() if module.bias is not None else None,
                }
                self.nodes.append(node)
                if prev_node is not None:
                    self.edges.append({"from": prev_node, "to": node_id})
                prev_node = node_id
                node_id += 1

            elif hasattr(module, "threshold_module"):
                # Spiking neuron node
                from spikeflow.neurons.if_cell import IFNode
                from spikeflow.neurons.lif import LIFNode

                if isinstance(module, LIFNode):
                    node = {
                        "id": node_id,
                        "name": name,
                        "type": "lif",
                        "threshold": module.threshold_module.threshold,
                        "tau": module.tau,
                        "dt": module.dt,
                        "v_reset": module.v_reset,
                    }
                elif isinstance(module, IFNode):
                    node = {
                        "id": node_id,
                        "name": name,
                        "type": "if",
                        "threshold": module.threshold_module.threshold,
                        "v_reset": module.v_reset,
                    }
                else:
                    node = {
                        "id": node_id,
                        "name": name,
                        "type": "lif",
                        "threshold": 1.0,
                        "tau": 2.0,
                    }

                self.nodes.append(node)
                if prev_node is not None:
                    self.edges.append({"from": prev_node, "to": node_id})
                prev_node = node_id
                node_id += 1

        return {
            "format": "spikeflow_nir",
            "version": "0.1.0",
            "nodes": self.nodes,
            "edges": self.edges,
            "input_shape": input_shape,
        }

    def save(self, nir_graph: dict, path: str):
        """Save NIR graph to file."""
        import json

        with open(path, "w") as f:
            json.dump(nir_graph, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else list(o))

    def load(self, path: str) -> dict:
        """Load NIR graph from file."""
        import json
        with open(path) as f:
            return json.load(f)

    def summary(self, nir_graph: dict) -> str:
        """Print summary of NIR graph."""
        lines = [
            "=" * 50,
            "NIR Graph Summary",
            "=" * 50,
            f"  Nodes: {len(nir_graph['nodes'])}",
            f"  Edges: {len(nir_graph['edges'])}",
            f"  Input: {nir_graph['input_shape']}",
            "",
            "  Layers:",
        ]
        for node in nir_graph["nodes"]:
            lines.append(f"    [{node['type']:>6}] {node['name']}")
        lines.append("=" * 50)
        return "\n".join(lines)
