"""Recurrent spiking neuron models: SpikingLSTM and SpikingRNN.

Enables temporal sequence modeling with spike-based computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spikeflow.neurons.lif import LIFNode


class SpikingLSTMCell(nn.Module):
    """LSTM cell with spiking membrane dynamics.

    Gates (i, f, o, g) use standard sigmoid/tanh, but the cell state
    interacts with LIF membrane potential for spike-based computation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)

        self.neuron = LIFNode(threshold=threshold, surrogate=surrogate, tau=tau)
        self.h: Tensor | None = None
        self.c: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.h is None:
            self.h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            self.c = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        i = torch.sigmoid(self.W_ii(x) + self.W_hi(self.h))
        f = torch.sigmoid(self.W_if(x) + self.W_hf(self.h))
        o = torch.sigmoid(self.W_io(x) + self.W_ho(self.h))
        g = torch.tanh(self.W_ig(x) + self.W_hg(self.h))

        self.c = f * self.c + i * g
        self.h = o * torch.tanh(self.c)

        return self.neuron(self.h)

    def reset_state(self):
        self.h = None
        self.c = None


class SpikingLSTM(nn.Module):
    """Multi-layer Spiking LSTM for temporal sequence tasks."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        threshold: float = 1.0,
        surrogate: str = "sigmoid",
        tau: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(SpikingLSTMCell(in_sz, hidden_size, threshold, surrogate, tau))

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x.transpose(0, 1)

        outputs = []
        for t in range(x.size(0)):
            inp = x[t]
            for cell in self.cells:
                inp = cell(inp)
            outputs.append(inp)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out

    def reset_state(self):
        for cell in self.cells:
            cell.reset_state()

    def set_sfa_mode(self, enabled: bool):
        for cell in self.cells:
            cell.neuron.set_sfa_mode(enabled)

    def set_readout(self, enabled: bool):
        for cell in self.cells:
            cell.neuron.set_readout(enabled)


class SpikingRNNCell(nn.Module):
    """Simple RNN cell with spiking activation."""

    def __init__(self, input_size: int, hidden_size: int, threshold: float = 1.0, surrogate: str = "sigmoid", tau: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.neuron = LIFNode(threshold=threshold, surrogate=surrogate, tau=tau)
        self.h: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.h is None:
            self.h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        self.h = self.neuron(self.linear(torch.cat([x, self.h], dim=-1)))
        return self.h

    def reset_state(self):
        self.h = None
