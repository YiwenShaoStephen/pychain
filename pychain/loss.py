# Copyright       2019 Yiwen Shao
#                 2020 Yiming Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from pychain.graph import ChainGraphBatch
import pychain_C


class ChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_lengths, graphs, leaky_coefficient):
        exp_input = input.clamp(-30, 30).exp()
        B = input.size(0)
        if B != graphs.batch_size:
            raise ValueError("input batch size {0} does not equal to graph batch size {1}"
                             .format(B, graphs.batch_size))
        forward_transitions = graphs.forward_transitions
        forward_transition_indices = graphs.forward_transition_indices
        forward_transition_probs = graphs.forward_transition_probs
        backward_transitions = graphs.backward_transitions
        backward_transition_indices = graphs.backward_transition_indices
        backward_transition_probs = graphs.backward_transition_probs
        leaky_probs = graphs.leaky_probs
        num_states = graphs.num_states
        final_probs = graphs.final_probs
        start_state = graphs.start_state
        leaky_coefficient = leaky_coefficient
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_lengths, batch_first=True)
        batch_sizes = packed_data.batch_sizes
        input_lengths = input_lengths.cpu()
        objf, input_grad, _ = pychain_C.forward_backward(
            forward_transitions,
            forward_transition_indices,
            forward_transition_probs,
            backward_transitions,
            backward_transition_indices,
            backward_transition_probs,
            leaky_probs, final_probs,
            start_state,
            exp_input,
            batch_sizes,
            input_lengths,
            num_states,
            leaky_coefficient)
        ctx.save_for_backward(input_grad)
        return objf

    @staticmethod
    def backward(ctx, objf_grad):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, objf_grad)
        return input_grad, None, None, None


class ChainLoss(nn.Module):
    def __init__(self, den_graph, den_leaky_coefficient=1e-5, num_leaky_coefficient=1e-20, avg=True):
        super(ChainLoss, self).__init__()
        self.den_graph = den_graph
        self.avg = avg
        self.den_leaky_coefficient = den_leaky_coefficient
        self.num_leaky_coefficient = num_leaky_coefficient

    def forward(self, x, x_lengths, num_graphs):
        batch_size = x.size(0)
        den_graphs = ChainGraphBatch(self.den_graph, batch_size)
        den_objf = ChainFunction.apply(
            x, x_lengths, den_graphs, self.den_leaky_coefficient)
        num_objf = ChainFunction.apply(
            x, x_lengths, num_graphs, self.num_leaky_coefficient)
        objf = -(num_objf - den_objf)
        if self.avg:
            objf = objf / x_lengths.sum()
        return objf
