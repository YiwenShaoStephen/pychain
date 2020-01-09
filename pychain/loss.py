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
    def forward(ctx, input, graphs):
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
        initial_probs = graphs.initial_probs
        num_states = graphs.num_states
        final_probs = graphs.final_probs
        leaky_hmm_coefficient = graphs.leaky_hmm_coefficient
        objf, input_grad, _ = pychain_C.forward_backward(
            forward_transitions,
            forward_transition_indices,
            forward_transition_probs,
            backward_transitions,
            backward_transition_indices,
            backward_transition_probs,
            initial_probs, final_probs,
            exp_input, num_states, leaky_hmm_coefficient)
        ctx.save_for_backward(input_grad)
        return objf

    @staticmethod
    def backward(ctx, objf_grad):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, objf_grad)
        return input_grad, None


class ChainLoss(nn.Module):
    def __init__(self, den_graph, avg=True):
        super(ChainLoss, self).__init__()
        self.den_graph = den_graph
        self.avg = avg

    def forward(self, x, num_graphs):
        batch_size = x.size(0)
        den_graphs = ChainGraphBatch(self.den_graph, batch_size)
        den_objf = ChainFunction.apply(
            x, den_graphs)
        num_objf = ChainFunction.apply(x, num_graphs)
        objf = -(num_objf - den_objf)
        if self.avg:
            objf = objf / (x.size(0) * x.size(1))
        return objf
