# Copyright       2019  Yiwen Shao
#                 2020  Yiming Wang
#                 2020  Facebook Inc.  (author: Vimal Manohar)

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

import logging

import torch
import torch.nn as nn
from pychain.graph import ChainGraphBatch
import pychain_C


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_lengths, graphs, leaky_coefficient=1e-5):
        B = input.size(0)
        if B != graphs.batch_size:
            raise ValueError("input batch size {0} does not equal to graph batch size {1}"
                             .format(B, graphs.batch_size))
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_lengths, batch_first=True)
        batch_sizes = packed_data.batch_sizes
        input_lengths = input_lengths.cpu()
        if not graphs.log_domain:  # for the denominator
            exp_input = input.clamp(-30, 30).exp()
            objf, input_grad, ok = pychain_C.forward_backward(
                graphs.forward_transitions,
                graphs.forward_transition_indices,
                graphs.forward_transition_probs,
                graphs.backward_transitions,
                graphs.backward_transition_indices,
                graphs.backward_transition_probs,
                graphs.leaky_probs,
                graphs.initial_probs,
                graphs.final_probs,
                graphs.start_state,
                exp_input,
                batch_sizes,
                input_lengths,
                graphs.num_states,
                leaky_coefficient,
            )
        else:  # for the numerator
            objf, log_probs_grad, ok = pychain_C.forward_backward_log_domain(
                graphs.forward_transitions,
                graphs.forward_transition_indices,
                graphs.forward_transition_probs,
                graphs.backward_transitions,
                graphs.backward_transition_indices,
                graphs.backward_transition_probs,
                graphs.initial_probs,
                graphs.final_probs,
                graphs.start_state,
                input,
                batch_sizes,
                input_lengths,
                graphs.num_states,
            )
            input_grad = log_probs_grad.exp()

        ctx.save_for_backward(input_grad)
        return objf

    @staticmethod
    def backward(ctx, objf_grad):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, objf_grad)

        return input_grad, None, None, None


class ChainLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_lengths, num_graphs, den_graph, leaky_coefficient=1e-5):
        B = input.size(0)
        if B != num_graphs.batch_size:
            raise ValueError("input batch size {0} does not equal to graph batch size {1}"
                             .format(B, num_graphs.batch_size))
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_lengths, batch_first=True)
        batch_sizes = packed_data.batch_sizes
        input_lengths = input_lengths.cpu()

        den_graphs = ChainGraphBatch(den_graph, B)
        exp_input = input.clamp(-30, 30).exp()
        den_objf, input_grad, denominator_ok = pychain_C.forward_backward(
            den_graphs.forward_transitions,
            den_graphs.forward_transition_indices,
            den_graphs.forward_transition_probs,
            den_graphs.backward_transitions,
            den_graphs.backward_transition_indices,
            den_graphs.backward_transition_probs,
            den_graphs.leaky_probs,
            den_graphs.initial_probs,
            den_graphs.final_probs,
            den_graphs.start_state,
            exp_input,
            batch_sizes,
            input_lengths,
            den_graphs.num_states,
            leaky_coefficient,
        )
        denominator_ok = denominator_ok.item()

        assert num_graphs.log_domain
        num_objf, log_probs_grad, numerator_ok = pychain_C.forward_backward_log_domain(
            num_graphs.forward_transitions,
            num_graphs.forward_transition_indices,
            num_graphs.forward_transition_probs,
            num_graphs.backward_transitions,
            num_graphs.backward_transition_indices,
            num_graphs.backward_transition_probs,
            num_graphs.initial_probs,
            num_graphs.final_probs,
            num_graphs.start_state,
            input,
            batch_sizes,
            input_lengths,
            num_graphs.num_states,
        )
        numerator_ok = numerator_ok.item()

        loss = -num_objf + den_objf

        if (loss - loss) != 0.0 or not denominator_ok or not numerator_ok:
            default_loss = 10
            input_grad = torch.zeros_like(input)
            logger.warn(
                f"Loss is {loss}) and denominator computation "
                f"(if done) returned {denominator_ok} "
                f"and numerator computation returned {numerator_ok} "
                f", setting loss to {default_loss} per frame"
            )
            loss = torch.full_like(num_objf, default_loss * input_lengths.sum())
        else:
            num_grad = log_probs_grad.exp()
            input_grad -= num_grad

        ctx.save_for_backward(input_grad)
        return loss

    @staticmethod
    def backward(ctx, objf_grad):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, objf_grad)

        return input_grad, None, None, None, None


class ChainLoss(nn.Module):
    def __init__(self, den_graph, leaky_coefficient=1e-5, avg=True):
        super(ChainLoss, self).__init__()
        self.den_graph = den_graph
        self.avg = avg
        self.leaky_coefficient = leaky_coefficient

    def forward(self, x, x_lengths, num_graphs):
        batch_size = x.size(0)
        den_graphs = ChainGraphBatch(self.den_graph, batch_size)
        den_objf = ChainFunction.apply(x, x_lengths, den_graphs, self.leaky_coefficient)
        num_objf = ChainFunction.apply(x, x_lengths, num_graphs)
        objf = -(num_objf - den_objf)
        if self.avg:
            objf = objf / x_lengths.sum()
        return objf
