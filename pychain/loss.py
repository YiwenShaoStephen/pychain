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

import logging
import torch
import torch.nn as nn
from pychain.graph import ChainGraphBatch
import pychain_C


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChainLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_lengths,
        num_graphs,
        den_graph,
        num_leaky_coefficient,
        den_leaky_coefficient
    ):
        exp_input = input.clamp(-30, 30).exp()
        B = input.size(0)
        T = input.size(1)
        if B != num_graphs.batch_size:
            raise ValueError("input batch size {0} does not equal to graph batch size {1}"
                             .format(B, num_graphs.batch_size))
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_lengths, batch_first=True)
        batch_sizes = packed_data.batch_sizes
        input_lengths = input_lengths.cpu()

        den_graphs = ChainGraphBatch(den_graph, B)

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
            den_leaky_coefficient,
        )

        den_mask = input_grad.argmax(axis=2)

        index_to_pdf = None
        if num_graphs.index_to_pdf is not None:
            index_to_pdf = num_graphs.index_to_pdf.to(device=input.device)

        if num_graphs.log_domain:
            if index_to_pdf is not None:
                # index_to_pdf B x U -> B x 1 x U -> B x T x U
                log_probs = torch.gather(
                    input,
                    2,
                    index_to_pdf.unsqueeze(1).expand((-1, T, -1)),
                )  # B x T x U

            num_objf, log_probs_grad, numerator_ok = pychain_C.numerator_fb(
                num_graphs.forward_transitions,
                num_graphs.forward_transition_indices,
                num_graphs.forward_transition_probs,
                num_graphs.backward_transitions,
                num_graphs.backward_transition_indices,
                num_graphs.backward_transition_probs,
                num_graphs.final_probs,
                num_graphs.start_state,
                log_probs.to(device='cpu')
                if index_to_pdf is not None
                else input.to(device='cpu'),
                batch_sizes,
                input_lengths,
                num_graphs.num_states,
            )
            num_objf = num_objf.to(device=input.device)
        else:
            num_objf, log_probs_grad, numerator_ok = pychain_C.forward_backward(
                num_graphs.forward_transitions,
                num_graphs.forward_transition_indices,
                num_graphs.forward_transition_probs,
                num_graphs.backward_transitions,
                num_graphs.backward_transition_indices,
                num_graphs.backward_transition_probs,
                num_graphs.leaky_probs,
                num_graphs.initial_probs,
                num_graphs.final_probs,
                num_graphs.start_state,
                exp_input,
                batch_sizes,
                input_lengths,
                num_graphs.num_states,
                num_leaky_coefficient,
            )

        numerator_ok = numerator_ok.item()
        denominator_ok = denominator_ok.item()

        loss = -num_objf + den_objf

        default_loss = 10
        if ((loss - loss) != 0 or not denominator_ok or not numerator_ok):
            input_grad = torch.zeros_like(input)
            logger.warn(
                f"Loss is {loss}) and denominator computation "
                f"(if done) returned {denominator_ok} "
                f"and numerator computation returned {numerator_ok} "
                f", setting loss to {default_loss} per frame"
            )
            loss = torch.zeros_like(num_objf).fill_(
                default_loss * input_lengths.sum()
            )
            frame_match = torch.zeros(B, T, device=input.device, dtype=torch.bool)
        else:
            if num_graphs.log_domain:
                if index_to_pdf is not None:
                    num_grad = log_probs_grad.to(device=input.device).exp()

                    # index_to_pdf B x U -> B x 1 x U -> B x T x U
                    input_grad.scatter_add_(
                        2,
                        index_to_pdf.unsqueeze(1).expand_as(num_grad),
                        -num_grad,
                    )
                    num_mask = torch.gather(
                        index_to_pdf,  # B x U
                        1,
                        num_grad.argmax(axis=2),  # B x T
                    )  # B x T
                else:
                    num_grad = log_probs_grad.to(device=input.device).exp()
                    input_grad -= num_grad
                    num_mask = num_grad.argmax(axis=2)
            else:
                num_grad = log_probs_grad
                input_grad -= num_grad
                num_mask = num_grad.argmax(axis=2)

            frame_match = (num_mask == den_mask)

        ctx.save_for_backward(input_grad)
        return loss, input_grad, frame_match

    @staticmethod
    def backward(ctx, loss_grad, input_grad_grad=None, frame_match_grad=None):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, loss_grad)

        return input_grad, None, None, None, None, None


class ChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_lengths, graphs, leaky_coefficient):
        exp_input = input.clamp(-30, 30).exp()
        B = input.size(0)
        T = input.size(1)
        if B != graphs.batch_size:
            raise ValueError("input batch size {0} does not equal to graph batch size {1}"
                             .format(B, graphs.batch_size))
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_lengths, batch_first=True)
        batch_sizes = packed_data.batch_sizes
        input_lengths = input_lengths.cpu()

        index_to_pdf = None
        if graphs.index_to_pdf is not None:
            index_to_pdf = graphs.index_to_pdf.to(device=input.device)

        if not graphs.log_domain:
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
                leaky_coefficient)
        else:
            if index_to_pdf is not None:
                # index_to_pdf B x U -> B x 1 x U -> B x T x U
                log_probs = torch.gather(
                    input,
                    2,
                    index_to_pdf.unsqueeze(1).expand((-1, T, -1)),
                )  # B x T x U
            objf, log_probs_grad, ok = pychain_pytorch_binding.numerator_fb(
                graphs.forward_transitions,
                graphs.forward_transition_indices,
                graphs.forward_transition_probs,
                graphs.backward_transitions,
                graphs.backward_transition_indices,
                graphs.backward_transition_probs,
                graphs.final_probs,
                graphs.start_state,
                log_probs.to(device='cpu')
                if index_to_pdf is not None
                else input.to(device='cpu'),
                batch_sizes,
                input_lengths,
                graphs.num_states,
            )
            objf = objf.to(device=input.device)

            if index_to_pdf is not None:
                input_grad = torch.zeros_like(input)
                # index_to_pdf B x U -> B x 1 x U -> B x T x U
                input_grad.scatter_add_(
                    2,
                    index_to_pdf.unsqueeze(1).expand_as(log_probs_grad),
                    log_probs_grad.to(device=input.device).exp(),
                )
            else:
                input_grad = log_probs_grad.to(device=input.device).exp()

        ctx.save_for_backward(input_grad)
        return objf, input_grad

    @staticmethod
    def backward(ctx, objf_grad, input_grad_grad=None):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, objf_grad)

        return input_grad, None, None, None
