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
import simplefst


class ChainGraph(object):

    def __init__(
        self, fst=None, transitions=None, transition_probs=None, num_states=None,
        final_probs=None, initial='simple', leaky_hmm_coefficient=1.0e-05, is_denominator=True,
    ):
        if fst:
            self.num_states = fst.num_states()
            if initial == 'simple':
                self.initial_probs = self.simple_initial_probs()
            elif initial == 'recursive':
                self.initial_probs = self.recursive_initial_probs(fst)
            else:
                raise ValueError(
                    'only simple or recursive is valid for initial')

            (self.forward_transitions,
             self.forward_transition_probs,
             self.forward_transition_indices,
             self.backward_transitions,
             self.backward_transition_probs,
             self.backward_transition_indices,
             self.final_probs) = simplefst.StdVectorFst.fst_to_tensor(fst)
            if is_denominator:  # set final-probs to ones
                self.final_probs = torch.ones(self.num_states, dtype=self.initial_probs.dtype)

        elif not (transitions is None and transition_probs is None and num_states is None):
            assert(transitions.size(0) == transition_probs.size(0))
            self.num_states = num_states
            self.initial_probs = self.simple_initial_probs
            self.final_probs = final_probs
            assert self.final_probs.size(0) == num_states

            (self.forward_transitions,
             self.forward_transition_probs,
             self.forward_transition_indices) = self.get_sorted_transitions(transitions,
                                                                            transition_probs,
                                                                            'forward')

            (self.backward_transitions,
             self.backward_transition_probs,
             self.backward_transition_indices) = self.get_sorted_transitions(transitions,
                                                                             transition_probs,
                                                                             'backward')

        else:
            raise ValueError('either a FST object or (transitions, transition_probs and num_states)'
                             'should be provided to initialize a ChainGraph')

        self.num_transitions = self.forward_transitions.size(0)
        self.leaky_hmm_coefficient = leaky_hmm_coefficient

    def simple_initial_probs(self):
        initial_probs = torch.zeros(self.num_states)
        initial_probs[0] = 1
        return initial_probs

    def recursive_initial_probs(self, fst):
        initial_probs = simplefst.StdVectorFst.set_initial_probs(fst)
        return initial_probs

    def get_sorted_transitions(self, transitions, transition_probs, mode='forward'):
        if mode == 'forward':
            col = 0
        elif mode == 'backward':
            col = 1
        else:
            raise ValueError('Only forward or backward is valid as mode param, but given {}'
                             .format(mode))

        order = transitions[:, col].argsort()
        sorted_transitions = transitions[order]
        sorted_transition_probs = transition_probs[order]
        transition_indices = self.get_transition_indices(
            sorted_transitions[:, col])
        return sorted_transitions, sorted_transition_probs, transition_indices

    def get_transition_indices(self, sorted_states):
        end_point = (sorted_states[:-1] -
                     sorted_states[1:]).nonzero().squeeze() + 1
        preffix = torch.zeros(1, dtype=end_point.dtype)
        suffix = torch.ones(1, dtype=end_point.dtype) * sorted_states.size(0)
        start = torch.cat((preffix, end_point), 0)
        end = torch.cat((end_point, suffix), 0)
        start_end = torch.cat((start.unsqueeze(1), end.unsqueeze(1)), 1)
        indices = torch.zeros(self.num_states, 2, dtype=start_end.dtype)
        states_nonzero = sorted_states[start]
        indices[states_nonzero.long()] = start_end
        return indices


class ChainGraphBatch(object):
    def __init__(self, graphs, batch_size=None, max_num_transitions=None, max_num_states=None):
        if isinstance(graphs, ChainGraph):
            if not batch_size:
                raise ValueError(
                    "batch size should be specified to expand a single graph")
            self.batch_size = batch_size
            self.leaky_hmm_coefficient = graphs.leaky_hmm_coefficient
            self.initialized_by_one(graphs)
        elif isinstance(graphs, (list, ChainGraph)):
            if not max_num_transitions:
                raise ValueError("max_num_transitions should be specified if given a "
                                 "a list of ChainGraph objects to initialize from")
            if not max_num_states:
                raise ValueError("max_num_states should be specified if given a "
                                 "a list of ChainGraph objects to initialize from")
            self.batch_size = len(graphs)
            self.leaky_hmm_coefficient = graphs[0].leaky_hmm_coefficient
            self.initialized_by_list(
                graphs, max_num_transitions, max_num_states)
        else:
            raise ValueError("ChainGraphBatch should be either initialized by a "
                             "single ChainGraph object or a list of ChainGraph objects "
                             "but given {}".format(type(graphs)))

    def initialized_by_one(self, graph):
        B = self.batch_size
        self.forward_transitions = graph.forward_transitions.repeat(
            B, 1, 1)
        self.forward_transition_indices = graph.forward_transition_indices.repeat(
            B, 1, 1)
        self.forward_transition_probs = graph.forward_transition_probs.repeat(
            B, 1)
        self.backward_transitions = graph.backward_transitions.repeat(
            B, 1, 1)
        self.backward_transition_indices = graph.backward_transition_indices.repeat(
            B, 1, 1)
        self.backward_transition_probs = graph.backward_transition_probs.repeat(
            B, 1)
        self.initial_probs = graph.initial_probs
        self.num_states = graph.num_states
        self.final_probs = graph.final_probs.repeat(B, 1)

    def initialized_by_list(self, graphs, max_num_transitions, max_num_states):
        transition_type = graphs[0].forward_transitions.dtype
        probs_type = graphs[0].forward_transition_probs.dtype
        self.num_states = max_num_states
        self.num_transitions = max_num_transitions
        self.initial_probs = self.simple_initial_probs()
        self.forward_transitions = torch.zeros(
            self.batch_size, max_num_transitions, 3, dtype=transition_type)
        self.forward_transition_indices = torch.zeros(
            self.batch_size, max_num_states, 2, dtype=transition_type)
        self.forward_transition_probs = torch.zeros(
            self.batch_size, max_num_transitions, dtype=probs_type)
        self.backward_transitions = torch.zeros(
            self.batch_size, max_num_transitions, 3, dtype=transition_type)
        self.backward_transition_indices = torch.zeros(
            self.batch_size, max_num_states, 2, dtype=transition_type)
        self.backward_transition_probs = torch.zeros(
            self.batch_size, max_num_transitions, dtype=probs_type)
        self.final_probs = torch.zeros(
            self.batch_size, max_num_states, dtype=probs_type,
        )

        for i in range(len(graphs)):
            graph = graphs[i]
            num_transitions = graph.num_transitions
            num_states = graph.num_states
            self.forward_transitions[i, :num_transitions, :].copy_(
                graph.forward_transitions)
            self.forward_transition_indices[i, :num_states, :].copy_(
                graph.forward_transition_indices)
            self.forward_transition_probs[i, :num_transitions].copy_(
                graph.forward_transition_probs)
            self.backward_transitions[i, :num_transitions, :].copy_(
                graph.backward_transitions)
            self.backward_transition_indices[i, :num_states, :].copy_(
                graph.backward_transition_indices)
            self.backward_transition_probs[i, :num_transitions].copy_(
                graph.backward_transition_probs)
            self.final_probs[i, :num_states].copy_(graph.final_probs)

    def simple_initial_probs(self):
        initial_probs = torch.zeros(self.num_states)
        initial_probs[0] = 1
        return initial_probs
