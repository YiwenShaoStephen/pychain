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

    def __init__(self, fst, leaky_mode='uniform'):
        self.num_states = fst.num_states()
        assert(leaky_mode in ['uniform', 'recursive'])
        self.leaky_mode = leaky_mode
        (self.forward_transitions,
         self.forward_transition_probs,
         self.forward_transition_indices,
         self.backward_transitions,
         self.backward_transition_probs,
         self.backward_transition_indices,
         self.final_probs) = simplefst.StdVectorFst.fst_to_tensor(fst)

        if leaky_mode == 'recursive':
            self.leaky_probs = self.recursive_leaky_probs(fst)
        else:
            self.leaky_probs = torch.ones(self.num_states) / self.num_states

        self.num_transitions = self.forward_transitions.size(0)

    def recursive_leaky_probs(self, fst):
        leaky_probs = simplefst.StdVectorFst.set_leaky_probs(fst)
        return leaky_probs


class ChainGraphBatch(object):
    def __init__(self, graphs, batch_size=None, max_num_transitions=None, max_num_states=None):
        if isinstance(graphs, ChainGraph):
            if not batch_size:
                raise ValueError(
                    "batch size should be specified to expand a single graph")
            self.batch_size = batch_size
            self.initialized_by_one(graphs)
        elif isinstance(graphs, (list, ChainGraph)):
            if not max_num_transitions:
                raise ValueError("max_num_transitions should be specified if given a "
                                 "a list of ChainGraph objects to initialize from")
            if not max_num_states:
                raise ValueError("max_num_states should be specified if given a "
                                 "a list of ChainGraph objects to initialize from")
            self.batch_size = len(graphs)
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
        self.num_states = graph.num_states
        self.final_probs = graph.final_probs.repeat(B, 1)
        self.leaky_probs = graph.leaky_probs.repeat(B, 1)

    def initialized_by_list(self, graphs, max_num_transitions, max_num_states):
        transition_type = graphs[0].forward_transitions.dtype
        probs_type = graphs[0].forward_transition_probs.dtype
        self.num_states = max_num_states
        self.num_transitions = max_num_transitions
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
        self.leaky_probs = torch.zeros(
            self.batch_size, max_num_states, dtype=probs_type)
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
            self.leaky_probs[i, :num_states].copy_(graph.leaky_probs)
            self.final_probs[i, :num_states].copy_(graph.final_probs)
