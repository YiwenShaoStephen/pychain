import torch
import torch.nn as nn
import simplefst
import pychain
import time


class DenominatorGraph(object):

    def __init__(self, fst):
        self.num_states = fst.num_states()
        (self.forward_transitions,
         self.forward_transition_probs,
         self.forward_transition_indices,
         self.backward_transitions,
         self.backward_transition_probs,
         self.backward_transition_indices) = simplefst.StdVectorFst.fst_to_tensor(
            fst)
        self.initial_probs = self.get_initial_probs(fst)

    def get_initial_probs(self, fst):
        initial_probs = simplefst.StdVectorFst.set_initial_probs(fst)
        return initial_probs


class ChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, den_graph):
        forward_transitions = den_graph.forward_transitions
        forward_transition_indices = den_graph.forward_transition_indices
        forward_transition_probs = den_graph.forward_transition_probs
        backward_transitions = den_graph.backward_transitions
        backward_transition_indices = den_graph.backward_transition_indices
        backward_transition_probs = den_graph.backward_transition_probs
        initial_probs = den_graph.initial_probs
        num_states = den_graph.num_states
        objf, input_grad, _ = pychain.forward_backward_den(
            forward_transitions,
            forward_transition_indices,
            forward_transition_probs,
            backward_transitions,
            backward_transition_indices,
            backward_transition_probs,
            initial_probs, input, num_states)
        ctx.save_for_backward(input_grad)
        return objf

    @staticmethod
    def backward(ctx, objf_grad):
        input_grad, = ctx.saved_tensors
        torch.mul(input_grad, objf_grad)
        return input_grad, None


class ChainLoss(nn.Module):
    def __init__(self, den_fst, avg=True):
        super(ChainLoss, self).__init__()
        self.den_fst = den_fst
        self.den_graph = DenominatorGraph(self.den_fst)
        self.avg = avg

    def forward(self, x):
        den_objf = ChainFunction.apply(
            x, self.den_graph)
        if self.avg:
            den_objf = den_objf / x.size(0)
        return den_objf


if __name__ == "__main__":
    torch.manual_seed(1)
    nnet_output = torch.rand(100, 300, 3237).cuda()
    nnet_output.requires_grad = True
    den_fst_file = "/export/a16/vmanoha1/pychain/tests/den.fst"
    den_fst = simplefst.StdVectorFst.read(den_fst_file)
    criterion = ChainLoss(den_fst)
    start_time = time.time()
    obj = criterion(nnet_output)
    obj.backward()
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print(obj)
