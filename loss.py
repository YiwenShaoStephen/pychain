import torch
import torch.nn as nn
import simplefst
import pychain


class ChainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, training_opts, den_graph, batch_size):
        input_grad = torch.zeros_like(input) # initialize it in python to save memory
                                             # when we add grad from numerator
        objf = pychain.compute_objf_and_deriv(
            training_opts, den_graph, batch_size, input, input_grad)
        ctx.save_for_backward(input_grad)
        return objf

    @staticmethod
    def backward(ctx, objf_grad):
        input_grad, = ctx.saved_tensors
        input_grad = torch.mul(input_grad, objf_grad)
        return input_grad, None, None, None


class ChainLoss(nn.Module):
    def __init__(self, den_fst_file, num_pdfs, cuda=False):
        super(ChainLoss, self).__init__()
        self.num_pdfs = num_pdfs
        self.den_fst = simplefst.StdVectorFst.read(den_fst_file)
        self.den_graph = pychain.DenominatorGraph(self.den_fst, num_pdfs, cuda)
        self.training_opts = pychain.ChainTrainingOptions()

    def forward(self, input):
        T, B, D = input.shape  # this is the nnet output shape from a RNN
        input = input.view(-1, D)  # (T * B, D)
        den_objf = ChainFunction.apply(
            input, self.training_opts, self.den_graph, B)
        return den_objf


if __name__ == "__main__":
    import torch
    import time
    D = 3237
    B = 10
    T = 100

    pychain.set_verbose_level(4)
    nnet_output = torch.zeros(T, B, D).cuda()
    nnet_output.requires_grad = True

    criterion = ChainLoss("./den.fst", D, True)

    start_time = time.time()
    obj = criterion(nnet_output)
    obj.backward()
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print(nnet_output.grad)
    print(obj)
