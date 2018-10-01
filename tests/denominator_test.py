#!python

import torch
import simplefst
import denominator_graph

# num-pdfs (D) = 3237
# num-hmm-states (N) = 16602
# minibatch-size (B) = 128

den_fst = simplefst.StdVectorFst.read("den.fst")
den_graph = denominator_graph.DenominatorGraph(den_fst, 3237)

# Unroll transition indices N  x D -> ND.
transitions_roll_ND = torch.tensor(den_graph.transitions().index_select(0, torch.LongTensor([1]))) * 3237 + den_graph.transitions().index_select(0, torch.LongTensor([2]));

# Create the sparse transition matrix sp_trans of size N x ND
sp_trans_indices = torch.tensor(den_graph.transitions().index_select(0, torch.LongTensor([0, 1])))
sp_trans_indices[torch.LongTensor([1])] = transitions_roll_ND
sp_trans_probs = torch.tensor(den_graph.transition_probs())

if torch.cuda.is_available():
    sp_trans_indices = sp_trans_indices.cuda()
    sp_trans_probs = sp_trans_probs.cuda()

sp_trans = torch.sparse_coo_tensor(sp_trans_indices, sp_trans_probs)

# Create alpha_0 of size N x B
alpha_0 = den_graph.initial_probs().unsqueeze(0).expand([128, den_graph.initial_probs().size(0)]).transpose(0, 1)
if torch.cuda.is_available():
    alpha_0 = alpha_0.cuda()

# Multiply transition_matrix and alpha_0
# tmp is a SparseTensor of size ND x B with K non-zero rows.
# i.e. tmp._indices() is a 1 x K Tensor
# tmp._values() is a K x B Tensor
tmp = torch.hspmm(sp_trans.transpose(0, 1), alpha_0)

# Roll indices of tmp from ND to N x D.
tmp_rolled_indices = torch.zeros([2, tmp._indices().size(1)], dtype=torch.long)
if torch.cuda.is_available():
    tmp_rolled_indices = tmp_rolled_indices.cuda()

tmp_rolled_indices[0] = tmp._indices() / 3237   # destination-states
tmp_rolled_indices[1] = torch.remainder(tmp._indices(), 3237)  # pdf-ids
tmp_rolled = torch.sparse_coo_tensor(tmp_rolled_indices, tmp._values())

# nnet_outputs at time t. We can do exp beforehand. size D x B
nnet_outputs = torch.randn([3237, 128])
if torch.cuda.is_available():
    nnet_outputs = nnet_outputs.cuda()
nnet_outputs.exp_()

# Lookup indices of nnet_outputs based on pdf-ids. size K x B
nnet_outputs_lookup = nnet_outputs.index_select(0, tmp_rolled_indices[1])

# Element-wise product with the nnet_outputs for the K rows
# Output is K x B.
tmp2 = torch.mul(tmp._values(), nnet_outputs_lookup)

# Create alpha_t Tensor of size N x B
alpha_t = torch.zeros([16602, 128])
if torch.cuda.is_available():
    alpha_t = alpha_t.cuda()
alpha_t.index_add_(0, tmp_rolled_indices[0], tmp2)

print (alpha_t)
