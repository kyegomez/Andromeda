import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from optimus_prime.x_transformers import XTransformer, Encoder, Decoder, CrossAttender, Attention, TransformerWrapper, ViTransformerWrapper, ContinuousTransformerWrapper
from optimus_prime.x_transformers import AndromedaEmbedding #, AndromedaBnBEmbedding

# d
from optimus_prime.autoregressive_wrapper import AutoregressiveWrapper
from optimus_prime.nonautoregressive_wrapper import NonAutoregressiveWrapper
from optimus_prime.continuous_autoregressive_wrapper import ContinuousAutoregressiveWrapper
from optimus_prime.xl_autoregressive_wrapper import XLAutoregressiveWrapper