import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)



def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2**math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = (1. / torch.pow(2, m))

    if _n_heads != n_heads:
        # if n_heads is not a power of two,
        # Huggingface and FasterTransformer calculate slopes normally,
        # then return this strided concatenation of slopes
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]

    return slopes.view(1, n_heads, 1, 1)



def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim -1) if dim < 0 else (t.ndim - dim -1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)

class RelativePositionBias(nn.Module):
    def __init__(
        self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=12
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, step=None):
        step = 0 if step is None else step
        context_position = torch.arange(
            step,
            step + qlen,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device,
        )[:, None]
        memory_position = torch.arange(
            klen, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)

        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, batch_size, qlen, klen, step=None):
        # shape (batch * num_heads, qlen, klen)
        return (
            self.compute_bias(qlen, klen, step)
            .repeat(batch_size, 1, 1, 1)
            .view(-1, qlen, klen)
        )



class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


from math import pi, log

import torch
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1.):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.use_xpos = use_xpos
        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.register_buffer('scale', scale)

    def rotate_queries_or_keys(self, t, seq_dim = -2):
        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
        device, seq_len = t.device, t.shape[seq_dim]
        freqs = self.forward(lambda: torch.arange(seq_len, device = device), cache_key = seq_len)
        return apply_rotary_emb(freqs, t)

    def rotate_queries_and_keys(self, q, k, seq_dim = -2):
        assert self.use_xpos
        device, seq_len = q.device, q.shape[seq_dim]
        seq = torch.arange(seq_len, device = device)
        freqs = self.forward(lambda: seq, cache_key = f'freqs:{seq_len}')
        scale = self.get_scale(lambda: seq, cache_key = f'scale:{seq_len}')
        rotated_q = apply_rotary_emb(freqs, q, scale = scale)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1)
        return rotated_q, rotated_k

    def get_scale(self, t, cache_key = None):
        assert self.use_xpos

        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        scale = 1.
        if self.use_xpos:
            power = t - (len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')

"""
My gut also tells me there is something more to rotations that can be exploited in artificial neural networks.

Install
$ pip install rotary-embedding-torch
Usage
import torch
from rotary_embedding_torch import RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

rotary_emb = RotaryEmbedding(dim = 32)

# mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
k = torch.randn(1, 8, 1024, 64) # keys

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

q = rotary_emb.rotate_queries_or_keys(q)
k = rotary_emb.rotate_queries_or_keys(k)

# then do your attention with your queries (q) and keys (k) as usual
If you do all the steps above correctly, you should see a dramatic improvement during training

Axial Rotary Embeddings
For easy use of 2d axial relative positional embedding, ie. vision transformers

import torch
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat

pos_emb = RotaryEmbedding(
    dim = 32,
    freqs_for = 'pixel',
    max_freq = 256
)

# queries and keys for frequencies to be rotated into

q = torch.randn(1, 256, 256, 64)
k = torch.randn(1, 256, 256, 64)

# get frequencies for each axial
# -1 to 1 has been shown to be a good choice for images and audio

freqs_h = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)
freqs_w = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)

# concat the frequencies along each axial
# broadcat function makes this easy without a bunch of expands

freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

# rotate in frequencies

q = apply_rotary_emb(freqs, q)
k = apply_rotary_emb(freqs, k)
Length Extrapolatable Rotary Embeddings
In this paper, they were able to fix length extrapolation issue with rotary embeddings by giving it a decay similar to ALiBi. They named this technique XPos, and you can use it by setting use_xpos = True on initialization.

This can only be used for autoregressive transformers

import torch
from rotary_embedding_torch import RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

rotary_emb = RotaryEmbedding(
    dim = 32,
    use_xpos = True   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
)

# mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
k = torch.randn(1, 8, 1024, 64) # keys

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

# instead of using `rotate_queries_or_keys`, you will use `rotate_queries_and_keys`, the rest is taken care of

q, k = rotary_emb.rotate_queries_and_keys(q, k)

"""



"""
Update (12/2022): Rotary embedding has since been hugely successful, widely adopted in many large language models, including the largest in the world, PaLM. However, it has been uncovered in the ALiBi paper that rotary embeddings cannot length extrapolate well. This was recently addressed in a Microsoft research paper. They propose a way to unobtrusively add the same decay as in ALiBi, and found that this resolves the extrapolation problem. You can use it in this repository by setting rotary_xpos = True. Like ALiBi, it would enforce the attention to be local. You can set the receptive field with rotary_xpos_scale_base value, which defaults to 512




"""


"""

ALiBi Positional Embedding
This paper proposes to simply apply a static linear bias to the attention matrix. The authors show this is not only effective as a relative positional encoding, but also allows the attention net to extrapolate to greater sequences length than what it was trained on, for autoregressive language models.

This repository also offers a bidirectional variant (nonsymmetric), proposed by the authors here. However, this is untested. If you need bidirectional length extrapolation, the safest option would be Dynamic Position Bias

Update: It may be that ALiBi enforces a strong local attention across the heads, and may hinder it from attending at distances greater than 1k. To avoid any issues with global message passing, I've decided to introduce another hyperparameter alibi_num_heads, so one can specify less heads for the ALiBi bias

Update: There are reports that ALiBi outperform Rotary embeddings for pretraining and downstream fine-tuning.

import torch
from x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        alibi_pos_bias = True, # turns on ALiBi positional embedding
        alibi_num_heads = 4    # only use ALiBi for 4 out of the 8 heads, so other 4 heads can still attend far distances
    )
)

"""


class AlibiPositionalBias(nn.Module):

    """
    The AlibiPositionalBias class is a module for adding a positional bias to attention scores in a transformer model. The purpose of this bias is to help the model understand relative positions of tokens when attending to them, potentially improving the performance of the attention mechanism in capturing long-range dependencies and contextual information.

    Let's go through the class step by step:

    __init__ method: It initializes the class by setting the number of heads and creating slopes based on the _get_slopes method. It registers slopes and bias as non-persistent buffers.

    get_bias method: This method takes the length of query and key sequences (i and j respectively) and creates a bias tensor with shape (1, i, j). The bias tensor represents the absolute differences between the query and key positions.

    _get_slopes method: This method generates slopes based on the number of heads. The slopes are determined using a power-of-2-based method, which helps distribute the slopes evenly.

    forward method: In the forward method, the attention scores (qk_dots) are passed as input. The method checks if the existing bias tensor is compatible with the current attention scores shape. If it is, it uses the existing bias tensor, otherwise, it calculates a new bias tensor using the get_bias method and updates the registered buffer. The bias tensor is then multiplied by the slopes and added to the attention scores.

    The LearnedAlibiPositionalBias class is a subclass of AlibiPositionalBias that allows learning the slopes during the training process. It has two main differences:

    In the __init__ method, it initializes the learned_logslopes as a learnable parameter by taking the logarithm of the initial slopes.
    In the forward method, it retrieves the learned slopes by exponentiating the learned_logslopes. The rest of the process is the same as in the parent class, with the only difference being that the learned slopes are used instead of the fixed slopes from the parent class.
    By learning the slopes, the model can potentially adapt to the specific positional biases present in the data, which may lead to better performance in capturing long-range dependencies and contextual information.
    
    """
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent=False)

        return qk_dots + self.bias

class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads):
        super().__init__(heads)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        def get_slopes(param):
            return pad_at_dim(param.exp(), (0, h - param.shape[0]), dim = -2)

        if exists(self.bias) and self.bias.shape[-1] >= j:
            bias = self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer('bias', bias, persistent=False)

        slopes = get_slopes(self.learned_logslopes)
        bias = bias * slopes

        return qk_dots + bias

#v2 
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512
    ):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale


def build_alibi_bias(
    n_heads,
    seq_len,
    full=False,
    alibi_bias_max=8,
    device=None,
    dtype=None,
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32,
                              device=device).view(1, 1, 1, seq_len)
    if full:
        # generate 1 x Heads x SeqLen x SeqLen alibi bias mask
        # otherwise the mask is 1 x Heads x 1 x SeqLen (which is broadcast to the appropriate size)
        alibi_bias = alibi_bias - torch.arange(
            1 - seq_len, 1, dtype=torch.int32, device=device).view(
                1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)




def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

# def apply_rotary_pos_emb(t, freqs, scale = 1):
#     seq_len = t.shape[-2]
#     freqs = freqs[-seq_len:, :]
#     return (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

# # norms

