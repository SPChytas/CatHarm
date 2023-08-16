import warnings
import torch
import torch.nn as nn
import sys
from typing import Optional, Tuple


class AttentionWithoutShortcutsLayer(nn.Module):
    r"""
    Multi-Head Attention Without Any Shortcuts.
        Can be used by directly replacing the conditional Attention Layers
        Support both Multi-head Self-Attention and Multi-head Cross-Attention.
    <p>
    Input sequence X [batch_size, T, d]
    Output Attn(X) [batch_size, T, d]
    """
    # current_layer = 1  # used as a static variable to count the current layer
    current_layer = 1

    def __init__(self,
                 sequence_length: int,
                 embedding_dim: int,
                 num_heads: int,
                 max_layers: int,  # warning: must be consecutive attention layers
                 Gamma: int = 1e3,
                 is_cross_attention: bool = False,
                 dim_q: int = 0,
                 dim_k: int = 0,
                 dim_v: int = 0,
                 gamma_L: float = 0.005,
                 gamma_0: int = sys.maxsize,
                 qkv_equal_dim: bool = True,
                 upper_triangle: bool = False):
        r"""
        Constructor for AttentionWithoutShortcutsLayer Objects.
        Support both Multi-head Self-Attention and Multi-head Cross-Attention.

        :param sequence_length: length T of the input sequence
        :param embedding_dim: embedding dimension of the input sequence
        :param num_heads: number of heads
        :param max_layers: the total number of consecutive attention layers in the MHA block
        :param Gamma: a Large Positive Constant
        :param is_cross_attention: whether this layer is a multi-head cross-attention layer. Default: False
        :param dim_q: dimension of the queries. Default: embedding_dim
        :param dim_k: dimension of the keys. Default: embedding_dim
        :param dim_v: dimension of the values. Default: embedding_dim
        :param gamma_L: a hyperparameter. Default: 0.005
        :param gamma_0: +inf
        :param qkv_equal_dim: whether the dimensions of q, k, v are equal. Default: True
        :param upper_triangle: we need the upper_triangle of the Cholesky Decomposition
        """
        super().__init__()
        warnings.filterwarnings("ignore")

        assert dim_q == dim_k, "[ERROR] Queries and Keys must have same dimension."
        assert AttentionWithoutShortcutsLayer.current_layer <= max_layers, "[ERROR]Current Layer are out of the Max_Layer."
        assert AttentionWithoutShortcutsLayer.current_layer > 0, "[ERROR]Current Layer Invalid."
        assert gamma_0 >= 1e4, "[ERROR] Initial gamma_0 has problem."
        assert embedding_dim % num_heads == 0, "[ERROR]Embedding dim should be evenly divided by the number of heads"
        assert Gamma >= 1e2, "[INFO]Gamma should be a large positive constant (i.g. > 1e2)."

        if qkv_equal_dim:
            dim_q = dim_k = dim_v = int(embedding_dim / num_heads)
        else:
            dim_q = dim_k = int(embedding_dim / num_heads)
            dim_v = dim_v

        self.gamma_L = gamma_L
        self.max_layers = max_layers
        self.alpha_L = (
            1 - torch.exp(torch.tensor(-2 * gamma_L)).to("cuda")) ** .5
        self.sequence_length = sequence_length
        self.current_layer = AttentionWithoutShortcutsLayer.current_layer
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.lower_triangle_mask = self.generate_causal_mask()
        self.Gamma = Gamma
        self.is_cross_attention = is_cross_attention
        self.W_Q = nn.Linear(in_features=embedding_dim,
                             out_features=dim_q, bias=True)  # (dim, d_q)
        self.W_K = nn.Linear(in_features=embedding_dim,
                             out_features=dim_k, bias=True)  # (dim, d_k)
        self.W_V = nn.Linear(in_features=embedding_dim,
                             out_features=dim_v, bias=True)  # (dim, d_v)
        self.W_O = nn.Linear(in_features=dim_v * num_heads,
                             out_features=embedding_dim, bias=True)  # (d_v * h, dim)

        print(f"Cross Attention Enabled {self.is_cross_attention}")

        # get L_in and L_out
        if self.current_layer == 1:
            L_in_matrix = self.eigenmatrix_generator(gamma_0)
        else:
            L_in_matrix = self.eigenmatrix_generator(
                self.calculate_gamma_l(int(self.current_layer - 1)))

        L_out_matrix = self.eigenmatrix_generator(
            self.calculate_gamma_l(self.current_layer))

        L_in_after_cholesky = torch.cholesky(
            input=L_in_matrix, upper=upper_triangle)
        L_out_after_cholesky = torch.cholesky(
            input=L_out_matrix, upper=upper_triangle)

        # then get A = L_{out} @ L_{in}^{-1}
        # warning: do not use torch.linalg
        L_in_after_cholesky_inverse = torch.inverse(L_in_after_cholesky)

        A = L_out_after_cholesky @ L_in_after_cholesky_inverse

        self.D, self.P = self.decompose_to_D_P(A)
        self.B = torch.log(self.P).to("cuda")
        self.D = self.D.to("cuda")
        self._reset_parameters()

        AttentionWithoutShortcutsLayer.current_layer += 1

    def _reset_parameters(self):
        # for every head, we need to re-initialize the parameters
        # 1. Let W_K and W_V be orthogonal
        self.W_K.weight = nn.Parameter(
            nn.init.orthogonal(torch.empty_like(self.W_K.weight)))
        self.W_V.weight = nn.Parameter(
            nn.init.orthogonal(torch.empty_like(self.W_V.weight)))

        # 2. Let W_K and W_V follow a Normal Distribution with \mu = 0.0, std = 1 / embedding_dim
        self.W_K.weight = nn.Parameter(
            self.W_K.weight.data.clone().mul(1 / self.embedding_dim ** .5))
        self.W_V.weight = nn.Parameter(
            self.W_V.weight.data.clone().mul(1 / self.embedding_dim ** .5))

        # 3. Let W_Q = 0 so that multi-head attention can keep full rank at the initial step
        self.W_Q.weight = nn.Parameter(nn.init.constant(
            torch.empty_like(self.W_Q.weight), 0))

    def forward(self,
                query_1: torch.Tensor,
                key_1: torch.Tensor,
                value_1: torch.Tensor,
                query_2: torch.Tensor = None,
                key_2: torch.Tensor = None,
                value_2: torch.Tensor = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = True) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Multi-Head Attention Layer forward pass.


        :param query_1: Queries from the first sequence (Input sequence 1)
        :param key_1: Keys from the first sequence (Input sequence 1)
        :param value_1: Values from the first sequence (Input sequence 1)
        :param query_2: Queries from the second sequence (Input sequence 2)
        :param key_2: Keys from the second sequence (Input sequence 2)
        :param value_2: Queries from the second sequence (Input sequence 2)
        :param key_padding_mask: key_padding_mask
        :param need_weights: whether attention weights needed. Default: False
        :param attention_mask: attention mask on the attention matrix

        :return: Output_sequence, None                        if need_weights == False
                 Output_sequence, A dictionary with weights   if need_weight == True
        """
        if query_2 is None:
            head_dim = self.embedding_dim / self.num_heads
            attention_coefficient = 1 / head_dim ** .5

            assert self.is_cross_attention == False, "[ERROR]We are using Cross Attention!"
            assert query_1.dim() == 3, "Data input should be batched"
            assert query_1.shape[-1] == self.embedding_dim, "Last dimension in the sequence data must be embedding dimension."
            assert query_1.shape[-2] == self.sequence_length, "Last second dimension must be sequence length."
            assert self.lower_triangle_mask.shape[
                -1] == self.sequence_length, "Mask's shape must euqal to [T, T], where T stands for Sequence Length"

            last_dim = query_1.dim() - 1

            if key_padding_mask is not None:
                assert key_padding_mask.dtype == torch.bool and torch.is_floating_point(
                    key_padding_mask), "Only float and bool types are supported for key_padding_mask"

            cache_weights = {}
            attn_for_heads = []
            for h in range(self.num_heads):
                self._reset_parameters()  # 1. Initialize W_Q, W_K, W_V

                # (T, d_q)  2. Calculate Q, K, V in each single head
                Query = self.W_Q(query_1)
                Key = self.W_K(key_1)  # (T, d_k)
                if key_padding_mask:
                    Key = Key * key_padding_mask
                Value = self.W_V(value_1)  # (T, d_v)

                P_h = torch.softmax((self.lower_triangle_mask * (
                    attention_coefficient * Query @ Key.transpose(last_dim - 1, last_dim)) - self.Gamma * (
                    torch.ones(self.sequence_length,
                               self.sequence_length).to(Query.device) - self.lower_triangle_mask)),
                    dim=last_dim)  # [T, T]

                # [T, dim_v] = [T, T][T, T][T, dim_v]
                attn = self.D @ P_h @ Value

                if attention_mask:
                    attn = attn * attention_mask

                cache_weights[
                    f"Layer {self.current_layer} Attention Enabled: {self.is_cross_attention} head: {h} W_Q"] = self.W_Q.weight
                cache_weights[
                    f"Layer {self.current_layer} Attention Enabled: {self.is_cross_attention} head: {h} W_K"] = self.W_K.weight
                cache_weights[
                    f"Layer {self.current_layer} Attention Enabled: {self.is_cross_attention} head: {h} W_V"] = self.W_V.weight

                attn_for_heads.append(attn)

            assert len(
                attn_for_heads) == self.num_heads, "Number of heads mismatches number of attention matrices."

            concatenated_attns = torch.cat(attn_for_heads, dim=last_dim)

            # then we need to initialize and normalize W_0 to \mu = 0 and std = 1 / dim
            self.W_O.weight = nn.Parameter(
                nn.init.orthogonal(torch.empty_like(self.W_O.weight)))
            self.W_O.weight = nn.Parameter(
                self.W_O.weight.data.clone().mul(1 / self.embedding_dim ** .5))

            if use_cache:
                present_key_value = (Key, Value)
            else:
                present_key_value = None

            if need_weights:
                return self.W_O(concatenated_attns), cache_weights, present_key_value
            else:
                return self.W_O(concatenated_attns), None, present_key_value

        assert self.is_cross_attention, "[ERROR]Cross Attention Enabled. The second sequence is needed."
        head_dim = self.embedding_dim / self.num_heads
        attention_coefficient = 1 / head_dim ** .5

        assert query_1.dim() == 3, "[ERROR]First Sequence Input is not batched"
        assert query_2.dim(
        ) == 3, "[ERROR]Second Sequence Input is not batched"
        assert query_1.shape[
            -1] == self.embedding_dim, "[ERROR]Last dimension in the sequence data must be embedding dimension.(First Sequence Input)"
        assert query_1.shape[
            -2] == self.sequence_length, "[ERROR]Last second dimension must be sequence length.(First Sequence Input)"
        assert query_2.shape[
            -1] == self.embedding_dim, "[ERROR]Last dimension in the sequence data must be embedding dimension.(second Sequence Input)"
        assert query_2.shape[
            -2] == self.sequence_length, "[ERROR]Last second dimension must be sequence length.(Second Sequence Input)"

        assert self.lower_triangle_mask.shape[
            -1] == self.sequence_length, "[ERROR]Mask's shape must euqal to [T, T], where T stands for Sequence Length"
        assert query_1.shape[-1] == query_2.shape[-1], "[ERROR]Two input sequence must have same embedding dimension."

        last_dim = query_1.dim() - 1

        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool and torch.is_floating_point(
                key_padding_mask), "Only float and bool types are supported for key_padding_mask"

        cache_weights = {}
        attn_for_heads = []
        for h in range(self.num_heads):
            # self._reset_parameters()  # 1. Initialize W_Q, W_K, W_V

            # (T, d_q)  2. Calculate Q, K, V in each single head
            Query = self.W_Q(query_1).requires_grad_()
            Key = self.W_K(key_2).requires_grad_()  # (T, d_k)
            if key_padding_mask:
                Key = Key * key_padding_mask
            Value = self.W_V(value_2).requires_grad_()  # (T, d_v)

            P_h = torch.softmax((self.lower_triangle_mask * (
                attention_coefficient * Query @ Key.transpose(last_dim - 1, last_dim)) - self.Gamma * (
                torch.ones(self.sequence_length,
                           self.sequence_length).to(Query.device) - self.lower_triangle_mask)),
                dim=last_dim)  # [T, T]

            attn = self.D @ P_h @ Value  # [T, dim_v] = [T, T][T, T][T, dim_v]

            if attention_mask:
                attn = attn * attention_mask

            cache_weights[
                f"Layer {self.current_layer} Attention Enabled: {self.is_cross_attention} head: {h} W_Q"] = self.W_Q.weight
            cache_weights[
                f"Layer {self.current_layer} Attention Enabled: {self.is_cross_attention} head: {h} W_K"] = self.W_K.weight
            cache_weights[
                f"Layer {self.current_layer} Attention Enabled: {self.is_cross_attention} head: {h} W_V"] = self.W_V.weight

            attn_for_heads.append(attn)

        assert len(
            attn_for_heads) == self.num_heads, "Number of heads mismatches number of attention matrices."

        concatenated_attns = torch.cat(attn_for_heads, dim=last_dim)

        # then we need to initialize and normalize W_0 to \mu = 0 and std = 1 / dim
        self.W_O.weight = nn.Parameter(
            nn.init.orthogonal(torch.empty_like(self.W_O.weight)))
        self.W_O.weight = nn.Parameter(
            self.W_O.weight.data.clone().mul(1 / self.embedding_dim ** .5))

        if use_cache:
            present_key_value = (Key, Value)
        else:
            present_key_value = None
        if need_weights:
            return self.W_O(concatenated_attns), cache_weights, present_key_value
        else:
            return self.W_O(concatenated_attns), None, present_key_value

    def calculate_gamma_l(self, layer_l: int,):
        return -.5 * torch.log(torch.Tensor([1 - self.alpha_L ** (2 * layer_l / self.max_layers)]).to("cuda"))

    def eigenmatrix_generator(self, gamma_l: int):
        if self.current_layer == 1:
            return torch.eye(self.sequence_length)

        i = torch.arange(self.sequence_length).reshape(-1, 1)
        j = torch.arange(self.sequence_length).reshape(1, -1)

        return torch.exp(-gamma_l * torch.abs(i - j).to("cuda")).to("cuda")

    def generate_causal_mask(self):
        return torch.tril(torch.ones(self.sequence_length, self.sequence_length)).to("cuda")

    def decompose_to_D_P(self, matrix: torch.Tensor):
        dim = len(matrix.shape) - 1
        D = torch.sum(input=matrix, dim=dim, keepdim=False)
        P = matrix / D

        # Warning: Don't move this line. Must be after the matrix / D, otherwise, there will be numerical overflow (nan)
        D = torch.diag(D)

        # Warning: dtype determinations are necessary!
        if D.dtype is not torch.float:
            D = D.float()

        if P.dtype is not torch.float:
            P = P.float()

        return D, P  # both are [T, T]
