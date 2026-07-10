"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=['<pad>', '<bos>', '<eos>', '<unk>']):
    # 1. Start with special tokens to reserve the first few IDs
    # Based on your expectation, <unk> needs to be at index 3
    token_to_id = {token: idx for idx, token in enumerate(specials)}
    
    # 2. Extract unique words from the sentences
    unique_words = []
    for sentence in sentences:
        for word in sentence.split():
            if word not in token_to_id and word not in unique_words:
                unique_words.append(word)
                
    
    # 3. Add the unique words to the vocabulary
    for word in unique_words:
        token_to_id[word] = len(token_to_id)
        
    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    out = {}
    for key, value in token_to_id.items():
        if value in out.keys():
            continue 
        else:
            out[value] = key 
    return out

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # Get the default fallback ID for unknown tokens
    unk_id = token_to_id[unk_token]
    
    # Split by whitespace and map each token to its ID
    return [token_to_id.get(word, unk_id) for word in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    return [id_to_token[id] for id in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # 1. Take up to max_len elements (handles truncation automatically if too long)
    # 2. Add as many pad_ids as needed to fill it up to max_len
    return ids[:max_len] + [pad_id] * (max_len - len(ids[:max_len]))

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings*math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    # for i in range d_model // 2:
    # wi = exp(2i * -log(10000)/d_model)
    return torch.tensor([10000**(-2*i/d_model) for i in range(d_model//2)])

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # 1. Create a 1D tensor from 0 to max_len-1 as float
    # 2. Reshape it to (max_len, 1) using unsqueeze or view
    return torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # pe shape: (max_len, d_model)
    # position * div_term shape: (max_len, d_model // 2)
    
    # 0::2 selects every second column starting from index 0 (0, 2, 4, ...)
    pe[:, 0::2] = torch.sin(position * div_term)
    
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # 1. Initialize an empty tensor of shape (max_len, d_model)
    pe = torch.zeros(max_len, d_model)
    
    # 2. Build the position column vector: shape (max_len, 1)
    position = build_position_index_column(max_len)
    
    # 3. Compute the geometric progression div_term: shape (d_model // 2,)
    div_term = compute_positional_div_term(d_model)
    
    # 4. Fill the even columns with sine and odd columns with cosine
    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)
    
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # 1. Get the sequence length (L) from the embedded batch
    # Note: Use embedded_batch.shape or embedded_batch.size(), not torch.shape()
    B, L, d_model = embedded_batch.shape
    
    # 2. Slice the positional encoding to match the sequence length L
    # Shape transitions from (max_len, d_model) -> (L, d_model)
    pe_sliced = positional_encoding[:L, :]
    
    # 3. Add them together. PyTorch broadcasts (L, d_model) across the batch dimension B.
    return embedded_batch + pe_sliced

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # 1. Create a 2D boolean mask of shape (B, L): True for valid tokens, False for padding
    mask = (token_ids != pad_id)
    
    # 2. Reshape to (B, 1, 1, L) so it correctly broadcasts against attention scores (B, H, L, L)
    # Using None indexing is a clean, standard way to inject singleton dimensions
    return mask[:, None, None, :]

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # 1. Create a 2D square matrix of ones of shape (seq_len, seq_len)
    ones_matrix = torch.ones((seq_len, seq_len))
    
    # 2. Get the lower triangular part (including diagonal)
    lower_triangular = torch.tril(ones_matrix)
    
    # 3. Cast to boolean (1 becomes True, 0 becomes False)
    bool_mask = lower_triangular.to(dtype=torch.bool)
    
    # 4. Reshape to (1, 1, seq_len, seq_len) for attention broadcasting
    return bool_mask[None, None, :, :]

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    """Combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L)."""
    # Bitwise AND ensures a position is True only if BOTH masks are True
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # query shape: (B, H, L, d_k)
    # key.transpose(-2, -1) shape: (B, H, d_k, L)
    return query @ key.transpose(-2, -1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores/math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    """Softmax over the last axis, safely turning entirely -inf rows into all zeros without NaN."""
    # 1. Identify rows that are completely filled with -inf 
    # (If the maximum value along the last axis is -inf, the whole row is -inf)
    is_fully_masked = (masked_scores.max(dim=-1, keepdim=True).values == float('-inf'))
    
    # 2. Replace -inf with 0.0 only in those fully masked rows to prevent 0/0 NaN
    # We use torch.where(condition, x, y) -> if condition is True, take from x, else y
    safe_scores = torch.where(is_fully_masked, torch.zeros_like(masked_scores), masked_scores)
    
    # 3. Compute standard softmax along the last axis
    weights = torch.nn.functional.softmax(safe_scores, dim=-1)
    
    # 4. Force the fully masked rows back to absolute zero
    return torch.where(is_fully_masked, torch.zeros_like(weights), weights)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    d_k = query.shape[-1]
    
    # 1. Compute raw attention scores: Q @ K^T -> shape (B, H, L, L)
    scores = compute_raw_attention_scores(query, key)
    
    # 2. Scale the scores by the square root of d_k
    scaled_scores = scale_attention_scores(scores, d_k)
    
    # 3. Optionally apply the mask (replace False/blocked positions with -inf)
    if mask is not None:
        # Pass 'scaled_scores' here to preserve the scaling step!
        scaled_scores = mask_attention_scores_with_neg_inf(scaled_scores, mask)
        
    # 4. Compute safe softmax attention weights across the last dimension -> shape (B, H, L, L)
    attention_weights = softmax_attention_weights(scaled_scores)
    
    # 5. Mix with values: Weights @ V -> shape (B, H, L, d_k)
    context = apply_attention_weights_to_values(attention_weights, value)
    
    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d = tensor.shape
    return torch.reshape(tensor, (B, L, num_heads, d // num_heads))

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return torch.transpose(split_tensor, 1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    """Merge the head axis back into the feature axis to reconstruct d_model.
    
    Input shape:  (B, num_heads, L, d_v)
    Output shape: (B, L, d_model) where d_model = num_heads * d_v
    """
    B, num_heads, L, d_v = multi_head_tensor.shape
    
    # 1. Permute axes from (B, num_heads, L, d_v) -> (B, L, num_heads, d_v)
    # This aligns the memory properly so sequence tokens stay isolated.
    permuted = multi_head_tensor.permute(0, 2, 1, 3)
    
    # 2. Reshape into (B, L, d_model)
    # Calling .contiguous() ensures the memory layout is clean before reshaping
    return permuted.contiguous().view(B, L, num_heads * d_v)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is not None:
        return x @ weight.T + bias
    else: 
        return x @ weight.T

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    Q = apply_linear_projection(x, w_q, b_q)
    K = apply_linear_projection(x, w_k, b_k)
    V = apply_linear_projection(x, w_v, b_v)
    return Q, K, V

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    """Split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple."""
    out = []
    for tensor in [q, k, v]:
        # 1. Reshape from (B, L, d_model) -> (B, L, num_heads, d_k)
        split = split_last_dim_into_heads(tensor, num_heads)
        
        # 2. Permute from (B, L, num_heads, d_k) -> (B, num_heads, L, d_k)
        transp = transpose_heads_before_sequence(split)
        
        out.append(transp)
    # Return as a clean, indexable tuple: (split_q, split_k, split_v)
    return tuple(out)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
import torch

def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """Run the complete forward pass for Multi-Head Attention.
    
    Inputs:
        query, key, value: Input tensors of shape (B, L, d_model)
        w_q, w_k, w_v: Linear projection weights for Q, K, V
        w_o: Final output projection weight matrix
        num_heads: Number of attention heads
        mask: Optional padding/causal mask
    """
    # 1. Project inputs to Q, K, V matrices -> shapes: (B, L, d_model)
    # Using your functional projection mapping
    q = torch.matmul(query, w_q)
    k = torch.matmul(key, w_k)
    v = torch.matmul(value, w_v)
    
    # 2. Split into multi-head layouts -> shapes: (B, num_heads, L, d_k)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    
    # 3. Run Scaled Dot-Product Attention -> context shape: (B, num_heads, L, d_k)
    # This also extracts attention weights if you need them for downstream analytics
    context_heads, attention_weights = scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)
    
    # 4. Merge heads back into model dimension -> shape: (B, L, d_model)
    merged_context = merge_heads_back_to_model_dim(context_heads)
    
    # 5. Output linear projection layer -> shape: (B, L, d_model)
    output = apply_linear_projection(merged_context, w_o, None)
    
    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x, w1, b1), w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    """Return (mean, variance) reduced over the last dim with shape (..., 1)."""
    # 1. Compute mean across the last dimension (dim=-1)
    mean = torch.mean(x, dim=-1, keepdim=True)
    
    # 2. Compute variance across the last dimension (dim=-1)
    # correction=0 ensures we use the biased variance (divided by N, not N-1)
    variance = torch.var(x, dim=-1, keepdim=True, correction=0)
    
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, var = compute_layer_norm_mean_and_variance(x)
    stand = (x-mean) / torch.sqrt(var+eps)
    return gamma*stand + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input+sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return x * keep_mask / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
import torch

def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    """Run multi-head self-attention on x and wrap with residual add-and-norm.
    
    Shapes:
        x: (B, L, d_model)
        gamma, beta: (d_model,) - LayerNorm learnable scale and shift parameters
    """
    # 1. Run Multi-Head Self-Attention
    # Since it is self-attention, Query, Key, and Value are all the same input tensor `x`
    attn_output = assemble_multi_head_attention_forward(
        query=x, key=x, value=x, 
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, 
        num_heads=num_heads, mask=src_mask
    )
    
    # 2. Apply Residual Connection followed by Layer Normalization
    # Formula: LayerNorm(x + Attention(x))
    # Passing the original 'x' is what creates the residual highway
    return apply_residual_add_and_norm(attn_output, x, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
import torch

def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    """Run the position-wise FFN on x and wrap it with residual add-and-norm.
    
    Shapes:
        x: (B, L, d_model)
        w1: (d_model, d_ff)
        b1: (d_ff,)
        w2: (d_ff, d_model)
        b2: (d_model,)
    """
    # 1. Run the position-wise feed-forward network
    ffn_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    
    # 2. Apply the residual shortcut and layer normalization
    # Formula: LayerNorm(x + FFN(x))
    return apply_residual_add_and_norm(ffn_output, x, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    """Chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    
    Shapes:
        x: (B, L, d_model)
        src_mask: (B, 1, 1, L) padding mask
    """
    # 1. Cleanly unpack all parameters from the layer_params dictionary
    w_q = layer_params["w_q"]
    w_k = layer_params["w_k"]
    w_v = layer_params["w_v"]
    w_o = layer_params["w_o"]
    
    w1 = layer_params["w1"]
    b1 = layer_params["b1"]
    w2 = layer_params["w2"]
    b2 = layer_params["b2"]
    
    attn_gamma = layer_params["attn_gamma"]
    attn_beta = layer_params["attn_beta"]
    ffn_gamma = layer_params["ffn_gamma"]
    ffn_beta = layer_params["ffn_beta"]

    # 2. Sublayer 1: Multi-Head Self-Attention + Residual Add & Norm
    # Output shape remains (B, L, d_model)
    attn_out = encoder_layer_self_attention_sublayer(
        x, w_q, w_k, w_v, w_o, 
        attn_gamma, attn_beta, 
        num_heads, src_mask
    )
    
    # 3. Sublayer 2: Position-Wise Feed-Forward Network + Residual Add & Norm
    # The output of the attention sublayer becomes the input to the FFN sublayer
    layer_out = encoder_layer_feed_forward_sublayer(
        attn_out, w1, b1, w2, b2, 
        ffn_gamma, ffn_beta
    )
    
    return layer_out

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    """Sequentially apply each encoder layer to the running hidden state and return the final tensor."""
    # Running hidden state initialized to the input x
    hidden_state = x
    
    # Iteratively pass the state through each encoder block in the list
    for layer_params in encoder_layer_params_list:
        hidden_state = assemble_encoder_layer(
            hidden_state, 
            layer_params, 
            num_heads, 
            src_mask
        )
        
    return hidden_state

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    """Run masked multi-head self-attention on y and wrap with residual add-and-norm.
    
    Shapes:
        y: (B, L_tgt, d_model)
        tgt_mask: (B, 1, L_tgt, L_tgt) causal/padding combined mask
    """
    # 1. Run Multi-Head Attention using 'y' for Query, Key, and Value
    # The tgt_mask ensures auto-regressive decoding (no look-ahead)
    attn_output = assemble_multi_head_attention_forward(
        query=y, key=y, value=y, 
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, 
        num_heads=num_heads, mask=tgt_mask
    )
    
    # 2. Apply the residual shortcut and layer normalization
    # Formula: LayerNorm(y + MaskedAttention(y))
    return apply_residual_add_and_norm(attn_output, y, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    """Run multi-head cross-attention and wrap with add-and-norm.
    
    This version automatically handles both 2D (B, L_src) and 4D (B, 1, 1, L_src) masks.
    """
    # If the mask is 2D (B, L_src), expand it to 4D (B, 1, 1, L_src)
    # so it broadcasts cleanly against attention scores (B, H, L_tgt, L_src)
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask[:, None, None, :]
    # ───────────────────────────────────────────────────────────────────

    # 1. Run Multi-Head Attention
    attn_output = assemble_multi_head_attention_forward(
        query=y, 
        key=encoder_output, 
        value=encoder_output, 
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, 
        num_heads=num_heads, 
        mask=src_mask
    )
    
    # 2. Apply Residual Connection followed by Layer Normalization
    return apply_residual_add_and_norm(attn_output, y, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    """Run the position-wise FFN on y and wrap it with residual add-and-norm.
    
    Shapes:
        y: (B, L_tgt, d_model)
        w1: (d_model, d_ff)
        b1: (d_ff,)
        w2: (d_ff, d_model)
        b2: (d_model,)
    """
    # 1. Run the position-wise feed-forward network
    ffn_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    
    # 2. Apply the residual shortcut and layer normalization
    # Formula: LayerNorm(y + FFN(y))
    return apply_residual_add_and_norm(ffn_output, y, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    
    # 1. Unpack Masked Self-Attention Parameters
    w_q_self, w_k_self, w_v_self, w_o_self = (
        layer_params["w_q_self"], layer_params["w_k_self"], 
        layer_params["w_v_self"], layer_params["w_o_self"]
    )
    self_gamma, self_beta = layer_params["self_gamma"], layer_params["self_beta"]
    
    # 2. Unpack Cross-Attention Parameters
    w_q_cross, w_k_cross, w_v_cross, w_o_cross = (
        layer_params["w_q_cross"], layer_params["w_k_cross"], 
        layer_params["w_v_cross"], layer_params["w_o_cross"]
    )
    cross_gamma, cross_beta = layer_params["cross_gamma"], layer_params["cross_beta"]
    
    # 3. Unpack Feed-Forward Network Parameters
    w1, b1, w2, b2 = layer_params["w1"], layer_params["b1"], layer_params["w2"], layer_params["b2"]
    ffn_gamma, ffn_beta = layer_params["ffn_gamma"], layer_params["ffn_beta"]

    # ──── STEP 1: Masked Self-Attention Sublayer ────
    # Processes the target sequence and prevents looking ahead
    y_self = decoder_layer_masked_self_attention_sublayer(
        y=y, 
        w_q=w_q_self, w_k=w_k_self, w_v=w_v_self, w_o=w_o_self, 
        gamma=self_gamma, beta=self_beta, 
        num_heads=num_heads, tgt_mask=tgt_mask
    )
    
    # ──── STEP 2: Cross-Attention Sublayer ────
    # Attends to the encoder hidden states using the self-attention output as Query
    y_cross = decoder_layer_cross_attention_sublayer(
        y=y_self, 
        encoder_output=encoder_output, 
        w_q=w_q_cross, w_k=w_k_cross, w_v=w_v_cross, w_o=w_o_cross, 
        gamma=cross_gamma, beta=cross_beta, 
        num_heads=num_heads, src_mask=src_mask
    )
    
    # ──── STEP 3: Feed-Forward Network Sublayer ────
    # Final non-linear mapping per token position
    layer_output = decoder_layer_feed_forward_sublayer(
        y=y_cross, 
        w1=w1, b1=b1, w2=w2, b2=b2, 
        gamma=ffn_gamma, beta=ffn_beta
    )
    
    return layer_output

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    """Sequentially apply each decoder layer to the running target hidden state.
    
    Inputs:
        y: Initial target hidden state of shape (B, L_tgt, d_model)
        encoder_output: Static encoder output of shape (B, L_src, d_model)
        decoder_layer_params_list: List containing the parameter dictionaries for each block
    """
    # Initialize the running hidden state with the input target states
    hidden_state = y
    
    # Sequentially route the state through each decoder layer
    for layer_params in decoder_layer_params_list:
        hidden_state = assemble_decoder_layer(
            y=hidden_state,
            encoder_output=encoder_output,
            layer_params=layer_params,
            num_heads=num_heads,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
    return hidden_state

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return torch.nn.functional.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

