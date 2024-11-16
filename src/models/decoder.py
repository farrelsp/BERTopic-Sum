"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, DecoderState

MAX_SIZE = 2000
MAX_BATCH = 30
EPSILON = 1e-6

class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, topic_len):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.topic_len = topic_len       
        
        if self.topic_len == 0: # Baseline
            mask = self._get_attn_subsequent_mask(MAX_SIZE)

        else: # If topic embedding is used
            mask = self._get_attn_subsequent_mask_with_topic(self.topic_len, MAX_SIZE)

            topic_pad_mask = torch.zeros(MAX_BATCH, MAX_SIZE, self.topic_len)
            topic_mask_tgt = abs(np.triu(np.ones((MAX_BATCH, self.topic_len, MAX_SIZE)), k=0) * 
                            np.tril(np.ones((MAX_BATCH, self.topic_len, MAX_SIZE)), k=0) - 1).astype('uint8')
            topic_mask_tgt = torch.from_numpy(topic_mask_tgt)
            topic_mask_src = torch.ones(MAX_BATCH, self.topic_len, MAX_SIZE)
            
            self.register_buffer('topic_pad_mask', topic_pad_mask)
            self.register_buffer('topic_mask_tgt', topic_mask_tgt)
            self.register_buffer('topic_mask_src', topic_mask_src)

            # # Scenario 8
            # mask = self._get_attn_subsequent_mask_with_topic(self.topic_len, MAX_SIZE)

            # topic_pad_mask = torch.zeros(MAX_BATCH, MAX_SIZE, self.topic_len)

            # # Scenario 8-1
            # topic_mask_tgt = np.triu(np.ones((MAX_BATCH, topic_len, MAX_SIZE)), k=1).astype("int8")
            # topic_mask_tgt = torch.from_numpy(topic_mask_tgt)
        
            # # Scenario 8-2
            # topic_mask_tgt = np.zeros((MAX_BATCH, topic_len, topic_len))
            # topic_mask_tgt = np.concatenate((topic_mask_tgt, np.ones((MAX_BATCH, topic_len, MAX_SIZE-topic_len))), axis=2).astype('uint8')
            # topic_mask_tgt = torch.from_numpy(topic_mask_tgt)
        
            # topic_mask_src = torch.ones(MAX_BATCH, self.topic_len, MAX_SIZE)
            
            # self.register_buffer('topic_pad_mask', topic_pad_mask)
            # self.register_buffer('topic_mask_tgt', topic_mask_tgt)
            # self.register_buffer('topic_mask_src', topic_mask_src)
            
        self.register_buffer('mask', mask)
 
    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, 
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x tgt_len x model_dim]`
                The current input to the decoder layer, typically the embedding of the target token at the current time step.
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
                The output from the encoder, which contains the encoded representations of the source sequence.
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
                Mask for the source sequence to ignore the padding tokens during attention.
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`
                Mask for the target sequence to ignore the padding tokens during self-attention.
        
        Returns:
            (`FloatTensor`, `FloatTensor`):
        
            * output `[batch_size x 1 x model_dim]`
                The output of the decoder layer.
            * all_input `[batch_size x current_step x model_dim]`
                The concatenated inputs up to the current step, used for subsequent layers and steps.
        """
        
        if inputs.size(1) == tgt_pad_mask.size(1):  # Baseline
            # Generate a mask to prevent attending to subsequent positions in the target sequence.
            dec_mask = torch.gt(tgt_pad_mask +
                                self.mask[:, :tgt_pad_mask.size(1),
                                        :tgt_pad_mask.size(1)], 0)
            
        else:
            # Scenario 1 (Default) ----------------------------------------------------------------------
            # Mask for SA decoder
            # Phase 1
            tgt_pad_mask = torch.cat((tgt_pad_mask, self.topic_pad_mask[:tgt_pad_mask.size(0), :tgt_pad_mask.size(1), :]), axis=2)
            tgt_pad_mask = tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(2)]
            
            # Phase 2
            topic_batch = tgt_pad_mask.size(0)
            total_len = tgt_pad_mask.size(2)
            
            # Phase 3
            dec_mask = torch.cat((self.topic_mask_tgt[:topic_batch, :self.topic_len, :total_len], tgt_pad_mask), axis=1)
            dec_mask = torch.gt(dec_mask, 0)
            
            # Mask for CA decoder
            src_len = src_pad_mask.shape[-1]
            src_pad_mask = torch.cat((self.topic_mask_src[:topic_batch, :self.topic_len, :src_len], src_pad_mask), axis=1)
            src_pad_mask = torch.gt(src_pad_mask, 0)
            # ------------------------------------------------------------------------------------------
            
        # Normalize the inputs using the first layer normalization.
        input_norm = self.layer_norm_1(inputs) # inputs = target summary
        
        # Initialize all_input with the normalized inputs.
        all_input = input_norm
        
        # If previous_input is provided, concatenate it with the current input to form all_input.
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None
        
        # Apply self-attention using all_input as both the query, key, and value.
        # The attention mask (dec_mask) is used to ensure proper masking.
        query = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        # Apply dropout and add the input to the query (residual connection).
        query = self.drop(query) + inputs

        # Normalize the query using the second layer normalization.
        query_norm = self.layer_norm_2(query)

        # Apply context-attention using the memory_bank (encoder output) as the key and value,
        # and the normalized query as the query.
        mid = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")

        # Apply dropout, add the query (residual connection), and pass through the feed-forward network.
        output = self.feed_forward(self.drop(mid) + query)
        
        # Return the output and the concatenated inputs up to the current step.
        return output, all_input
        
    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask
        
    def _get_attn_subsequent_mask_with_topic(self, topic_len, tgt_len):
        attn_shape = (1, tgt_len, topic_len + tgt_len)
        subsequent_mask = np.triu(np.ones(attn_shape), k=topic_len+1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask

class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O

    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, vocab_size, topic_len):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        
        self.topic_len = topic_len
        if self.topic_len > 0:
            word_emb = torch.tensor([i for i in range(vocab_size)])
            self.register_buffer('word_emb', word_emb)

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout, topic_len)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, tgt, memory_bank, state, topic_dist=None, memory_lengths=None, 
                step=None, cache=None, memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        Perform a forward pass through the TransformerDecoder.

        Args:
            tgt (Tensor): The target sequence input.
            memory_bank (Tensor): The encoder's output (memory bank).
            state (TransformerDecoderState): The state of the decoder.
            memory_lengths (Tensor, optional): Lengths of the memory bank sequences.
            step (int, optional): The current step in the sequence generation.
            cache (dict, optional): Cache for the transformer layers to speed up decoding.
            memory_masks (Tensor, optional): Masks for the memory bank to handle padding.
    
        Returns:
            Tuple[Tensor, TransformerDecoderState]: The decoder outputs and the updated decoder state.
        """

        # Extract the source words from the state and target words from the input.
        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        if topic_dist is None:  # Baseline
            emb = self.embeddings(tgt)
            assert emb.dim() == 3   # Ensure the embedding dimension is correct (batch x len_tokens x embedding_dim).
            
            # Apply positional encoding to the embedded target sequence.
            output = self.pos_emb(emb, step)
            
        else: # If topic embedding is used
            topic_batch, topic_len, _ = topic_dist.size()

            emb = self.embeddings(tgt)
            assert emb.dim() == 3   # Ensure the embedding dimension is correct (batch x len_tokens x embedding_dim).

            # Apply positional encoding to the embedded target sequence.
            output = self.pos_emb(emb, step)

            topic_dist = topic_dist.float()

            # Obtain topic embedding
            word_embeddings = self.embeddings(self.word_emb)
            topic_emb = torch.matmul(topic_dist.float(), word_embeddings.float())
                        
            # Add topic embedding to the target that has been embedded
            output = torch.cat((topic_emb, output), 1) 

            # # Scenario 3 ---------------------------------------------------------------------------------------
            # emb = self.embeddings(tgt)
            # assert emb.dim() == 3   # Ensure the embedding dimension is correct (batch x len_tokens x embedding_dim).

            # # Apply positional encoding to the embedded target sequence.
            # output = self.pos_emb(emb, step)

            # topic_dist = topic_dist.long()

            # topic_dist = self.embeddings(topic_dist)
            # topic_emb = torch.sum(topic_dist, axis=2)
                        
            # # Add topic embedding to the target that has been embedded
            # output = torch.cat((topic_emb, output), 1)

            # # Scenario 5 ------------------------------------------------------------------------
            # emb = self.embeddings(tgt)
            # assert emb.dim() == 3   # Ensure the embedding dimension is correct (batch x len_tokens x embedding_dim).
    
            # word_embeddings = self.embeddings(self.word_emb)
            # topic_emb = torch.matmul(topic_dist.float(), word_embeddings.float())
    
            # emb = torch.cat((topic_emb, emb), 1)
    
            # output = self.pos_emb(emb, step)

            # # Scenario 6 ------------------------------------------------------------------------
            # emb = self.embeddings(tgt)
            # assert emb.dim() == 3   # Ensure the embedding dimension is correct (batch x len_tokens x embedding_dim).
    
            # output = self.pos_emb(emb, step)
    
            # word_embeddings = self.embeddings(self.word_emb)
            # topic_emb = torch.matmul(topic_dist.float(), word_embeddings.float())
            # topic_emb = self.pos_emb(topic_emb, step)
            
            # output = torch.cat((topic_emb, output), 1)

        # Prepare masks for the source and target sequences.
        src_memory_bank = memory_bank  # shape = batch_size x src_len x emb_size
        padding_idx = self.embeddings.padding_idx

        # Create padding mask for the target sequence.
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        # If memory masks are provided, use them; otherwise, create padding masks for the source sequence.
        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        # Initialize storage for the inputs to each layer if no cache is used.
        if state.cache is None:
            saved_inputs = []

        # Process the target sequence through each transformer decoder layer.
        for i in range(self.num_layers):
            prev_layer_input = None

            # Retrieve previous layer input if available.
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]

            # Forward pass through the i-th transformer decoder layer.
            output, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)

            # Save the input to this layer if no cache is used.
            if state.cache is None:
                saved_inputs.append(all_input)

        # Stack saved inputs if no cache is used.
        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        # Apply layer normalization to the output.
        output = self.layer_norm(output)

        # Update the decoder state with the new inputs if no cache is used.
        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        # Scenario Default
        # Return the final output and the updated decoder state.
        if self.topic_len > 0:
            output = output[:, self.topic_len:, :]
        
        return output, state

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


# # Scenario 7
# class TransformerDecoderLayer(nn.Module):
#     """
#     Args:
#       d_model (int): the dimension of keys/values/queries in
#                        MultiHeadedAttention, also the input size of
#                        the first-layer of the PositionwiseFeedForward.
#       heads (int): the number of heads for MultiHeadedAttention.
#       d_ff (int): the second-layer of the PositionwiseFeedForward.
#       dropout (float): dropout probability(0-1.0).
#       self_attn_type (string): type of self-attention scaled-dot, average
#     """

#     def __init__(self, d_model, heads, d_ff, dropout, topic_len):
#         super(TransformerDecoderLayer, self).__init__()

#         self.self_attn = MultiHeadedAttention(
#             heads, d_model, dropout=dropout)

#         self.context_attn = MultiHeadedAttention(
#             heads, d_model, dropout=dropout)
#         self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
#         self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
#         self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
#         self.drop = nn.Dropout(dropout)
#         self.topic_len = topic_len       
        
#         if self.topic_len == 0: # Baseline
#             mask = self._get_attn_subsequent_mask(MAX_SIZE)

#         else: # If topic embedding is used
#             mask = self._get_attn_subsequent_mask_with_topic_reversed(self.topic_len, MAX_SIZE)
#             topic_pad_mask = torch.zeros(MAX_BATCH, MAX_SIZE, self.topic_len)
#             topic_mask_tgt = abs(np.triu(np.ones((MAX_BATCH, self.topic_len, MAX_SIZE)), k=MAX_SIZE-topic_len) * 
#                 np.tril(np.ones((MAX_BATCH, self.topic_len, MAX_SIZE)), k=MAX_SIZE-topic_len) - 1).astype('uint8')
#             topic_mask_tgt = torch.from_numpy(topic_mask_tgt)
#             topic_mask_src = torch.ones(MAX_BATCH, self.topic_len, MAX_SIZE)

#             self.register_buffer('topic_pad_mask', topic_pad_mask)
#             self.register_buffer('topic_mask_tgt', topic_mask_tgt)
#             self.register_buffer('topic_mask_src', topic_mask_src)
            
#         self.register_buffer('mask', mask)
 
#     def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, 
#                 previous_input=None, layer_cache=None, step=None):
#         """
#         Args:
#             inputs (`FloatTensor`): `[batch_size x tgt_len x model_dim]`
#                 The current input to the decoder layer, typically the embedding of the target token at the current time step.
#             memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
#                 The output from the encoder, which contains the encoded representations of the source sequence.
#             src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
#                 Mask for the source sequence to ignore the padding tokens during attention.
#             tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`
#                 Mask for the target sequence to ignore the padding tokens during self-attention.
        
#         Returns:
#             (`FloatTensor`, `FloatTensor`):
        
#             * output `[batch_size x 1 x model_dim]`
#                 The output of the decoder layer.
#             * all_input `[batch_size x current_step x model_dim]`
#                 The concatenated inputs up to the current step, used for subsequent layers and steps.
#         """
        
#         if inputs.size(1) == tgt_pad_mask.size(1):  # Baseline
#             # Generate a mask to prevent attending to subsequent positions in the target sequence.
#             dec_mask = torch.gt(tgt_pad_mask +
#                                 self.mask[:, :tgt_pad_mask.size(1),
#                                         :tgt_pad_mask.size(1)], 0)
#         else:
#             # Mask for SA decoder
#             # Phase 1
#             tgt_pad_mask = torch.cat((tgt_pad_mask, self.topic_pad_mask[:tgt_pad_mask.size(0), :tgt_pad_mask.size(1), :]), axis=2)
#             tgt_pad_mask = tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(2)]

#             # Phase 2
#             topic_batch = tgt_pad_mask.size(0)
#             total_len = tgt_pad_mask.size(2)

#             # Phase 3
#             dec_mask = torch.cat((self.topic_mask_tgt[:topic_batch, :self.topic_len, -total_len:], tgt_pad_mask), axis=1)
#             dec_mask = torch.gt(dec_mask, 0)

#             # Mask for CA decoder
#             src_len = src_pad_mask.shape[-1]
#             src_pad_mask = torch.cat((src_pad_mask, self.topic_mask_src[:topic_batch, :self.topic_len, :src_len]), axis=1)
#             src_pad_mask = torch.gt(src_pad_mask, 0)
            
#         # Normalize the inputs using the first layer normalization.
#         input_norm = self.layer_norm_1(inputs) # inputs = target summary
        
#         # Initialize all_input with the normalized inputs.
#         all_input = input_norm

#         # If previous_input is provided, concatenate it with the current input to form all_input.
#         if previous_input is not None:
#             all_input = torch.cat((previous_input, input_norm), dim=1)
#             dec_mask = None
        
#         # Apply self-attention using all_input as both the query, key, and value.
#         # The attention mask (dec_mask) is used to ensure proper masking.
#         query = self.self_attn(all_input, all_input, input_norm,
#                                      mask=dec_mask,
#                                      layer_cache=layer_cache,
#                                      type="self")

#         # Apply dropout and add the input to the query (residual connection).
#         query = self.drop(query) + inputs

#         # Normalize the query using the second layer normalization.
#         query_norm = self.layer_norm_2(query)

#         # Apply context-attention using the memory_bank (encoder output) as the key and value,
#         # and the normalized query as the query.
#         mid = self.context_attn(memory_bank, memory_bank, query_norm,
#                                       mask=src_pad_mask,
#                                       layer_cache=layer_cache,
#                                       type="context")

#         # Apply dropout, add the query (residual connection), and pass through the feed-forward network.
#         output = self.feed_forward(self.drop(mid) + query)
        
#         # Return the output and the concatenated inputs up to the current step.
#         return output, all_input
        
#     def _get_attn_subsequent_mask(self, size):
#         attn_shape = (1, size, size)
#         subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#         subsequent_mask = torch.from_numpy(subsequent_mask)
#         return subsequent_mask
        
#     def _get_attn_subsequent_mask_with_topic(self, topic_len, tgt_len):
#         attn_shape = (1, tgt_len, topic_len + tgt_len)
#         subsequent_mask = np.triu(np.ones(attn_shape), k=topic_len+1).astype('uint8')
#         subsequent_mask = torch.from_numpy(subsequent_mask)
#         return subsequent_mask

#     def _get_attn_subsequent_mask_with_topic_reversed(self, topic_len, tgt_len):
#         attn_shape = (1, tgt_len, tgt_len)
#         addition_shape = (1, tgt_len, topic_len)
#         subsequent_mask = np.triu(np.ones(attn_shape), k=1)
#         subsequent_mask = np.concatenate((subsequent_mask, np.zeros(addition_shape)), axis=2).astype('uint8')
#         subsequent_mask = torch.from_numpy(subsequent_mask)
#         return subsequent_mask


# class TransformerDecoder(nn.Module):
#     """
#     The Transformer decoder from "Attention is All You Need".

#     .. mermaid::

#        graph BT
#           A[input]
#           B[multi-head self-attn]
#           BB[multi-head src-attn]
#           C[feed forward]
#           O[output]
#           A --> B
#           B --> BB
#           BB --> C
#           C --> O

#     Args:
#        num_layers (int): number of encoder layers.
#        d_model (int): size of the model
#        heads (int): number of heads
#        d_ff (int): size of the inner FF layer
#        dropout (float): dropout parameters
#        embeddings (:obj:`onmt.modules.Embeddings`):
#           embeddings to use, should have positional encodings
#        attn_type (str): if using a seperate copy attention
#     """

#     def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, vocab_size, topic_len):
#         super(TransformerDecoder, self).__init__()

#         # Basic attributes.
#         self.decoder_type = 'transformer'
#         self.num_layers = num_layers
#         self.embeddings = embeddings
#         self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        
#         self.topic_len = topic_len
#         if self.topic_len > 0:
#             word_emb = torch.tensor([i for i in range(vocab_size)])
#             self.register_buffer('word_emb', word_emb)

#         # Build TransformerDecoder.
#         self.transformer_layers = nn.ModuleList(
#             [TransformerDecoderLayer(d_model, heads, d_ff, dropout, topic_len)
#              for _ in range(num_layers)])

#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
#     def forward(self, tgt, memory_bank, state, topic_dist=None, memory_lengths=None, 
#                 step=None, cache=None, memory_masks=None):
#         """
#         See :obj:`onmt.modules.RNNDecoderBase.forward()`
#         Perform a forward pass through the TransformerDecoder.

#         Args:
#             tgt (Tensor): The target sequence input.
#             memory_bank (Tensor): The encoder's output (memory bank).
#             state (TransformerDecoderState): The state of the decoder.
#             memory_lengths (Tensor, optional): Lengths of the memory bank sequences.
#             step (int, optional): The current step in the sequence generation.
#             cache (dict, optional): Cache for the transformer layers to speed up decoding.
#             memory_masks (Tensor, optional): Masks for the memory bank to handle padding.
    
#         Returns:
#             Tuple[Tensor, TransformerDecoderState]: The decoder outputs and the updated decoder state.
#         """

#         # Extract the source words from the state and target words from the input.
#         src_words = state.src
#         tgt_words = tgt
#         src_batch, src_len = src_words.size()
#         tgt_batch, tgt_len = tgt_words.size()

#         if topic_dist is None:  # Baseline
#             emb = self.embeddings(tgt)
#             assert emb.dim() == 3   # Ensure the embedding dimension is correct (batch x len_tokens x embedding_dim).
            
#             # Apply positional encoding to the embedded target sequence.
#             output = self.pos_emb(emb, step)
            
#         else: # If topic embedding is used
#             topic_batch, topic_len, _ = topic_dist.size()

#             emb = self.embeddings(tgt)
#             assert emb.dim() == 3   # Ensure the embedding dimension is correct (batch x len_tokens x embedding_dim).

#             # Apply positional encoding to the embedded target sequence.
#             output = self.pos_emb(emb, step)

#             topic_dist = topic_dist.float()

#             # Obtain topic embedding
#             word_embeddings = self.embeddings(self.word_emb)
#             topic_emb = torch.matmul(topic_dist.float(), word_embeddings.float())
                        
#             # Add topic embedding to the target that has been embedded
#             output = torch.cat((output, topic_emb), 1) 
 
#         # Prepare masks for the source and target sequences.
#         src_memory_bank = memory_bank  # shape = batch_size x src_len x emb_size
#         padding_idx = self.embeddings.padding_idx

#         # Create padding mask for the target sequence.
#         tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
#             .expand(tgt_batch, tgt_len, tgt_len)

#         # If memory masks are provided, use them; otherwise, create padding masks for the source sequence.
#         if (not memory_masks is None):
#             src_len = memory_masks.size(-1)
#             src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

#         else:
#             src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
#                 .expand(src_batch, tgt_len, src_len)

#         # Initialize storage for the inputs to each layer if no cache is used.
#         if state.cache is None:
#             saved_inputs = []

#         # Process the target sequence through each transformer decoder layer.
#         for i in range(self.num_layers):
#             prev_layer_input = None

#             # Retrieve previous layer input if available.
#             if state.cache is None:
#                 if state.previous_input is not None:
#                     prev_layer_input = state.previous_layer_inputs[i]

#             # Forward pass through the i-th transformer decoder layer.
#             output, all_input \
#                 = self.transformer_layers[i](
#                     output, src_memory_bank,
#                     src_pad_mask, tgt_pad_mask,
#                     previous_input=prev_layer_input,
#                     layer_cache=state.cache["layer_{}".format(i)]
#                     if state.cache is not None else None,
#                     step=step)

#             # Save the input to this layer if no cache is used.
#             if state.cache is None:
#                 saved_inputs.append(all_input)

#         # Stack saved inputs if no cache is used.
#         if state.cache is None:
#             saved_inputs = torch.stack(saved_inputs)

#         # Apply layer normalization to the output.
#         output = self.layer_norm(output)

#         # Update the decoder state with the new inputs if no cache is used.
#         if state.cache is None:
#             state = state.update_state(tgt, saved_inputs)

#         # Scenario 7-1
#         output = output[:, self.topic_len:, :]

#         # Scenario 7-2
#         # output = output[:, :-self.topic_len, :]
        
#         return output, state

#     def init_decoder_state(self, src, memory_bank,
#                            with_cache=False):
#         """ Init decoder state """
#         state = TransformerDecoderState(src)
#         if with_cache:
#             state._init_cache(memory_bank, self.num_layers)
#         return state


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)