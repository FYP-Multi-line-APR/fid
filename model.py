import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# from ..utils import ModelOutput
from transformers.modeling_outputs import ModelOutput

import src.meta

class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
        ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        print("[INFO]: expand inputs for generation calling....")
        print("EXPAND SIZE:",expand_size)
        print("is_encoder_decoder:",is_encoder_decoder)
        print("input ids:",input_ids)
        # print("kwargs:",model_kwargs)

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs




    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        print("[INFO]: extract past from model output calling....")
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the CodeT5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        print("[INFO]: fid forward calling....")
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_contexts = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    # def generate(self, input_ids, attention_mask, max_length, num_beams, **kwargs):
    #     self.encoder.n_contexts = input_ids.size(1)
    #     x = super().generate(
    #         input_ids=input_ids.view(input_ids.size(0), -1),
    #         attention_mask=attention_mask.view(attention_mask.size(0), -1),
    #         max_length=max_length,
    #         num_beams=num_beams
    #     )
    #     print (x.shape)
    #     return x

    # def generate(self, input_ids, attention_mask, max_length, num_beams=1):
    #     self.encoder.n_contexts = input_ids.size(1)
        
    #     # Ensure that encoder_outputs is included in the model's forward method
    #     encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    #     return super().generate(
    #         input_ids=input_ids.view(input_ids.size(0), -1),
    #         attention_mask=attention_mask.view(attention_mask.size(0), -1),
    #         max_length=max_length,
    #         num_beams=num_beams,
    #         encoder_outputs=encoder_outputs  # Pass encoder_outputs to the generation method
    #     )

    def generate(self, input_ids, attention_mask, max_length, num_beams, **kwargs):
        self.encoder.n_contexts = input_ids.size(1)

        print("[INFO]: starting...")

        encoder_outputs = self.encoder(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            **kwargs
        )

        # Extract the last hidden states from the encoder outputs
        last_hidden_states = encoder_outputs[0]

        # Prepare the decoder inputs (if needed)
        decoder_inputs = self.prepare_decoder_input(
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states,
        )

        print("[INFO]: before generate")

        # Generate sequences using the current model (FiDT5)
        # generated_output = super().generate(
        #     input_ids=input_ids.view(input_ids.size(0), -1),
        #     attention_mask=attention_mask.view(attention_mask.size(0), -1),
        #     max_length=max_length,
        #     # num_beams=2,
        #     **kwargs
        # )

        # Generate sequences using the decoder
        generated_output = self.decoder.generate(
            **decoder_inputs,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

        print("[INFO]:generated output:",generated_output.shape)

        return generated_output

    def prepare_decoder_input(self, input_ids, attention_mask, last_hidden_states):
        # Assuming some preprocessing is needed for the decoder input
        # Modify this according to your specific requirements
        decoder_inputs = {
            'input_ids': input_ids,  # Add any other necessary inputs
            'attention_mask': attention_mask,
            'encoder_hidden_states': last_hidden_states,
        }
        return decoder_inputs



    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_contexts = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_contexts, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_contexts, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

         # key code
        self.main_input_name = encoder.main_input_name

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

        self.embed_tokens = self.encoder.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_contexts * context_length
        bsz, total_length = input_ids.shape
        context_length = total_length // self.n_contexts
        input_ids = input_ids.view(bsz*self.n_contexts, context_length)
        attention_mask = attention_mask.view(bsz*self.n_contexts, context_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_contexts*context_length, -1), ) + outputs[1:]
        return outputs
    # def forward(self, input_ids=None, attention_mask=None, **kwargs):
    #     print("[INFO]: encoder wrapper forward calling....")
    #     # total_length = n_contexts * context_length
    #     bsz, total_length = input_ids.shape[:2]  # Use indexing to get the first two dimensions
    #     context_length = total_length // self.n_contexts
    #     input_ids = input_ids.view(bsz * self.n_contexts, context_length)
    #     attention_mask = attention_mask.view(bsz * self.n_contexts, context_length)
    #     outputs = self.encoder(input_ids, attention_mask, **kwargs)
    #     print("[INFO]: encoder wrapper output shape - 1:",outputs.shape)
    #     outputs = (outputs[0].view(bsz, self.n_contexts * context_length, -1),) + outputs[1:]
    #     print("[INFO]: encoder wrapper output shape - 2:",outputs.shape)
    #     return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        print("[INFO]: checkpoint wrapper forward calling....")
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

# class RetrieverConfig(transformers.BertConfig):

#     def __init__(self,
#                  indexing_dimension=768,
#                  apply_bug_mask=False,
#                  apply_context_mask=False,
#                  extract_cls=False,
#                  context_max_length=450,
#                  bug_max_length=62,
#                  projection=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.indexing_dimension = indexing_dimension
#         self.apply_bug_mask = apply_bug_mask
#         self.apply_context_mask = apply_context_mask
#         self.extract_cls=extract_cls
#         self.context_max_length = context_max_length
#         self.bug_max_length = bug_max_length
#         self.projection = projection


# use RoBERTa instead of BERT
class RetrieverConfig(transformers.RobertaConfig):
        def __init__(self,
                    indexing_dimension=768,
                    apply_bug_mask=False,
                    apply_context_mask=False,
                    extract_cls=False,
                    context_max_length=258,
                    bug_max_length=62,
                    projection=True,
                    **kwargs):
            super().__init__(**kwargs)
            self.indexing_dimension = indexing_dimension
            self.apply_bug_mask = apply_bug_mask
            self.apply_context_mask = apply_context_mask
            self.extract_cls=extract_cls
            self.context_max_length = context_max_length
            self.bug_max_length = bug_max_length
            self.projection = projection

# class Retriever(transformers.PreTrainedModel):

#     config_class = RetrieverConfig
#     base_model_prefix = "retriever"

#     def __init__(self, config, initialize_wBERT=False):
#         super().__init__(config)
#         assert config.projection or config.indexing_dimension == 768, \
#             'If no projection then indexing dimension must be equal to 768'
#         self.config = config
#         if initialize_wBERT:
#             self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
#         else:
#             self.model = transformers.BertModel(config)
#         if self.config.projection:
#             self.proj = nn.Linear(
#                 self.model.config.hidden_size,
#                 self.config.indexing_dimension
#             )
#             self.norm = nn.LayerNorm(self.config.indexing_dimension)
#         self.loss_fct = torch.nn.KLDivLoss()

#     def forward(self,
#                 bug_ids,
#                 bug_mask,
#                 context_ids,
#                 context_mask,
#                 gold_score=None):
#         bug_output = self.embed_text(
#             text_ids=bug_ids,
#             text_mask=bug_mask,
#             apply_mask=self.config.apply_bug_mask,
#             extract_cls=self.config.extract_cls,
#         )
#         bsz, n_contexts, plen = context_ids.size()
#         context_ids = context_ids.view(bsz * n_contexts, plen)
#         context_mask = context_mask.view(bsz * n_contexts, plen)
#         context_output = self.embed_text(
#             text_ids=context_ids,
#             text_mask=context_mask,
#             apply_mask=self.config.apply_context_mask,
#             extract_cls=self.config.extract_cls,
#         )

#         score = torch.einsum(
#             'bd,bid->bi',
#             bug_output,
#             context_output.view(bsz, n_contexts, -1)
#         )
#         score = score / np.sqrt(bug_output.size(-1))
#         if gold_score is not None:
#             loss = self.kldivloss(score, gold_score)
#         else:
#             loss = None

#         return bug_output, context_output, score, loss

#     def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
#         text_output = self.model(
#             input_ids=text_ids,
#             attention_mask=text_mask if apply_mask else None
#         )
#         if type(text_output) is not tuple:
#             text_output.to_tuple()
#         text_output = text_output[0]
#         if self.config.projection:
#             text_output = self.proj(text_output)
#             text_output = self.norm(text_output)

#         if extract_cls:
#             text_output = text_output[:, 0]
#         else:
#             if apply_mask:
#                 text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
#                 text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
#             else:
#                 text_output = torch.mean(text_output, dim=1)
#         return text_output

#     def kldivloss(self, score, gold_score):
#         gold_score = torch.softmax(gold_score, dim=-1)
#         score = torch.nn.functional.log_softmax(score, dim=-1)
#         return self.loss_fct(score, gold_score)


class Retriever(transformers.PreTrainedModel):
    
        config_class = RetrieverConfig
        base_model_prefix = "retriever"
    
        def __init__(self, config, initialize_wBERT=False):
            super().__init__(config)
            assert config.projection or config.indexing_dimension == 768, \
                'If no projection then indexing dimension must be equal to 768'
            self.config = config
            if initialize_wBERT:
                self.model = transformers.RobertaModel.from_pretrained(meta.MODEL_NAME)
            else:
                self.model = transformers.RobertaModel(config)
            if self.config.projection:
                self.proj = nn.Linear(
                    self.model.config.hidden_size,
                    self.config.indexing_dimension
                )
                self.norm = nn.LayerNorm(self.config.indexing_dimension)
            self.loss_fct = torch.nn.KLDivLoss()
    
        def forward(self,
                    bug_ids,
                    bug_mask,
                    context_ids,
                    context_mask,
                    gold_score=None):
            bug_output = self.embed_text(
                text_ids=bug_ids,
                text_mask=bug_mask,
                apply_mask=self.config.apply_bug_mask,
                extract_cls=self.config.extract_cls,
            )
            bsz, n_contexts, plen = context_ids.size()
            context_ids = context_ids.view(bsz * n_contexts, plen)
            context_mask = context_mask.view(bsz * n_contexts, plen)
            context_output = self.embed_text(
                text_ids=context_ids,
                text_mask=context_mask,
                apply_mask=self.config.apply_context_mask,
                extract_cls=self.config.extract_cls,
            )
    
            score = torch.einsum(
                'bd,bid->bi',
                bug_output,
                context_output.view(bsz, n_contexts, -1)
            )
            score = score / np.sqrt(bug_output.size(-1))
            if gold_score is not None:
                loss = self.kldivloss(score, gold_score)
            else:
                loss = None
    
            return bug_output, context_output, score, loss
    
        def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
            text_output = self.model(
                input_ids=text_ids,
                attention_mask=text_mask if apply_mask else None
            )
            if type(text_output) is not tuple:
                text_output.to_tuple()
            text_output = text_output[0]
            if self.config.projection:
                text_output = self.proj(text_output)
                text_output = self.norm(text_output)

            if extract_cls:
                text_output = text_output[:, 0]
            else:
                if apply_mask:
                    text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                    text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
                else:
                    text_output = torch.mean(text_output, dim=1)
            return text_output
        
        def kldivloss(self, score, gold_score):
            gold_score = torch.softmax(gold_score, dim=-1)
            score = torch.nn.functional.log_softmax(score, dim=-1)
            return self.loss_fct(score, gold_score)
        