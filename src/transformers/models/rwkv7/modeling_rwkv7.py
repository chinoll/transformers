# Copyright 2026 Chinoll and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RWKV7 model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from .configuration_rwkv7 import Rwkv7Config


logger = logging.get_logger(__name__)


def rwkv7_linear_attention(
    decay, receptance, key, value, in_context_state, in_context_rate, state=None, return_state=False
):
    batch_size, seq_length, num_heads, head_size = key.shape
    input_dtype = value.dtype
    has_initial_state = state is not None

    if state is None:
        state = torch.zeros(
            batch_size,
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=value.device,
        )
    else:
        state = state.float()

    if seq_length == 1:
        current_decay = decay[:, 0].float()
        current_receptance = receptance[:, 0].float()
        current_key = key[:, 0]
        current_value = value[:, 0]
        current_in_context_state = in_context_state[:, 0]
        current_in_context_rate = in_context_rate[:, 0]

        value_key = (current_value.unsqueeze(-1) @ current_key.unsqueeze(-2)).float()
        in_context = (current_in_context_state.unsqueeze(-1) @ current_in_context_rate.unsqueeze(-2)).float()
        state = state * current_decay.unsqueeze(-2) + state @ in_context + value_key
        output = (state.to(input_dtype) @ current_receptance.to(input_dtype).unsqueeze(-1)).squeeze(-1)

        if not (return_state or has_initial_state):
            state = None

        return output.unsqueeze(1), state

    output = torch.empty_like(value)
    for current_index in range(seq_length):
        current_decay = decay[:, current_index].float()
        current_receptance = receptance[:, current_index].float()
        current_key = key[:, current_index]
        current_value = value[:, current_index]
        current_in_context_state = in_context_state[:, current_index]
        current_in_context_rate = in_context_rate[:, current_index]

        value_key = (current_value.unsqueeze(-1) @ current_key.unsqueeze(-2)).float()
        in_context = (current_in_context_state.unsqueeze(-1) @ current_in_context_rate.unsqueeze(-2)).float()
        state = state * current_decay.unsqueeze(-2) + state @ in_context + value_key
        output[:, current_index] = (state.to(input_dtype) @ current_receptance.to(input_dtype).unsqueeze(-1)).squeeze(
            -1
        )

    if not (return_state or has_initial_state):
        state = None

    return output, state


class Rwkv7SelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.head_size = config.head_size
        if self.hidden_size % self.head_size != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `head_size` ({self.head_size}) for RWKV7."
            )
        self.num_heads = self.hidden_size // self.head_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_r = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.x_w = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.x_k = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.x_v = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.x_a = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.x_g = nn.Parameter(torch.empty(1, 1, self.hidden_size))

        self.w0 = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.w1 = nn.Parameter(torch.empty(self.hidden_size, config.decay_lora_rank))
        self.w2 = nn.Parameter(torch.empty(config.decay_lora_rank, self.hidden_size))

        self.a0 = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.a1 = nn.Parameter(torch.empty(self.hidden_size, config.in_context_learning_lora_rank))
        self.a2 = nn.Parameter(torch.empty(config.in_context_learning_lora_rank, self.hidden_size))

        self.v0 = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.v1 = nn.Parameter(torch.empty(self.hidden_size, config.value_lora_rank))
        self.v2 = nn.Parameter(torch.empty(config.value_lora_rank, self.hidden_size))

        self.g1 = nn.Parameter(torch.empty(self.hidden_size, config.gate_lora_rank))
        self.g2 = nn.Parameter(torch.empty(config.gate_lora_rank, self.hidden_size))

        self.k_k = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.k_a = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        self.r_k = nn.Parameter(torch.empty(self.num_heads, self.head_size))

        self.receptance = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(self.num_heads, self.hidden_size, eps=config.group_norm_epsilon)

    def _shift_hidden(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            return state[0][:, :, self.layer_id].unsqueeze(1)

        shifted = self.time_shift(hidden)
        if state is not None:
            shifted[:, 0] = state[0][:, :, self.layer_id]
        return shifted

    def forward(self, hidden, state=None, use_cache=False, v_first=None):
        batch_size, seq_length, _ = hidden.shape
        shifted = self._shift_hidden(hidden, state=state)
        time_shift = shifted - hidden

        receptance_hidden = hidden + time_shift * self.x_r
        decay_hidden = hidden + time_shift * self.x_w
        key_hidden = hidden + time_shift * self.x_k
        value_hidden = hidden + time_shift * self.x_v
        in_context_hidden = hidden + time_shift * self.x_a
        gate_hidden = hidden + time_shift * self.x_g

        receptance = self.receptance(receptance_hidden)
        raw_decay = self.w0.float() + (torch.tanh(decay_hidden @ self.w1) @ self.w2).float()
        decay = torch.exp(-0.606531 * torch.sigmoid(raw_decay))

        key = self.key(key_hidden)
        value = self.value(value_hidden)
        if self.layer_id == 0:
            v_first = value
        else:
            value_residual = torch.sigmoid(self.v0 + (value_hidden @ self.v1) @ self.v2)
            value = value + (v_first - value) * value_residual

        in_context_rate = torch.sigmoid(self.a0 + (in_context_hidden @ self.a1) @ self.a2)
        gate = torch.sigmoid(gate_hidden @ self.g1) @ self.g2

        normalized_key = F.normalize(
            (key * self.k_k).view(batch_size, seq_length, self.num_heads, self.head_size), dim=-1, p=2.0
        )
        key = key * (1 + (in_context_rate - 1) * self.k_a)

        receptance = receptance.view(batch_size, seq_length, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_size)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_size)
        decay = decay.view(batch_size, seq_length, self.num_heads, self.head_size)
        in_context_rate = in_context_rate.view(batch_size, seq_length, self.num_heads, self.head_size)

        layer_state = state[1][:, self.layer_id] if state is not None else None
        rwkv, layer_state = rwkv7_linear_attention(
            decay,
            receptance,
            key,
            value,
            -normalized_key,
            normalized_key * in_context_rate,
            state=layer_state,
            return_state=use_cache,
        )

        if layer_state is not None:
            state[1][:, self.layer_id] = layer_state
        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        rwkv = rwkv.reshape(batch_size * seq_length, self.hidden_size)
        rwkv = self.ln_x(rwkv).view(batch_size, seq_length, self.num_heads, self.head_size)
        bonus = ((receptance * key * self.r_k).sum(dim=-1, keepdim=True) * value).view(
            batch_size, seq_length, self.hidden_size
        )
        rwkv = rwkv.view(batch_size, seq_length, self.hidden_size) + bonus

        return self.output(rwkv * gate), state, v_first


class Rwkv7FeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.feed_forward_embed_rank = config.feed_forward_embed_rank
        if config.feed_forward_embed_size is not None:
            if config.feed_forward_embed_rank is None:
                raise ValueError("`feed_forward_embed_rank` must be set when `feed_forward_embed_size` is set.")
            if config.feed_forward_embed_size != config.feed_forward_embed_rank**2:
                raise ValueError("`feed_forward_embed_size` must equal `feed_forward_embed_rank ** 2` for RWKV7.")
            self.s_emb = nn.Embedding(config.vocab_size, config.feed_forward_embed_size)
            self.s_emb_x = nn.Linear(hidden_size, config.feed_forward_embed_size, bias=False)
            self.s0 = nn.Parameter(torch.empty(1, 1, intermediate_size))
            self.s1 = nn.Parameter(torch.empty(hidden_size, config.feed_forward_embed_rank))
            self.s2 = nn.Parameter(torch.empty(config.feed_forward_embed_rank, intermediate_size))
        else:
            self.s_emb = None

    def _shift_hidden(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            return state[2][:, :, self.layer_id].unsqueeze(1)

        shifted = self.time_shift(hidden)
        if state is not None:
            shifted[:, 0] = state[2][:, :, self.layer_id]
        return shifted

    def forward(self, hidden, state=None, input_ids=None, token_embeddings=None):
        shifted = self._shift_hidden(hidden, state=state)
        key = hidden + (shifted - hidden) * self.x_k
        key = torch.square(torch.relu(self.key(key)))
        if self.s_emb is not None:
            if input_ids is None or token_embeddings is None:
                raise ValueError("RWKV7 feed-forward token embeddings require `input_ids`.")
            rank = self.feed_forward_embed_rank
            deep_embedding = self.s_emb(input_ids) + self.s_emb_x(token_embeddings)
            scaling = (hidden @ self.s1).view(hidden.size(0), hidden.size(1), 1, rank)
            scaling = scaling @ deep_embedding.view(hidden.size(0), hidden.size(1), rank, rank)
            key = key * ((scaling.view(hidden.size(0), hidden.size(1), rank) @ self.s2) + self.s0)
        value = self.value(key)

        if state is not None:
            state[2][:, :, self.layer_id] = hidden[:, -1]

        return value, state


class Rwkv7Qkv(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        if config.qkv_size is None or config.qkv_rank is None:
            raise ValueError("`qkv_size` and `qkv_rank` must be set for RWKV7 QKV/DEA layers.")

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.qkv_size = config.qkv_size
        self.qkv_rank = config.qkv_rank
        self.qkv_softcap = config.qkv_softcap
        self.qkv_score_scale = (
            config.qkv_score_scale if config.qkv_score_scale is not None else 1 / (4 * self.qkv_size)
        )

        self.qq = nn.Linear(self.hidden_size, self.qkv_size, bias=False)
        self.k1 = nn.Parameter(torch.empty(self.hidden_size, self.qkv_rank))
        self.k2 = nn.Parameter(torch.empty(self.qkv_rank, self.qkv_size))
        self.k_emb = nn.Embedding(config.vocab_size, self.qkv_size)
        self.k_emb_x = nn.Linear(self.hidden_size, self.qkv_size, bias=False)

        self.v1 = nn.Parameter(torch.empty(self.hidden_size, self.qkv_rank))
        self.v2 = nn.Parameter(torch.empty(self.qkv_rank, self.hidden_size))
        self.v_emb = nn.Embedding(config.vocab_size, self.hidden_size)
        self.v_emb_x = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.x_q = nn.Parameter(torch.empty(1, 1, self.qkv_size))
        self.x_k = nn.Parameter(torch.empty(1, 1, self.qkv_size))
        self.x_v = nn.Parameter(torch.empty(1, 1, self.hidden_size))

        self.lnq = nn.LayerNorm(self.qkv_size)
        self.lnk = nn.LayerNorm(self.qkv_size)
        self.lnv = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden, state=None, input_ids=None, context_input_ids=None, context_token_embeddings=None):
        if input_ids is None or context_input_ids is None or context_token_embeddings is None:
            raise ValueError("RWKV7 QKV/DEA layers require `input_ids`.")

        seq_length = hidden.shape[1]
        context_length = context_input_ids.size(1)

        query = self.qq(hidden)
        key_cache = hidden @ self.k1
        value_cache = hidden @ self.v1

        if state is not None:
            state[4][:, self.layer_id, -seq_length:] = key_cache
            state[5][:, self.layer_id, -seq_length:] = value_cache
            key_cache = state[4][:, self.layer_id]
            value_cache = state[5][:, self.layer_id]

        key = (key_cache @ self.k2) * (self.k_emb(context_input_ids) + self.k_emb_x(context_token_embeddings))
        value = torch.tanh(value_cache @ self.v2) * (
            self.v_emb(context_input_ids) + self.v_emb_x(context_token_embeddings)
        )

        if state is not None:
            shifted_query = torch.cat([state[6][:, self.layer_id].unsqueeze(1), query[:, :-1]], dim=1)
            state[6][:, self.layer_id] = query[:, -1]
        else:
            shifted_query = torch.cat([torch.zeros_like(query[:, :1]), query[:, :-1]], dim=1)
        query = query + (shifted_query - query) * self.x_q

        shifted_key = torch.cat([torch.zeros_like(key[:, :1]), key[:, :-1]], dim=1)
        shifted_value = torch.cat([torch.zeros_like(value[:, :1]), value[:, :-1]], dim=1)
        key = key + (shifted_key - key) * self.x_k
        value = value + (shifted_value - value) * self.x_v

        query = self.lnq(query)
        key = self.lnk(key)
        value = self.lnv(value)

        scores = self.qkv_softcap * torch.tanh((query @ key.transpose(-1, -2)) * self.qkv_score_scale)
        if seq_length > 1:
            causal_mask = ~torch.tril(
                torch.ones(context_length, context_length, dtype=torch.bool, device=hidden.device)
            )[-seq_length:]
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        return scores.softmax(dim=-1) @ value, state


class Rwkv7Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = Rwkv7SelfAttention(config, layer_id)
        self.feed_forward = Rwkv7FeedForward(config, layer_id)
        self.qkv = Rwkv7Qkv(config, layer_id) if config.qkv_size is not None else None

    def forward(
        self,
        hidden,
        state=None,
        use_cache=False,
        output_attentions=False,
        v_first=None,
        input_ids=None,
        token_embeddings=None,
        context_input_ids=None,
        context_token_embeddings=None,
    ):
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)

        if self.qkv is not None:
            qkv, state = self.qkv(
                hidden,
                state=state,
                input_ids=input_ids,
                context_input_ids=context_input_ids,
                context_token_embeddings=context_token_embeddings,
            )
        else:
            qkv = None

        attention, state, v_first = self.attention(self.ln1(hidden), state=state, use_cache=use_cache, v_first=v_first)
        hidden = hidden + attention
        if qkv is not None:
            hidden = hidden + qkv

        feed_forward, state = self.feed_forward(
            self.ln2(hidden), state=state, input_ids=input_ids, token_embeddings=token_embeddings
        )
        hidden = hidden + feed_forward

        outputs = (hidden, state)
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        outputs += (v_first,)
        return outputs


@auto_docstring
class Rwkv7PreTrainedModel(PreTrainedModel):
    config: Rwkv7Config
    base_model_prefix = "rwkv7"
    _no_split_modules = ["Rwkv7Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        if isinstance(module, Rwkv7SelfAttention):
            for parameter in [module.x_r, module.x_w, module.x_k, module.x_v, module.x_a, module.x_g]:
                init.zeros_(parameter)
            for parameter in [module.w0, module.w1, module.w2, module.a0, module.a1, module.a2]:
                init.zeros_(parameter)
            for parameter in [module.v0, module.v1, module.v2, module.g1, module.g2, module.r_k]:
                init.zeros_(parameter)
            init.ones_(module.k_k)
            init.ones_(module.k_a)
        elif isinstance(module, Rwkv7FeedForward):
            init.zeros_(module.x_k)
            if module.s_emb is not None:
                init.ones_(module.s0)
                init.zeros_(module.s1)
                init.zeros_(module.s2)
        elif isinstance(module, Rwkv7Qkv):
            for parameter in [module.k1, module.k2, module.v1, module.v2]:
                init.orthogonal_(parameter)
            for parameter in [module.x_q, module.x_k, module.x_v]:
                init.zeros_(parameter)
        elif isinstance(module, nn.Linear):
            shape = module.weight.shape
            gain = 1.0
            scale = 1.0
            if module.bias is not None:
                init.zeros_(module.bias)
            if shape[0] > shape[1]:
                gain = (shape[0] / shape[1]) ** 0.5
            if shape[0] == self.config.vocab_size and shape[1] == self.config.hidden_size:
                scale = 0.5
            init.orthogonal_(module.weight, gain=gain * scale)
        elif isinstance(module, nn.Embedding):
            shape = module.weight.shape
            init.orthogonal_(module.weight, gain=1e-4 * max(shape[0], shape[1]) ** 0.5)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)


@auto_docstring(
    custom_intro="""
    Class for the RWKV7 model outputs.
    """
)
@dataclass
class Rwkv7Output(ModelOutput):
    r"""
    state (list of `torch.FloatTensor`):
        The state of the model at the last time step. It can be passed back to the next forward call.
        Standard RWKV7 checkpoints use three state tensors. RWKV7 checkpoints with QKV/DEA layers use seven state
        tensors.
    """

    last_hidden_state: torch.FloatTensor | None = None
    state: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None

    @property
    def past_key_values(self):
        return self.state


@auto_docstring(
    custom_intro="""
    Base class for RWKV7 causal language model outputs.
    """
)
@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head before SoftMax.
    state (list of `torch.FloatTensor`):
        The state of the model at the last time step. It can be passed back to the next forward call.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    state: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None

    @property
    def past_key_values(self):
        return self.state


@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        logger.warning_once(
            "RWKV7 is currently using the plain Python implementation for token mixing. "
            "This implementation is numerically aligned with the RWKV-LM reference, but it may be very slow, "
            "especially for long sequences."
        )

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv7Block(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @staticmethod
    def _slice_state(state, batch_index):
        if state is None:
            return None
        return [state_tensor[batch_index : batch_index + 1].clone() for state_tensor in state]

    @staticmethod
    def _concat_states(states):
        if not states or states[0] is None:
            return None
        return [torch.cat([state[index] for state in states], dim=0) for index in range(len(states[0]))]

    def _forward_eval_batch_samplewise(
        self,
        input_ids=None,
        state=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        sample_outputs = []
        for batch_index in range(input_ids.size(0)):
            sample_outputs.append(
                self._forward_recurrent_eval(
                    input_ids=input_ids[batch_index : batch_index + 1],
                    state=self._slice_state(state, batch_index),
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
            )

        last_hidden_state = torch.cat([output.last_hidden_state for output in sample_outputs], dim=0)
        output_state = self._concat_states([output.state for output in sample_outputs])
        hidden_states = (
            tuple(
                torch.cat([output.hidden_states[layer_index] for output in sample_outputs], dim=0)
                for layer_index in range(len(sample_outputs[0].hidden_states))
            )
            if output_hidden_states
            else None
        )
        attentions = (
            tuple(
                torch.cat([output.attentions[layer_index] for output in sample_outputs], dim=0)
                for layer_index in range(len(sample_outputs[0].attentions))
            )
            if output_attentions
            else None
        )

        if not return_dict:
            return tuple(x for x in [last_hidden_state, output_state, hidden_states, attentions] if x is not None)

        return Rwkv7Output(
            last_hidden_state=last_hidden_state,
            state=output_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def _forward_recurrent_eval(
        self,
        input_ids=None,
        state=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        has_initial_state = state is not None
        last_hidden_states = []
        step_hidden_states = None
        step_attentions = None

        for token_index in range(input_ids.size(1)):
            step_outputs = self.forward(
                input_ids=input_ids[:, token_index : token_index + 1],
                state=state,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            state = step_outputs.state
            last_hidden_states.append(step_outputs.last_hidden_state)

            if output_hidden_states:
                if step_hidden_states is None:
                    step_hidden_states = [[] for _ in range(len(step_outputs.hidden_states))]
                for layer_index, hidden_state in enumerate(step_outputs.hidden_states):
                    step_hidden_states[layer_index].append(hidden_state)

            if output_attentions:
                if step_attentions is None:
                    step_attentions = [[] for _ in range(len(step_outputs.attentions))]
                for layer_index, attention in enumerate(step_outputs.attentions):
                    step_attentions[layer_index].append(attention)

        last_hidden_state = torch.cat(last_hidden_states, dim=1)
        hidden_states = (
            tuple(torch.cat(layer_hidden_states, dim=1) for layer_hidden_states in step_hidden_states)
            if step_hidden_states is not None
            else None
        )
        attentions = (
            tuple(torch.cat(layer_attentions, dim=1) for layer_attentions in step_attentions)
            if step_attentions is not None
            else None
        )
        output_state = state if use_cache or has_initial_state else None

        if not return_dict:
            return tuple(x for x in [last_hidden_state, output_state, hidden_states, attentions] if x is not None)

        return Rwkv7Output(
            last_hidden_state=last_hidden_state,
            state=output_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | Rwkv7Output:
        r"""
        state (list of `torch.FloatTensor`, *optional*):
            Previous RWKV7 state. Standard checkpoints use three state tensors. QKV/DEA checkpoints use seven state
            tensors.
        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to generate the next logits.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is not None:
            logger.warning_once("`attention_mask` was passed, but it is unused in this model.")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if not self.training and input_ids is not None and inputs_embeds is None and input_ids.size(0) > 1:
            return self._forward_eval_batch_samplewise(
                input_ids=input_ids,
                state=state,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if not self.training and input_ids is not None and inputs_embeds is None and input_ids.size(1) > 1:
            return self._forward_recurrent_eval(
                input_ids=input_ids,
                state=state,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        has_deep_embed = self.config.feed_forward_embed_size is not None
        has_qkv = self.config.qkv_size is not None

        token_embeddings = None
        context_input_ids = None
        context_token_embeddings = None
        if has_deep_embed or has_qkv:
            if input_ids is None:
                raise ValueError("`input_ids` are required for RWKV7 checkpoints with token-dependent layers.")
            token_embeddings = self.blocks[0].pre_ln(inputs_embeds)

        if use_cache and state is None:
            state = [
                torch.zeros(
                    inputs_embeds.size(0),
                    self.config.hidden_size,
                    self.config.num_hidden_layers,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                ),
                torch.zeros(
                    inputs_embeds.size(0),
                    self.config.num_hidden_layers,
                    self.config.hidden_size // self.config.head_size,
                    self.config.head_size,
                    self.config.head_size,
                    dtype=torch.float32,
                    device=inputs_embeds.device,
                ),
                torch.zeros(
                    inputs_embeds.size(0),
                    self.config.hidden_size,
                    self.config.num_hidden_layers,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                ),
            ]
            if self.config.qkv_size is not None:
                state.extend(
                    [
                        torch.empty(
                            inputs_embeds.size(0),
                            0,
                            dtype=input_ids.dtype,
                            device=inputs_embeds.device,
                        ),
                        torch.empty(
                            inputs_embeds.size(0),
                            self.config.num_hidden_layers,
                            0,
                            self.config.qkv_rank,
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        ),
                        torch.empty(
                            inputs_embeds.size(0),
                            self.config.num_hidden_layers,
                            0,
                            self.config.qkv_rank,
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        ),
                        torch.zeros(
                            inputs_embeds.size(0),
                            self.config.num_hidden_layers,
                            self.config.qkv_size,
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device,
                        ),
                    ]
                )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if has_qkv:
            if state is not None:
                if len(state) != 7:
                    raise ValueError("RWKV7 QKV/DEA checkpoints require a state with 7 tensors.")
                past_input_ids = state[3]
                past_length = past_input_ids.size(1)
                if state[4].size(2) != past_length or state[5].size(2) != past_length:
                    raise ValueError("RWKV7 QKV/DEA state caches have inconsistent sequence lengths.")

                context_input_ids = torch.cat([past_input_ids, input_ids], dim=1)
                state[3] = context_input_ids
                empty_shape = (
                    inputs_embeds.size(0),
                    self.config.num_hidden_layers,
                    inputs_embeds.size(1),
                    self.config.qkv_rank,
                )
                state[4] = torch.cat(
                    [
                        state[4],
                        torch.empty(*empty_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device),
                    ],
                    dim=2,
                )
                state[5] = torch.cat(
                    [
                        state[5],
                        torch.empty(*empty_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device),
                    ],
                    dim=2,
                )
            else:
                context_input_ids = input_ids

            context_token_embeddings = self.blocks[0].pre_ln(self.embeddings(context_input_ids))
            token_embeddings = context_token_embeddings[:, -inputs_embeds.size(1) :]

        hidden_states = inputs_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        v_first = None
        for block in self.blocks:
            hidden_states, state, attentions, v_first = block(
                hidden_states,
                state=state,
                use_cache=use_cache,
                output_attentions=output_attentions,
                v_first=v_first,
                input_ids=input_ids,
                token_embeddings=token_embeddings,
                context_input_ids=context_input_ids,
                context_token_embeddings=context_token_embeddings,
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(x for x in [hidden_states, state, all_hidden_states, all_self_attentions] if x is not None)

        return Rwkv7Output(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring(
    custom_intro="""
    The RWKV7 Model with a language modeling head on top.
    """
)
class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"head.weight": "rwkv7.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.rwkv7 = Rwkv7Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | Rwkv7CausalLMOutput:
        r"""
        state (list of `torch.FloatTensor`, *optional*):
            Previous RWKV7 state.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. The labels are shifted inside the model.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        rwkv7_outputs = self.rwkv7(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = rwkv7_outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + rwkv7_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Rwkv7CausalLMOutput(
            loss=loss,
            logits=logits,
            state=rwkv7_outputs.state,
            hidden_states=rwkv7_outputs.hidden_states,
            attentions=rwkv7_outputs.attentions,
        )


__all__ = ["Rwkv7ForCausalLM", "Rwkv7Model", "Rwkv7PreTrainedModel"]
