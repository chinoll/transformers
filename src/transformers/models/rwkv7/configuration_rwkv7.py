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
"""RWKV7 configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BlinkDL/rwkv7-g1")
@strict
class Rwkv7Config(PreTrainedConfig):
    r"""
    vocab_size (`int`, *optional*, defaults to 65536):
        Vocabulary size of the RWKV7 model.
    context_length (`int`, *optional*, defaults to 8192):
        Maximum sequence length used by the checkpoint.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the hidden representations.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of RWKV7 blocks.
    intermediate_size (`int`, *optional*):
        Dimensionality of the feed-forward hidden layer. Defaults to `4 * hidden_size`.
    head_size (`int`, *optional*, defaults to 64):
        Per-head hidden size used by token mixing.
    decay_lora_rank (`int`, *optional*):
        Rank of the time-decay LoRA projection.
    in_context_learning_lora_rank (`int`, *optional*):
        Rank of the in-context learning-rate LoRA projection.
    value_lora_rank (`int`, *optional*):
        Rank of the value-residual LoRA projection.
    gate_lora_rank (`int`, *optional*):
        Rank of the gate LoRA projection.
    feed_forward_embed_size (`int`, *optional*):
        Size of the optional token-dependent feed-forward embedding.
    feed_forward_embed_rank (`int`, *optional*):
        Rank used by the optional token-dependent feed-forward projection.
    qkv_size (`int`, *optional*):
        Query/key hidden size used by optional QKV/DEA layers.
    qkv_rank (`int`, *optional*):
        Intermediate cache size used by optional QKV/DEA layers.
    qkv_softcap (`float`, *optional*, defaults to 64.0):
        Soft cap multiplier used by optional QKV/DEA layers.
    qkv_score_scale (`float`, *optional*):
        Scale applied before the QKV/DEA soft cap. Defaults to `1 / (4 * qkv_size)` when `qkv_size` is set.
    group_norm_epsilon (`float`, *optional*, defaults to 0.00064):
        Epsilon used by the per-head group normalization.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
        Epsilon used by layer normalization.

    ```python
    >>> from transformers import Rwkv7Config, Rwkv7Model

    >>> configuration = Rwkv7Config()
    >>> model = Rwkv7Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "rwkv7"
    attribute_map = {"max_position_embeddings": "context_length"}

    vocab_size: int = 65536
    context_length: int = 8192
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int | None = None
    head_size: int = 64
    decay_lora_rank: int | None = None
    in_context_learning_lora_rank: int | None = None
    value_lora_rank: int | None = None
    gate_lora_rank: int | None = None
    feed_forward_embed_size: int | None = None
    feed_forward_embed_rank: int | None = None
    qkv_size: int | None = None
    qkv_rank: int | None = None
    qkv_softcap: float = 64.0
    qkv_score_scale: float | None = None
    group_norm_epsilon: float = 64e-5
    layer_norm_epsilon: float = 1e-5
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        self.intermediate_size = self.intermediate_size if self.intermediate_size is not None else 4 * self.hidden_size
        if self.decay_lora_rank is None:
            self.decay_lora_rank = max(32, round(2.5 * self.hidden_size**0.5 / 32) * 32)
        if self.in_context_learning_lora_rank is None:
            self.in_context_learning_lora_rank = max(32, round(2.5 * self.hidden_size**0.5 / 32) * 32)
        if self.value_lora_rank is None:
            self.value_lora_rank = max(32, round(1.7 * self.hidden_size**0.5 / 32) * 32)
        if self.gate_lora_rank is None:
            self.gate_lora_rank = max(32, round(5 * self.hidden_size**0.5 / 32) * 32)
        if self.qkv_size is not None and self.qkv_rank is None:
            raise ValueError("`qkv_rank` must be set when `qkv_size` is set.")
        if self.qkv_size is not None and self.qkv_score_scale is None:
            self.qkv_score_scale = 1 / (4 * self.qkv_size)

        super().__post_init__(**kwargs)


__all__ = ["Rwkv7Config"]
