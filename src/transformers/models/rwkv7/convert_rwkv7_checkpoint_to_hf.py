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
"""Convert a RWKV7 checkpoint from BlinkDL to the Hugging Face format."""

import argparse
import ast
import gc
import inspect
import json
import os
import re

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from transformers import AutoModelForCausalLM, Rwkv7Config, Rwkv7Tokenizer
from transformers.utils import SAFE_WEIGHTS_NAME


CHAT_TEMPLATE = """{%- for message in messages -%}
{%- set content = message['content'] | trim | replace('\\n\\n', '\\n') -%}
{%- if message['role'] == 'system' -%}System: {{ content }}{%- elif message['role'] == 'user' -%}User: {{ content }}{%- elif message['role'] == 'assistant' -%}Assistant: {{ content }}{%- else -%}{{ raise_exception('Unsupported role: ' + message['role']) }}{%- endif -%}{%- if not loop.last -%}{{ '\\n\\n' }}{%- endif -%}
{%- endfor -%}{%- if add_generation_prompt -%}{%- if messages|length > 0 -%}{{ '\\n\\n' }}{%- endif -%}Assistant:{%- endif -%}"""


def load_checkpoint_state_dict(checkpoint_file, map_location="cpu"):
    load_kwargs = {"map_location": map_location, "weights_only": True}
    use_mmap = (
        os.environ.get("TRANSFORMERS_RWKV_DISABLE_MMAP") != "1"
        and isinstance(checkpoint_file, (str, os.PathLike))
        and "mmap" in inspect.signature(torch.load).parameters
    )
    if use_mmap:
        load_kwargs["mmap"] = True
    try:
        return torch.load(checkpoint_file, **load_kwargs)
    except RuntimeError as error:
        if use_mmap and "unable to mmap" in str(error):
            raise RuntimeError(
                "Could not memory-map the checkpoint. The checkpoint may be larger than the system memory commit "
                "limit. Add RAM/swap, or set `TRANSFORMERS_RWKV_DISABLE_MMAP=1` to try regular `torch.load` on a "
                "machine with enough memory."
            ) from error
        raise


def convert_state_dict_rwkv7(state_dict):
    converted_state_dict = {}
    for name, weight in state_dict.items():
        if name.startswith("emb."):
            name = name.replace("emb.", "embeddings.")
        if name.startswith("blocks.0.ln0"):
            name = name.replace("blocks.0.ln0", "blocks.0.pre_ln")

        name = re.sub(r"blocks\.(\d+)\.att", r"blocks.\1.attention", name)
        name = re.sub(r"blocks\.(\d+)\.ffn", r"blocks.\1.feed_forward", name)

        if name != "head.weight":
            name = "rwkv7." + name

        converted_state_dict[name] = weight.contiguous()

    return converted_state_dict


def infer_rwkv7_config(state_dict, checkpoint_file, context_length=None):
    check_supported_rwkv7_state_dict(state_dict)

    hidden_size = state_dict["emb.weight"].shape[1]
    vocab_size = state_dict["emb.weight"].shape[0]
    num_hidden_layers = 1 + max(
        int(match.group(1)) for key in state_dict if (match := re.match(r"blocks\.(\d+)\.", key))
    )
    head_size = state_dict["blocks.0.att.r_k"].shape[1]

    if context_length is None:
        context_match = re.search(r"ctx(\d+)", checkpoint_file)
        if context_match is None:
            raise ValueError(
                "Could not infer `context_length` from the checkpoint filename. "
                "Please pass `--context_length` explicitly."
            )
        context_length = int(context_match.group(1))

    feed_forward_embed_size = None
    feed_forward_embed_rank = None
    if "blocks.0.ffn.s_emb.weight" in state_dict:
        feed_forward_embed_size = state_dict["blocks.0.ffn.s_emb.weight"].shape[1]
        feed_forward_embed_rank = state_dict["blocks.0.ffn.s1"].shape[1]

    qkv_size = None
    qkv_rank = None
    if "blocks.0.qkv.qq.weight" in state_dict:
        qkv_size = state_dict["blocks.0.qkv.qq.weight"].shape[0]
        qkv_rank = state_dict["blocks.0.qkv.k1"].shape[1]

    config = Rwkv7Config(
        vocab_size=vocab_size,
        context_length=context_length,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=state_dict["blocks.0.ffn.key.weight"].shape[0],
        head_size=head_size,
        decay_lora_rank=state_dict["blocks.0.att.w1"].shape[1],
        in_context_learning_lora_rank=state_dict["blocks.0.att.a1"].shape[1],
        value_lora_rank=state_dict["blocks.0.att.v1"].shape[1],
        gate_lora_rank=state_dict["blocks.0.att.g1"].shape[1],
        feed_forward_embed_size=feed_forward_embed_size,
        feed_forward_embed_rank=feed_forward_embed_rank,
        qkv_size=qkv_size,
        qkv_rank=qkv_rank,
        group_norm_epsilon=64e-5,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config.architectures = ["Rwkv7ForCausalLM"]
    return config


def check_supported_rwkv7_state_dict(state_dict):
    layer_ids = sorted({int(match.group(1)) for key in state_dict if (match := re.match(r"blocks\.(\d+)\.", key))})
    qkv_layers = {int(match.group(1)) for key in state_dict if (match := re.match(r"blocks\.(\d+)\.qkv\.", key))}
    if not qkv_layers:
        return

    if qkv_layers != set(layer_ids):
        raise ValueError("RWKV7 QKV/DEA weights must be present in every layer or no layer.")

    required_keys = {
        "k1",
        "k2",
        "k_emb.weight",
        "k_emb_x.weight",
        "lnk.bias",
        "lnk.weight",
        "lnq.bias",
        "lnq.weight",
        "lnv.bias",
        "lnv.weight",
        "qq.weight",
        "v1",
        "v2",
        "v_emb.weight",
        "v_emb_x.weight",
        "x_k",
        "x_q",
        "x_v",
    }
    for layer_id in layer_ids:
        prefix = f"blocks.{layer_id}.qkv."
        missing_keys = [key for key in sorted(required_keys) if prefix + key not in state_dict]
        if missing_keys:
            raise ValueError(f"Missing RWKV7 QKV/DEA weights for layer {layer_id}: {missing_keys}")

    qkv_size = state_dict["blocks.0.qkv.qq.weight"].shape[0]
    qkv_rank = state_dict["blocks.0.qkv.k1"].shape[1]
    hidden_size = state_dict["emb.weight"].shape[1]
    for layer_id in layer_ids:
        prefix = f"blocks.{layer_id}.qkv."
        expected_shapes = {
            "qq.weight": (qkv_size, hidden_size),
            "k1": (hidden_size, qkv_rank),
            "k2": (qkv_rank, qkv_size),
            "k_emb.weight": (state_dict["emb.weight"].shape[0], qkv_size),
            "k_emb_x.weight": (qkv_size, hidden_size),
            "v1": (hidden_size, qkv_rank),
            "v2": (qkv_rank, hidden_size),
            "v_emb.weight": (state_dict["emb.weight"].shape[0], hidden_size),
            "v_emb_x.weight": (hidden_size, hidden_size),
            "x_q": (1, 1, qkv_size),
            "x_k": (1, 1, qkv_size),
            "x_v": (1, 1, hidden_size),
            "lnq.weight": (qkv_size,),
            "lnq.bias": (qkv_size,),
            "lnk.weight": (qkv_size,),
            "lnk.bias": (qkv_size,),
            "lnv.weight": (hidden_size,),
            "lnv.bias": (hidden_size,),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = state_dict[prefix + name].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Unexpected RWKV7 QKV/DEA shape for `{prefix + name}`: "
                    f"expected {expected_shape}, got {actual_shape}."
                )


def convert_rwkv_vocab_to_json(vocab_file, output_dir):
    vocab = {}
    with open(vocab_file, encoding="utf-8") as vocab_handle:
        for line in vocab_handle:
            idx = int(line[: line.index(" ")])
            token = ast.literal_eval(line[line.index(" ") : line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            if not isinstance(token, bytes):
                raise ValueError(f"Invalid RWKV token at index {idx}: expected bytes or str, got {type(token)}")
            if len(token) != int(line[line.rindex(" ") :]):
                raise ValueError(f"Invalid RWKV token length at index {idx}.")
            vocab[token.decode("latin-1")] = idx

    output_vocab_file = os.path.join(output_dir, "vocab.json")
    with open(output_vocab_file, "w", encoding="utf-8") as vocab_handle:
        json.dump(vocab, vocab_handle, ensure_ascii=False, indent=2)
        vocab_handle.write("\n")

    return output_vocab_file


def convert_rwkv7_checkpoint_to_hf_format(
    repo_id, checkpoint_file, output_dir, vocab_file, context_length=None, push_to_hub=False, model_name=None
):
    if vocab_file is None:
        raise ValueError("RWKV7 conversion requires `--vocab_file` with rwkv_vocab_v20230424.txt.")

    os.makedirs(output_dir, exist_ok=True)

    model_file = checkpoint_file if os.path.isfile(checkpoint_file) else hf_hub_download(repo_id, checkpoint_file)
    state_dict = load_checkpoint_state_dict(model_file, map_location="cpu")
    config = infer_rwkv7_config(state_dict, checkpoint_file, context_length=context_length)
    config.save_pretrained(output_dir)

    vocab_file = convert_rwkv_vocab_to_json(vocab_file, output_dir)
    tokenizer = Rwkv7Tokenizer(vocab_file, vocab_size=config.vocab_size, chat_template=CHAT_TEMPLATE)
    tokenizer.save_pretrained(output_dir)

    state_dict = convert_state_dict_rwkv7(state_dict)
    save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})

    del state_dict
    gc.collect()

    if push_to_hub:
        if model_name is None:
            raise ValueError("Please provide a `model_name` to push the model to the Hub.")
        model = AutoModelForCausalLM.from_pretrained(output_dir, use_safetensors=True)
        model.push_to_hub(model_name, max_shard_size="2GB")
        tokenizer.push_to_hub(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, type=str, help="Repo ID from which to pull the checkpoint.")
    parser.add_argument("--checkpoint_file", required=True, type=str, help="Name of the checkpoint file in the repo.")
    parser.add_argument("--output_dir", required=True, type=str, help="Where to save the converted model.")
    parser.add_argument(
        "--vocab_file", required=True, type=str, help="Path to rwkv_vocab_v20230424.txt for RWKV7 conversion."
    )
    parser.add_argument(
        "--context_length",
        default=None,
        type=int,
        help="RWKV7 context length. If omitted, it is inferred from a `ctx...` checkpoint filename.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push the converted model to the Hub.")
    parser.add_argument("--model_name", default=None, type=str, help="Repo ID to push to.")

    args = parser.parse_args()
    convert_rwkv7_checkpoint_to_hf_format(
        args.repo_id,
        args.checkpoint_file,
        args.output_dir,
        vocab_file=args.vocab_file,
        context_length=args.context_length,
        push_to_hub=args.push_to_hub,
        model_name=args.model_name,
    )
