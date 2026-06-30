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

import json
import os
import re
import tempfile
import unittest

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Rwkv7Config,
    Rwkv7Tokenizer,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import Rwkv7ForCausalLM, Rwkv7Model
    from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import (
        check_supported_rwkv7_state_dict,
        convert_rwkv_vocab_to_json,
        convert_state_dict_rwkv7,
        infer_rwkv7_config,
    )
    from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_linear_attention


@require_torch
class Rwkv7ModelTest(unittest.TestCase):
    all_model_classes = (Rwkv7Model, Rwkv7ForCausalLM) if is_torch_available() else ()

    def setUp(self):
        self.batch_size = 3
        self.seq_length = 5
        self.config_tester = ConfigTester(
            self, config_class=Rwkv7Config, n_embd=32, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def get_config(self):
        return Rwkv7Config(
            vocab_size=99,
            context_length=16,
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            head_size=8,
            decay_lora_rank=16,
            in_context_learning_lora_rank=16,
            value_lora_rank=8,
            gate_lora_rank=16,
            bos_token_id=98,
            eos_token_id=98,
            pad_token_id=98,
        )

    def get_deep_embed_config(self):
        config = self.get_config()
        config.feed_forward_embed_rank = 4
        config.feed_forward_embed_size = 16
        return config

    def get_qkv_config(self):
        config = self.get_deep_embed_config()
        config.qkv_size = 16
        config.qkv_rank = 4
        config.qkv_score_scale = 1 / 64
        return config

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        model = Rwkv7Model(config=config).to(torch_device).eval()

        result = model(input_ids)

        self.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, config.hidden_size))
        self.assertIs(result.past_key_values, result.state)
        self.assertEqual(len(result.state), 3)
        self.assertEqual(
            result.state[1].shape,
            (
                self.batch_size,
                config.num_hidden_layers,
                config.hidden_size // config.head_size,
                config.head_size,
                config.head_size,
            ),
        )

    def test_lm_head_model(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()

        result = model(input_ids, labels=input_ids)

        self.assertEqual(result.loss.shape, ())
        self.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))
        self.assertIs(result.past_key_values, result.state)

    def test_state_equivalency(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        model = Rwkv7Model(config=config).to(torch_device).eval()

        output_whole = model(input_ids).last_hidden_state
        outputs = model(input_ids[:, :2])
        output_one = outputs.last_hidden_state
        output_two = model(input_ids[:, 2:], state=outputs.state).last_hidden_state

        self.assertTrue(torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5))

    def test_eval_batch_matches_single_samples(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size).to(torch_device)
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            batch_logits = model(input_ids).logits
            single_logits = torch.cat([model(input_ids[index : index + 1]).logits for index in range(self.batch_size)])

        self.assertTrue(torch.equal(batch_logits, single_logits))

    def test_training(self):
        config = self.get_config()
        input_ids = ids_tensor([2, self.seq_length], config.vocab_size).to(torch_device)
        model = Rwkv7ForCausalLM(config).to(torch_device).train()

        loss = model(input_ids, labels=input_ids).loss
        loss.backward()

        self.assertIsNotNone(model.rwkv7.embeddings.weight.grad)

    def test_deep_embed_model(self):
        config = self.get_deep_embed_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()

        result = model(input_ids)

        self.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))
        outputs = model(input_ids[:, :2])
        output_two = model(input_ids[:, 2:], state=outputs.state).logits
        self.assertTrue(torch.allclose(torch.cat([outputs.logits, output_two], dim=1), result.logits, atol=1e-5))

    def test_qkv_model(self):
        config = self.get_qkv_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()

        result = model(input_ids)

        self.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))
        self.assertEqual(len(result.state), 7)
        self.assertEqual(result.state[3].shape, (self.batch_size, self.seq_length))
        self.assertEqual(
            result.state[4].shape, (self.batch_size, config.num_hidden_layers, self.seq_length, config.qkv_rank)
        )
        self.assertEqual(result.state[6].shape, (self.batch_size, config.num_hidden_layers, config.qkv_size))

    def test_qkv_state_equivalency(self):
        config = self.get_qkv_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()

        result = model(input_ids).logits
        outputs = model(input_ids[:, :2])
        output_two = model(input_ids[:, 2:], state=outputs.state).logits

        self.assertTrue(torch.allclose(torch.cat([outputs.logits, output_two], dim=1), result, atol=1e-5))

    def test_save_load_safetensors_and_auto(self):
        config = self.get_config()
        input_ids = ids_tensor([2, self.seq_length], config.vocab_size).to(torch_device)
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model.safetensors")))

            reloaded = Rwkv7ForCausalLM.from_pretrained(tmp_dir, use_safetensors=True).to(torch_device).eval()
            auto_config = AutoConfig.from_pretrained(tmp_dir)
            auto_model = AutoModelForCausalLM.from_pretrained(tmp_dir, use_safetensors=True).to(torch_device).eval()
            with torch.no_grad():
                expected = model(input_ids).logits
                actual = reloaded(input_ids).logits
                auto_actual = auto_model(input_ids).logits

        self.assertIsInstance(auto_config, Rwkv7Config)
        self.assertIsInstance(auto_model, Rwkv7ForCausalLM)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
        self.assertTrue(torch.allclose(auto_actual, expected, atol=1e-5))

    def test_converter_key_mapping_and_config(self):
        state_dict = self.get_rwkv7_state_dict(include_deep_embed=True, include_qkv=True)

        config = infer_rwkv7_config(state_dict, "rwkv7-g1d-0.1b-20260129-ctx8192.pth")
        self.assertEqual(config.model_type, "rwkv7")
        self.assertEqual(config.architectures, ["Rwkv7ForCausalLM"])
        self.assertEqual(config.vocab_size, 64)
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.context_length, 8192)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.head_size, 4)
        self.assertEqual(config.decay_lora_rank, 6)
        self.assertEqual(config.in_context_learning_lora_rank, 5)
        self.assertEqual(config.value_lora_rank, 3)
        self.assertEqual(config.gate_lora_rank, 7)
        self.assertEqual(config.feed_forward_embed_size, 16)
        self.assertEqual(config.feed_forward_embed_rank, 4)
        self.assertEqual(config.qkv_size, 8)
        self.assertEqual(config.qkv_rank, 3)
        self.assertEqual(config.qkv_score_scale, 1 / 32)

        config = infer_rwkv7_config(state_dict, "rwkv7-custom.pth", context_length=2048)
        self.assertEqual(config.context_length, 2048)
        with self.assertRaises(ValueError):
            infer_rwkv7_config(state_dict, "rwkv7-custom.pth")

        converted = convert_state_dict_rwkv7(state_dict)
        expected_keys = {
            "rwkv7.embeddings.weight",
            "rwkv7.blocks.0.pre_ln.weight",
            "rwkv7.blocks.0.attention.x_r",
            "rwkv7.blocks.0.attention.w1",
            "rwkv7.blocks.0.attention.ln_x.weight",
            "rwkv7.blocks.0.qkv.qq.weight",
            "rwkv7.blocks.0.qkv.k1",
            "rwkv7.blocks.0.qkv.v_emb_x.weight",
            "rwkv7.blocks.1.feed_forward.x_k",
            "rwkv7.blocks.1.feed_forward.s_emb.weight",
            "rwkv7.blocks.1.feed_forward.s_emb_x.weight",
            "rwkv7.ln_out.weight",
            "head.weight",
        }
        self.assertTrue(expected_keys.issubset(converted.keys()))
        self.assertNotIn("rwkv7.head.weight", converted)

        check_supported_rwkv7_state_dict(state_dict)
        missing_qkv_state_dict = dict(state_dict)
        missing_qkv_state_dict.pop("blocks.0.qkv.k1")
        with self.assertRaises(ValueError):
            check_supported_rwkv7_state_dict(missing_qkv_state_dict)

    def test_linear_attention_matches_reference_kernel(self):
        batch_size = 2
        seq_length = 3
        num_heads = 2
        head_size = 4
        generator = torch.Generator().manual_seed(0)
        decay = torch.rand(batch_size, seq_length, num_heads, head_size, generator=generator)
        receptance = torch.randn(batch_size, seq_length, num_heads, head_size, generator=generator)
        key = torch.randn(batch_size, seq_length, num_heads, head_size, generator=generator)
        value = torch.randn(batch_size, seq_length, num_heads, head_size, generator=generator)
        in_context_state = torch.randn(batch_size, seq_length, num_heads, head_size, generator=generator)
        in_context_rate = torch.randn(batch_size, seq_length, num_heads, head_size, generator=generator)
        state = torch.randn(batch_size, num_heads, head_size, head_size, generator=generator)

        expected, expected_state = self.reference_rwkv7_linear_attention(
            decay, receptance, key, value, in_context_state, in_context_rate, state=state
        )
        actual, actual_state = rwkv7_linear_attention(
            decay,
            receptance,
            key,
            value,
            in_context_state,
            in_context_rate,
            state=state.clone(),
            return_state=True,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-6, rtol=1e-6))

    def test_converted_module_and_logits_match_reference(self):
        state_dict = self.get_random_rwkv7_state_dict()
        config = infer_rwkv7_config(state_dict, "rwkv7-g1d-0.1b-20260129-ctx8192.pth")
        model = Rwkv7ForCausalLM(config).eval()

        missing_keys, unexpected_keys = model.load_state_dict(convert_state_dict_rwkv7(state_dict), strict=False)
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])

        input_ids = torch.tensor([[1, 5, 9, 2]])
        with torch.no_grad():
            hidden = model.rwkv7.embeddings(input_ids)
            hidden = model.rwkv7.blocks[0].pre_ln(hidden)
            hidden = model.rwkv7.blocks[0].ln1(hidden)

            actual_attention = model.rwkv7.blocks[0].attention(hidden, use_cache=False)[0]
            expected_attention = self.reference_rwkv7_attention(hidden, state_dict, layer_id=0)[0]
            actual_logits = model(input_ids).logits
            expected_logits = self.reference_rwkv7_logits(input_ids, state_dict)

        self.assertTrue(torch.allclose(actual_attention, expected_attention, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(actual_logits, expected_logits, atol=1e-5, rtol=1e-5))

    def test_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, "vocab.json")
            with open(vocab_file, "w", encoding="utf-8") as vocab_handle:
                json.dump({"H": 1, "e": 2, "l": 3, "o": 4, " ": 5, "Hello": 6}, vocab_handle)

            chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}"
            tokenizer = Rwkv7Tokenizer(vocab_file, vocab_size=16, chat_template=chat_template)
            self.assertEqual(len(tokenizer), 16)
            self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
            expected_input_ids = [6, 5]
            self.assertEqual(tokenizer.encode("Hello ", add_special_tokens=False), expected_input_ids)
            self.assertEqual(tokenizer.decode(expected_input_ids), "Hello ")
            expected_chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=False
            )

            tokenizer.save_pretrained(tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "vocab.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "tokenizer.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "merges.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "chat_template.jinja")))
            with open(os.path.join(tmp_dir, "tokenizer.json"), encoding="utf-8") as tokenizer_handle:
                self.assertIn("added_tokens", json.load(tokenizer_handle))
            tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertIsInstance(tokenizer, Rwkv7Tokenizer)
            self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
            self.assertEqual(tokenizer.encode("Hello ", add_special_tokens=False), expected_input_ids)
            self.assertEqual(tokenizer.decode(expected_input_ids), "Hello ")
            actual_chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=False
            )
            self.assertEqual(actual_chat, expected_chat)

            with self.assertRaises(ValueError):
                Rwkv7Tokenizer(os.path.join(tmp_dir, "rwkv_vocab_v20230424.txt"), vocab_size=16)

    def test_convert_rwkv_vocab_to_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rwkv_vocab_file = os.path.join(tmp_dir, "rwkv_vocab_v20230424.txt")
            with open(rwkv_vocab_file, "w", encoding="utf-8") as vocab_handle:
                vocab_handle.write("1 'H' 1\n")
                vocab_handle.write("2 'e' 1\n")
                vocab_handle.write("3 'l' 1\n")
                vocab_handle.write("4 'o' 1\n")
                vocab_handle.write("5 ' ' 1\n")
                vocab_handle.write("6 'Hello' 5\n")

            vocab_file = convert_rwkv_vocab_to_json(rwkv_vocab_file, tmp_dir)
            self.assertEqual(os.path.basename(vocab_file), "vocab.json")
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                vocab = json.load(vocab_handle)

            self.assertEqual(vocab["Hello"], 6)
            tokenizer = Rwkv7Tokenizer(vocab_file, vocab_size=16)
            self.assertEqual(tokenizer.encode("Hello ", add_special_tokens=False), [6, 5])

    def reference_rwkv7_linear_attention(
        self, decay, receptance, key, value, in_context_state, in_context_rate, state=None
    ):
        batch_size, seq_length, num_heads, head_size = key.shape
        if state is None:
            state = torch.zeros(batch_size, num_heads, head_size, head_size, dtype=torch.float32, device=value.device)
        else:
            state = state.clone().float()

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
            output[:, current_index] = (
                state.to(value.dtype) @ current_receptance.to(value.dtype).unsqueeze(-1)
            ).squeeze(-1)

        return output, state

    def reference_rwkv7_attention(self, hidden, state_dict, layer_id, v_first=None):
        batch_size, seq_length, hidden_size = hidden.shape
        prefix = f"blocks.{layer_id}.att."
        num_heads = state_dict[prefix + "r_k"].shape[0]
        head_size = hidden_size // num_heads

        shifted = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
        time_shift = shifted - hidden

        receptance_hidden = hidden + time_shift * state_dict[prefix + "x_r"]
        decay_hidden = hidden + time_shift * state_dict[prefix + "x_w"]
        key_hidden = hidden + time_shift * state_dict[prefix + "x_k"]
        value_hidden = hidden + time_shift * state_dict[prefix + "x_v"]
        in_context_hidden = hidden + time_shift * state_dict[prefix + "x_a"]
        gate_hidden = hidden + time_shift * state_dict[prefix + "x_g"]

        receptance = torch.nn.functional.linear(receptance_hidden, state_dict[prefix + "receptance.weight"])
        raw_decay = (
            state_dict[prefix + "w0"]
            + torch.tanh(decay_hidden @ state_dict[prefix + "w1"]) @ state_dict[prefix + "w2"]
        )
        decay = torch.exp(-0.606531 * torch.sigmoid(raw_decay.float()))

        key = torch.nn.functional.linear(key_hidden, state_dict[prefix + "key.weight"])
        value = torch.nn.functional.linear(value_hidden, state_dict[prefix + "value.weight"])
        if layer_id == 0:
            v_first = value
        else:
            value_residual = torch.sigmoid(
                state_dict[prefix + "v0"] + (value_hidden @ state_dict[prefix + "v1"]) @ state_dict[prefix + "v2"]
            )
            value = value + (v_first - value) * value_residual

        in_context_rate = torch.sigmoid(
            state_dict[prefix + "a0"] + (in_context_hidden @ state_dict[prefix + "a1"]) @ state_dict[prefix + "a2"]
        )
        gate = torch.sigmoid(gate_hidden @ state_dict[prefix + "g1"]) @ state_dict[prefix + "g2"]

        normalized_key = torch.nn.functional.normalize(
            (key * state_dict[prefix + "k_k"]).view(batch_size, seq_length, num_heads, head_size), dim=-1, p=2.0
        )
        key = key * (1 + (in_context_rate - 1) * state_dict[prefix + "k_a"])

        receptance = receptance.view(batch_size, seq_length, num_heads, head_size)
        key = key.view(batch_size, seq_length, num_heads, head_size)
        value = value.view(batch_size, seq_length, num_heads, head_size)
        decay = decay.view(batch_size, seq_length, num_heads, head_size)
        in_context_rate = in_context_rate.view(batch_size, seq_length, num_heads, head_size)

        rwkv, _ = self.reference_rwkv7_linear_attention(
            decay, receptance, key, value, -normalized_key, normalized_key * in_context_rate
        )
        rwkv = torch.nn.functional.group_norm(
            rwkv.reshape(batch_size * seq_length, hidden_size),
            num_heads,
            weight=state_dict[prefix + "ln_x.weight"],
            bias=state_dict[prefix + "ln_x.bias"],
            eps=64e-5,
        ).view(batch_size, seq_length, num_heads, head_size)
        bonus = ((receptance * key * state_dict[prefix + "r_k"]).sum(dim=-1, keepdim=True) * value).view(
            batch_size, seq_length, hidden_size
        )
        rwkv = rwkv.view(batch_size, seq_length, hidden_size) + bonus

        output = torch.nn.functional.linear(rwkv * gate, state_dict[prefix + "output.weight"])
        return output, v_first

    def reference_rwkv7_feed_forward(self, hidden, state_dict, layer_id):
        prefix = f"blocks.{layer_id}.ffn."
        shifted = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
        key_hidden = hidden + (shifted - hidden) * state_dict[prefix + "x_k"]
        key = torch.relu(torch.nn.functional.linear(key_hidden, state_dict[prefix + "key.weight"])).square()
        return torch.nn.functional.linear(key, state_dict[prefix + "value.weight"])

    def reference_rwkv7_logits(self, input_ids, state_dict):
        hidden_size = state_dict["emb.weight"].shape[1]
        num_hidden_layers = 1 + max(
            int(match.group(1)) for key in state_dict if (match := re.match(r"blocks\.(\d+)\.", key))
        )

        hidden = torch.nn.functional.embedding(input_ids, state_dict["emb.weight"])
        v_first = None
        for layer_id in range(num_hidden_layers):
            prefix = f"blocks.{layer_id}."
            if layer_id == 0:
                hidden = torch.nn.functional.layer_norm(
                    hidden,
                    (hidden_size,),
                    weight=state_dict[prefix + "ln0.weight"],
                    bias=state_dict[prefix + "ln0.bias"],
                )

            attention_input = torch.nn.functional.layer_norm(
                hidden, (hidden_size,), weight=state_dict[prefix + "ln1.weight"], bias=state_dict[prefix + "ln1.bias"]
            )
            attention, v_first = self.reference_rwkv7_attention(
                attention_input, state_dict, layer_id=layer_id, v_first=v_first
            )
            hidden = hidden + attention

            feed_forward_input = torch.nn.functional.layer_norm(
                hidden, (hidden_size,), weight=state_dict[prefix + "ln2.weight"], bias=state_dict[prefix + "ln2.bias"]
            )
            hidden = hidden + self.reference_rwkv7_feed_forward(feed_forward_input, state_dict, layer_id=layer_id)

        hidden = torch.nn.functional.layer_norm(
            hidden, (hidden_size,), weight=state_dict["ln_out.weight"], bias=state_dict["ln_out.bias"]
        )
        return torch.nn.functional.linear(hidden, state_dict["head.weight"])

    def get_random_rwkv7_state_dict(self):
        state_dict = self.get_rwkv7_state_dict()
        generator = torch.Generator().manual_seed(0)
        for name, tensor in state_dict.items():
            if name.endswith(".weight") and (".ln" in name or name.startswith("ln_out")):
                state_dict[name] = 1 + torch.randn(tensor.shape, generator=generator) * 0.01
            elif name.endswith(".bias"):
                state_dict[name] = torch.randn(tensor.shape, generator=generator) * 0.01
            else:
                state_dict[name] = torch.randn(tensor.shape, generator=generator) * 0.02

        return state_dict

    def get_rwkv7_state_dict(self, include_deep_embed=False, include_qkv=False):
        hidden_size = 16
        intermediate_size = 32
        state_dict = {
            "emb.weight": torch.zeros(64, hidden_size),
            "head.weight": torch.zeros(64, hidden_size),
            "ln_out.weight": torch.zeros(hidden_size),
            "ln_out.bias": torch.zeros(hidden_size),
            "blocks.0.ln0.weight": torch.zeros(hidden_size),
            "blocks.0.ln0.bias": torch.zeros(hidden_size),
        }

        attention_shapes = {
            "x_r": (1, 1, hidden_size),
            "x_w": (1, 1, hidden_size),
            "x_k": (1, 1, hidden_size),
            "x_v": (1, 1, hidden_size),
            "x_a": (1, 1, hidden_size),
            "x_g": (1, 1, hidden_size),
            "w0": (1, 1, hidden_size),
            "w1": (hidden_size, 6),
            "w2": (6, hidden_size),
            "a0": (1, 1, hidden_size),
            "a1": (hidden_size, 5),
            "a2": (5, hidden_size),
            "v0": (1, 1, hidden_size),
            "v1": (hidden_size, 3),
            "v2": (3, hidden_size),
            "g1": (hidden_size, 7),
            "g2": (7, hidden_size),
            "k_k": (1, 1, hidden_size),
            "k_a": (1, 1, hidden_size),
            "r_k": (4, 4),
            "receptance.weight": (hidden_size, hidden_size),
            "key.weight": (hidden_size, hidden_size),
            "value.weight": (hidden_size, hidden_size),
            "output.weight": (hidden_size, hidden_size),
            "ln_x.weight": (hidden_size,),
            "ln_x.bias": (hidden_size,),
        }

        for layer_id in range(2):
            prefix = f"blocks.{layer_id}."
            state_dict[prefix + "ln1.weight"] = torch.zeros(hidden_size)
            state_dict[prefix + "ln1.bias"] = torch.zeros(hidden_size)
            state_dict[prefix + "ln2.weight"] = torch.zeros(hidden_size)
            state_dict[prefix + "ln2.bias"] = torch.zeros(hidden_size)
            for name, shape in attention_shapes.items():
                state_dict[prefix + "att." + name] = torch.zeros(*shape)
            state_dict[prefix + "ffn.x_k"] = torch.zeros(1, 1, hidden_size)
            state_dict[prefix + "ffn.key.weight"] = torch.zeros(intermediate_size, hidden_size)
            state_dict[prefix + "ffn.value.weight"] = torch.zeros(hidden_size, intermediate_size)
            if include_deep_embed:
                state_dict[prefix + "ffn.s0"] = torch.zeros(1, 1, intermediate_size)
                state_dict[prefix + "ffn.s1"] = torch.zeros(hidden_size, 4)
                state_dict[prefix + "ffn.s2"] = torch.zeros(4, intermediate_size)
                state_dict[prefix + "ffn.s_emb.weight"] = torch.zeros(64, 16)
                state_dict[prefix + "ffn.s_emb_x.weight"] = torch.zeros(16, hidden_size)
            if include_qkv:
                state_dict[prefix + "qkv.qq.weight"] = torch.zeros(8, hidden_size)
                state_dict[prefix + "qkv.k1"] = torch.zeros(hidden_size, 3)
                state_dict[prefix + "qkv.k2"] = torch.zeros(3, 8)
                state_dict[prefix + "qkv.k_emb.weight"] = torch.zeros(64, 8)
                state_dict[prefix + "qkv.k_emb_x.weight"] = torch.zeros(8, hidden_size)
                state_dict[prefix + "qkv.v1"] = torch.zeros(hidden_size, 3)
                state_dict[prefix + "qkv.v2"] = torch.zeros(3, hidden_size)
                state_dict[prefix + "qkv.v_emb.weight"] = torch.zeros(64, hidden_size)
                state_dict[prefix + "qkv.v_emb_x.weight"] = torch.zeros(hidden_size, hidden_size)
                state_dict[prefix + "qkv.x_q"] = torch.zeros(1, 1, 8)
                state_dict[prefix + "qkv.x_k"] = torch.zeros(1, 1, 8)
                state_dict[prefix + "qkv.x_v"] = torch.zeros(1, 1, hidden_size)
                state_dict[prefix + "qkv.lnq.weight"] = torch.zeros(8)
                state_dict[prefix + "qkv.lnq.bias"] = torch.zeros(8)
                state_dict[prefix + "qkv.lnk.weight"] = torch.zeros(8)
                state_dict[prefix + "qkv.lnk.bias"] = torch.zeros(8)
                state_dict[prefix + "qkv.lnv.weight"] = torch.zeros(hidden_size)
                state_dict[prefix + "qkv.lnv.bias"] = torch.zeros(hidden_size)

        return state_dict
