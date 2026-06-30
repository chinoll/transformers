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
"""Tokenization classes for RWKV7."""

import json
import os

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

MERGES_FILE = "merges.txt"
TOKENIZER_FILE = "tokenizer.json"
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class Rwkv7Tokenizer(PreTrainedTokenizer):
    """
    RWKV byte-level tokenizer using the longest-token match vocabulary from `vocab.json`.

    The vocabulary stores explicit token ids and leaves a few ids unused. This tokenizer preserves those ids instead
    of compacting the vocabulary.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        vocab_size=None,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        kwargs.pop("tokenizer_file", None)
        kwargs.pop("merges_file", None)
        self.vocab_file = vocab_file
        self.idx2token, sorted_tokens = self._load_vocab(vocab_file)
        self.vocab_size_value = max(self.idx2token) + 1 if vocab_size is None else vocab_size
        if self.vocab_size_value <= max(self.idx2token):
            raise ValueError(
                f"`vocab_size` ({self.vocab_size_value}) must be greater than the largest token id "
                f"({max(self.idx2token)}) in the RWKV vocabulary file."
            )
        kwargs["vocab_size"] = self.vocab_size_value

        for token_id in range(self.vocab_size_value):
            if token_id not in self.idx2token:
                if token_id == 0:
                    self.idx2token[token_id] = b"<|endoftext|>"
                else:
                    self.idx2token[token_id] = f"<|rwkv_unused_{token_id}|>".encode()

        self.token2idx = {token: token_id for token_id, token in self.idx2token.items()}
        self.str2idx = {self._bytes_to_token(token): token_id for token, token_id in self.token2idx.items()}
        self.table, self.good, self.max_token_length = self._build_lookup_tables(sorted_tokens)

        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token, **kwargs)

    @staticmethod
    def _bytes_to_token(token):
        return token.decode("latin-1")

    @staticmethod
    def _token_to_bytes(token):
        return token.encode("latin-1")

    @classmethod
    def _load_vocab(cls, vocab_file):
        if not vocab_file.endswith(".json"):
            raise ValueError("`Rwkv7Tokenizer` expects a `vocab.json` file.")
        return cls._load_json_vocab(vocab_file)

    @classmethod
    def _load_json_vocab(cls, vocab_file):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            vocab = json.load(vocab_handle)

        idx2token = {}
        sorted_tokens = []
        for token, idx in vocab.items():
            token = cls._token_to_bytes(token)
            idx = int(idx)
            idx2token[idx] = token
            if idx == 0 or not token.startswith(b"<|rwkv_unused_"):
                sorted_tokens.append(token)

        return idx2token, sorted_tokens

    @staticmethod
    def _build_lookup_tables(sorted_tokens):
        table = [[[] for _ in range(256)] for _ in range(256)]
        good = [set() for _ in range(256)]
        max_token_length = [0 for _ in range(256)]

        for token in sorted_tokens:
            first_byte = token[0]
            second_byte = token[1] if len(token) > 1 else 0
            table[first_byte][second_byte].append(token)
            good[first_byte].add(second_byte)
            max_token_length[first_byte] = max(max_token_length[first_byte], len(token))

        for first_byte in range(256):
            for second_byte in range(256):
                table[first_byte][second_byte].sort(key=len, reverse=True)

        return table, good, max_token_length

    @property
    def vocab_size(self):
        return self.vocab_size_value

    def get_vocab(self):
        return dict(self.str2idx, **self.added_tokens_encoder)

    def encode_bytes(self, text):
        idx = 0
        tokens = []
        while idx < len(text):
            first_byte = text[idx]
            token = bytes([first_byte])

            if idx < len(text) - 1:
                second_byte = text[idx + 1]
                if second_byte in self.good[first_byte]:
                    candidate = text[idx : idx + self.max_token_length[first_byte]]
                    for possible_token in self.table[first_byte][second_byte]:
                        if candidate.startswith(possible_token):
                            token = possible_token
                            break

            tokens.append(self.token2idx[token])
            idx += len(token)

        return tokens

    def decode_bytes(self, tokens):
        return b"".join(self.idx2token[token_id] for token_id in tokens)

    def _tokenize(self, text, **kwargs):
        return [self._bytes_to_token(self.idx2token[token_id]) for token_id in self.encode_bytes(text.encode("utf-8"))]

    def _convert_token_to_id(self, token):
        return self.str2idx.get(token, self.str2idx[self.unk_token])

    def _convert_id_to_token(self, index):
        return self._bytes_to_token(self.idx2token[index])

    def convert_tokens_to_string(self, tokens):
        text = b"".join(self._token_to_bytes(token) for token in tokens)
        return text.decode("utf-8", errors="replace")

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [1 if token_id in self.all_special_ids else 0 for token_id in token_ids_0]
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path (%s) should be a directory.", save_directory)
            return None

        prefix = filename_prefix + "-" if filename_prefix else ""
        out_vocab_file = os.path.join(
            save_directory,
            prefix + VOCAB_FILES_NAMES["vocab_file"],
        )
        vocab = {
            self._bytes_to_token(token): token_id
            for token_id, token in sorted(self.idx2token.items(), key=lambda item: item[0])
        }
        with open(out_vocab_file, "w", encoding="utf-8") as vocab_handle:
            json.dump(vocab, vocab_handle, ensure_ascii=False, indent=2)
            vocab_handle.write("\n")

        out_tokenizer_file = os.path.join(save_directory, prefix + TOKENIZER_FILE)
        tokenizer_manifest = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": self._serialize_added_tokens(),
            "normalizer": None,
            "pre_tokenizer": {"type": "RWKV7ByteLevel"},
            "post_processor": None,
            "decoder": {"type": "RWKV7ByteLevel"},
            "model": {
                "type": "RWKV7LongestMatch",
                "vocab_file": VOCAB_FILES_NAMES["vocab_file"],
                "vocab_size": self.vocab_size,
            },
        }
        with open(out_tokenizer_file, "w", encoding="utf-8") as tokenizer_handle:
            json.dump(tokenizer_manifest, tokenizer_handle, ensure_ascii=False, indent=2)
            tokenizer_handle.write("\n")

        out_merges_file = os.path.join(save_directory, prefix + MERGES_FILE)
        with open(out_merges_file, "w", encoding="utf-8") as merges_handle:
            merges_handle.write("# RWKV7 uses byte-level longest-match tokenization; no BPE merges are used.\n")

        return (out_vocab_file, out_tokenizer_file, out_merges_file)

    def _serialize_added_tokens(self):
        added_tokens = []
        for token_id, token in sorted(self.added_tokens_decoder.items()):
            token_state = token.__getstate__()
            token_state["id"] = token_id
            added_tokens.append(token_state)
        return added_tokens


__all__ = ["Rwkv7Tokenizer"]
