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

import ast
import os
import shutil

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "rwkv_vocab_v20230424.txt"}


class Rwkv7Tokenizer(PreTrainedTokenizer):
    """
    RWKV byte-level tokenizer using the longest-token match vocabulary from `rwkv_vocab_v20230424.txt`.

    The vocabulary file stores explicit token ids and leaves a few ids unused. This tokenizer preserves those ids
    instead of compacting the vocabulary.
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
        **kwargs,
    ):
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

        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

    @staticmethod
    def _bytes_to_token(token):
        return token.decode("latin-1")

    @staticmethod
    def _token_to_bytes(token):
        return token.encode("latin-1")

    @staticmethod
    def _load_vocab(vocab_file):
        idx2token = {}
        sorted_tokens = []
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            for line in vocab_handle:
                idx = int(line[: line.index(" ")])
                token = ast.literal_eval(line[line.index(" ") : line.rindex(" ")])
                token = token.encode("utf-8") if isinstance(token, str) else token
                if not isinstance(token, bytes):
                    raise ValueError(f"Invalid RWKV token at index {idx}: expected bytes or str, got {type(token)}")
                if len(token) != int(line[line.rindex(" ") :]):
                    raise ValueError(f"Invalid RWKV token length at index {idx}.")
                sorted_tokens.append(token)
                idx2token[idx] = token
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

        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            shutil.copyfile(self.vocab_file, out_vocab_file)
        return (out_vocab_file,)


__all__ = ["Rwkv7Tokenizer"]
