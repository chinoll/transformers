<!--Copyright 2026 Chinoll and HuggingFace Inc. team.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-01.*

# RWKV7

## Overview

RWKV7 is a recurrent language model architecture from the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
project. The implementation in Transformers uses the RWKV7 recurrent state and exposes standard causal language model
APIs.

The current implementation uses a plain PyTorch loop for token mixing. This path is intended to match the RWKV-LM
reference numerically, but it can be slow for long sequences.

## Usage example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("chinoll/rwkv7-g1d-0.1b", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("chinoll/rwkv7-g1d-0.1b")

inputs = tokenizer("Hello my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=16)
print(tokenizer.decode(outputs[0]))
```

## Rwkv7Config

[[autodoc]] Rwkv7Config

## Rwkv7Tokenizer

[[autodoc]] Rwkv7Tokenizer

## Rwkv7Model

[[autodoc]] Rwkv7Model
    - forward

## Rwkv7ForCausalLM

[[autodoc]] Rwkv7ForCausalLM
    - forward
