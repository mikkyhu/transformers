#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

# TODO: setting custom seed not robust.
def set_seed(args=None, seed=42):
    if args is None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        # not robust. will crash if no gpus available.
        torch.cuda.manual_seed_all(seed)
    else:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def get_lookahead_entropies(model, context, batch_size, vocab_size, candidates=None, is_xlnet=False, device='cpu'):
    # in: context (1-dim tensor of t tokens)
    # out: ents (1-dim tensor of entropies of w_{t+2} given each candidate w_{t+1})
    #      i.e. ents[w] = H(w_{t+2} | w_{<=t}, w_{t+1}=w)
    # tries all words by default, but can pass in array of candidates

    batch = torch.zeros(len(context)+1, dtype=torch.long, device=device)
    batch[:len(context)] = context
    batch = batch.unsqueeze(0).repeat(batch_size, 1)

    # mark uncomputed entropies with -1 for processing later
    ents = -torch.ones(vocab_size, device=device)

    # if not passed an array of candidates, try all candidates
    if candidates is None:
        candidates = torch.arange(vocab_size, dtype=torch.long, device=device)

    batch_start = 0

    with torch.no_grad():
        # loop over all w_{t+1}, chunking into batches
        while batch_start < len(candidates):
            batch_end = min(len(candidates), batch_start + batch_size)
            batch[:batch_end-batch_start, -1] = candidates[batch_start:batch_end]

            inputs = {'input_ids': batch}
            if is_xlnet: 
                pass # not equipped to deal with xlnet yet

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :]
            
            next_probs = F.softmax(next_token_logits, dim=-1)
            next_ents = torch.sum(-next_probs * torch.log(next_probs + 1e-20), dim=-1)
            
            ents[batch_start:batch_end] = next_ents[:batch_end-batch_start]
            batch_start += batch_size

    return ents

def sample_sequence_calibrated(model, length, context, batch_size, vocab_size, alpha=0.0, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
    num_samples = 1 # since entropy measurement is slow, no point parallelizing
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    
    # ents[k,t] = H(w_t | w_<t) for independent generation k
    ents = torch.zeros((num_samples, length), device=device)

    with torch.no_grad():
        for gen_index in range(length):

            inputs = {'input_ids': generated}
            if is_xlnet: 
                pass # not equipped to deal with xlnet yet

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            model_next_logits = outputs[0][:, -1, :] / temperature
            
            if top_k == 0:
                candidates = None # try all words for lookahead
            else:
                # try k most likely words
                candidates = torch.argsort(model_next_logits[0], descending=True)[:top_k]

            lookahead_ents = get_lookahead_entropies(model, generated[0], batch_size, vocab_size, candidates=candidates, device=device, is_xlnet=is_xlnet).unsqueeze(0)
            
            if top_k != 0:
                # replace uncomputed entropies with average (for centered adjustment)
                top_average_ent = lookahead_ents[lookahead_ents != -1].mean()
                lookahead_ents[lookahead_ents != -1] = top_average_ent
            
            calibrated_next_logits = model_next_logits - alpha * lookahead_ents 
            next_probs = F.softmax(calibrated_next_logits, dim=-1)
            next_ents = torch.sum(-next_probs * torch.log(next_probs + 1e-20), dim=-1)
            
            ents[:, gen_index] = next_ents

            # sample from top k
            filtered_logits = top_k_top_p_filtering(calibrated_next_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated, ents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--save_name", type=str, default="test.npz")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)

    vocab_size = tokenizer.vocab_size
    print('vocab_size:', vocab_size)

    while True:
        raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
        context_tokens = tokenizer.encode(raw_text)

        avg_ents = torch.zeros((1, args.length), device=args.device)

        for i in range(args.num_samples):

            set_seed(seed=i + args.seed)
        
            out, ents = sample_sequence_calibrated(
                model=model,
                context=context_tokens,
                length=args.length,
                batch_size=args.batch_size,
                vocab_size=vocab_size,
                alpha=args.alpha,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                is_xlnet=bool(args.model_type == "xlnet"),
            )
        
            # show all generations from this batch
            for j in range(len(out)):
                seq = out[j, len(context_tokens):].tolist()
                text = tokenizer.decode(seq, clean_up_tokenization_spaces=True)
                print(text) 

            ents = ents.mean(axis=0)
            avg_ents = (avg_ents * i + ents) / (i + 1)

            np.savez(args.save_name, avg_ents=avg_ents.cpu().numpy())
        
        if args.prompt:
            break
    return text


if __name__ == '__main__':
    main()
