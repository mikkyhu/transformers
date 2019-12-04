from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

from generate_with_calibration import get_lookahead_entropies

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

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def calibrate(model, context, vocab_size, batch_size = 512, top_k=0, iters=100000,device='cpu'):
    N = len(context)
    Pr = torch.zeros(N-1)
    H = torch.zeros(N-1)
    S = torch.zeros(N-1)

    alpha = torch.randn(1, requires_grad=True)
    optimizer = optim.SGD([alpha], lr=0.001, momentum=0.9)

    def init_CEL():
        # TODO: missing 1st word
        for i in trange(1, N):
            context_i = torch.tensor(context[:i], dtype = torch.long, device=device).unsqueeze(0)
            lookahead_entropies = get_lookahead_entropies(
                model = model,
                context = context_i[0],
                batch_size = batch_size,
                vocab_size = vocab_size,
                device = device
            )

            inputs = {'input_ids': context_i}
            outputs = model(**inputs)
            next_logits = outputs[0][:, -1, :]
            next_probs = F.softmax(next_logits, dim=-1)[0]
    
            # cache useful values
            next_word = context[i]
            Pr[i-1] = next_probs[next_word]
            H[i-1] = lookahead_entropies[next_word]
            S[i-1] = torch.dot(next_probs, torch.exp(lookahead_entropies))

    def CEL(a):
        Za = S * torch.exp(-a)
        return torch.sum(Pr * torch.exp(-a * H) / Za)
    
    init_CEL()
    loss = CEL(alpha)
    loss.backward()
    
    lastloss = loss.clone().item()
    for i in range(iters):
        optimizer.step()
        loss = CEL(alpha)
        
        # print(loss)
        # print(lastloss)
        
        if i % 10000 == 9999:    
            print(f'Loss at iter {i}: {loss}. Alpha: {alpha}.')
        
        if loss.item() - lastloss > 0:
            print(f'Stopping at iteration {i}. Alpha: {alpha}.')
            break
        
        lastloss = loss.clone().item()
         
    return alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    # parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    # parser.add_argument("--alpha", type=float, default=0)
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

    print(args)

    vocab_size = tokenizer.vocab_size
    print('vocab_size:', vocab_size)

    while True:
        raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
        context_tokens = tokenizer.encode(raw_text)
        alpha = calibrate(
            model=model,
            context=context_tokens,
            batch_size=args.batch_size,
            vocab_size=vocab_size,
            top_k=args.top_k,
            device=args.device,
        )
        print(alpha)

if __name__ == '__main__':
    main()
