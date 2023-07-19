import json
import os

import argparse
import deepspeed
import deepspeed.comm as dist
import numpy as np
import sentencepiece as spm
import torch

from models.configuration_baichuan import BaiChuanConfig
from models.modeling_baichuan import BaiChuanForCausalLM


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_path", type=str,
                        default="tokenizer.model",
                        help="Tokenizer model file path")

    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max tokens per sentence in corpus")

    parser.add_argument("--steps_per_epoch", type=int, default=10,
                        help="Step intervals to save checkpoint")

    parser.add_argument("--checkpoint_saving_path", type=str,
                        default="checkpoints",
                        help="Path to store checkpoint files")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Reserved for deepspeed framework")
    parser.add_argument('--gradient_checkpointing',
                    action='store_true',
                    help='Enable HF gradient checkpointing for model.')
    return parser


arg_parser = get_argument_parser()
arg_parser = deepspeed.add_config_arguments(arg_parser)
args = arg_parser.parse_args()
print(f"args===={args}")
deepspeed.init_distributed()


def get_data():
    ds_config = json.load(open(args.deepspeed_config))
    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]
    max_length = args.max_length
    print(f"train_micro_batch_size_per_gpu={micro_batch_size}")
    data = torch.ones(micro_batch_size, max_length + 1,dtype=torch.long)
    data = data.cuda(non_blocking=True)
    return data


def prepare_model():
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config,
                             enabled=True,
                             mem_efficient_linear=False,
                             mpu=None):
        model = BaiChuanForCausalLM(BaiChuanConfig())
    # print(model)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 optimizer=None,
                                                 model_parameters=model_parameters)
    if args.gradient_checkpointing:
        model_engine.gradient_checkpointing_enable()
    return model_engine


def train(model_engine):
    model_engine.train()
    step = 0
    while step < args.steps_per_epoch:
        data = get_data()
        loss = model_engine(data, labels=data).loss
        model_engine.backward(loss)
        model_engine.step()
        step += 1
        print(f"step={step}")
        deepspeed.comm.log_summary()            
    return


if __name__ == "__main__":
    model_engine = prepare_model()
    train(model_engine)

