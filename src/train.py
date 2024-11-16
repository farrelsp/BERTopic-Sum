from __future__ import division # ensure that division between integers always results in a floating-point number

import os
import argparse
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, test_abs
from train_extractive import train_ext, validate_ext, test_ext

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-language", default='english', type=str, choices=['english', 'indonesia'])
    parser.add_argument("-model_version", default='p1', type=str, choices=['p1', 'p2'])
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='../bert_data/xlsum')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/')
    parser.add_argument("-temp_dir", default='../temp')  

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/example.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int) # 50000 for extractive, 200000 for abstractive
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-handle_repetition", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-use_valid_data", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    # General Parameters
    parser.add_argument("-label_smoothing", default=0.1, type=float)  # smoothing factor
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int) # beam search size
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int) # maximum length for generated summary
    parser.add_argument("-max_tgt_len", default=140, type=int) # maximum length for target input

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str) # optimizer
    parser.add_argument("-lr", default=2e-3, type=float)  # learning rate 
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=10000, type=int) # follows Vaswani (2017) setup
    parser.add_argument("-warmup_steps_bert", default=20000, type=int)
    parser.add_argument("-warmup_steps_dec", default=10000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    # Params for ABSTRACTIVE
    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float) # learning rate encoder
    parser.add_argument("-lr_dec", default=0.1, type=float) # learning rate decoder
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_use_cache", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-use_topic_emb", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-topic_len", default=5, type=int)
    
    # Params for EXTRACTIVE
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int) # L     
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    args.vocab_size = 30521
    if args.language == "indonesia" and args.large:
        args.vocab_size = 30522
    elif args.language == "indonesia" and not args.large:
        args.vocab_size = 50000

    if not args.use_topic_emb:
        args.topic_len = 0
        
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
        
    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
            
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
            
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        