from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch
from transformers import BertTokenizer

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from

    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    # Set up the tokenizer based on the specified language
    if args.language == "english":
        tokenizer_name = 'bert-base-uncased'
    else:
        tokenizer_name = f'indobenchmark/indobert-{"large" if args.large else "base"}-{args.model_version}'
    
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True, cache_dir=args.temp_dir)
    logger.info(f"Tokenizer: {tokenizer.name_or_path}")
    
    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device, tokenizer,
                                        shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()

def validate_ext(args, device_id):
    timestep = 0

    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []

        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))

            if (i - max_step > 10):
                break

        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))

        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_ext(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)

            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)

                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue

                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_ext(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)

            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)

                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)

def test_ext(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from

    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    # Set up the tokenizer based on the specified language
    if args.language == "english":
        tokenizer_name = 'bert-base-uncased'
    else:
        tokenizer_name = f'indobenchmark/indobert-{"large" if args.large else "base"}-{args.model_version}'
    
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True, cache_dir=args.temp_dir)
    logger.info(f"Tokenizer: {tokenizer.name_or_path}")

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device, tokenizer,
                                       shuffle=False, is_test=True)
    
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)

def train_ext(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

     # Set up the tokenizer based on the specified language
    if args.language == "english":
        tokenizer_name = 'bert-base-uncased'
    else:
        tokenizer_name = f'indobenchmark/indobert-{"large" if args.large else "base"}-{args.model_version}'
    
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True, cache_dir=args.temp_dir)
    logger.info(f"Tokenizer: {tokenizer.name_or_path}")

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device, tokenizer,
                                      shuffle=True, is_test=False)

    model = ExtSummarizer(args, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)
