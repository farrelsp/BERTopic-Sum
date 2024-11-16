from __future__ import division

import argparse

import os
import glob
import time
import random

import torch
from transformers import BertTokenizer

from models import data_loader
from models import model_builder
from models.data_loader import load_dataset
from models.model_builder import AbsSummarizer
from models.loss import abs_loss
from models.trainer import build_trainer
from models.predictor import build_predictor

from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_abs(args, device_id):
    # Initialize the logger and log the arguments
    init_logger(args.log_file)
    logger.info(str(args))

    # Determine the device to use (CPU or GPU)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Set the GPU device if a GPU is available
    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    # Load checkpoint if specified
    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        print("Checkpoint:", checkpoint['model']['decoder.transformer_layers.0.mask'].size())
        opt = vars(checkpoint['opt'])
        
        # Update the arguments with options from the checkpoint
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    # Load a pre-trained extractive model if specified
    if args.load_from_extractive != '':
        logger.info('Loading bert from extractive model %s' % args.load_from_extractive)
        bert_from_extractive = torch.load(args.load_from_extractive, map_location=lambda storage, loc: storage)
        bert_from_extractive = bert_from_extractive['model']
    else:
        bert_from_extractive = None

    # Reset random seeds again for safety
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Initialize the abstractive summarizer model
    model = AbsSummarizer(args, device, checkpoint, bert_from_extractive)

    # Set up the optimizer(s)
    if args.sep_optim:
        optim_bert = model_builder.build_optim_bert(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    logger.info(model)

    # Set up the tokenizer based on the specified language
    if args.language == "english":
        tokenizer_name = 'bert-base-uncased'
    else:
        tokenizer_name = f'indobenchmark/indobert-{"large" if args.large else "base"}-{args.model_version}'

    # Define unused tokens and padding token
    token_un0 = '[unused0]' if args.language == "english" else '[UNUSED_0]'
    token_un1 = '[unused1]' if args.language == "english" else '[UNUSED_1]'
    token_un2 = '[unused2]' if args.language == "english" else '[UNUSED_2]'
    token_pad = '[PAD]'

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab[token_un0], 'EOS': tokenizer.vocab[token_un1],
               'PAD': tokenizer.vocab[token_pad], 'EOQ': tokenizer.vocab[token_un2]}

    logger.info(f"Tokenizer: {tokenizer.name_or_path}")
    # logger.info(f"Tokens: {token_un0}, {token_un1}, {token_un2}, {token_pad}")
    # logger.info(f"Symbols: {symbols}")

    logger.info(f"Vocab size: {model.vocab_size}")
    # logger.info(f"Label smoothing: {args.label_smoothing}")

    # Define a function to get the training data loader
    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=False), args.batch_size, device, tokenizer,
                                      shuffle=False, is_test=False)

    # Define the loss function for training
    train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True, label_smoothing=args.label_smoothing)

    # Build the trainer
    trainer = build_trainer(args, device_id, model, optim, train_loss)

    # Start the training process
    trainer.train(train_iter_fct, args.train_steps)

def validate_abs(args, device_id):
    timestep = 0

    # Continuously monitor the directory for new checkpoints
    while (True):
        # Get the list of checkpoint files sorted by modification time
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        
        if (cp_files):
            cp = cp_files[-1]
            time_of_cp = os.path.getmtime(cp)

            # If the checkpoint file is empty, wait for a minute and check again
            if (not os.path.getsize(cp) > 0):
                time.sleep(60)
                continue
            
            # If the checkpoint file is newer than the last seen checkpoint, validate and test it
            if (time_of_cp > timestep):
                timestep = time_of_cp
                step = int(cp.split('.')[-2].split('_')[-1])
                validate(args, device_id, cp, step)
                test_abs(args, device_id, cp, step)

        # Update the list of checkpoint files
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)

        if (cp_files):
            cp = cp_files[-1]
            time_of_cp = os.path.getmtime(cp)
            # If a new checkpoint has been added, validate it immediately
            if (time_of_cp > timestep):
                continue
        else:
            # If no new checkpoint is found, wait for 5 minutes before checking again
            time.sleep(300)

def validate(args, device_id, pt, step):
    # Checks the device to be used, whether it's CPU or GPU
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    # Loads the checkpoint from a specified path
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from

    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)

    # Extracts and applies model configuration options from the checkpoint
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    # Initializes the summarization model using the loaded checkpoint
    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    # Chooses the appropriate tokenizer based on the specified language
    if args.language == "english":
        tokenizer_name = 'bert-base-uncased' 
    else:
        tokenizer_name = f'indobenchmark/indobert-{"large" if args.large else "base"}-{args.model_version}'
    
    # Sets special tokens based on the language
    token_un0 = '[unused0]' if args.language == "english" else '[UNUSED_0]'
    token_un1 = '[unused1]' if args.language == "english" else '[UNUSED_1]'
    token_un2 = '[unused2]' if args.language == "english" else '[UNUSED_2]'
    token_pad = '[PAD]'

    # Loads the tokenizer with the specified configuration
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True, cache_dir=args.temp_dir)
    
    # Defines symbols for special tokens
    symbols = {'BOS': tokenizer.vocab[token_un0], 'EOS': tokenizer.vocab[token_un1],
               'PAD': tokenizer.vocab[token_pad], 'EOQ': tokenizer.vocab[token_un2]}
        
    logger.info(f"Tokenizer: {tokenizer.name_or_path}")
    logger.info(f"Tokens: {token_un0}, {token_un1}, {token_un2}, {token_pad}")
    logger.info(f"Symbols: {symbols}")

    # Logs model vocabulary size and label smoothing configuration
    logger.info(f"Vocab size: {model.vocab_size}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    
    # Prepares the data loader for validation
    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device, tokenizer,
                                        shuffle=False, is_test=False)

    # Initializes the loss function for validation
    valid_loss = abs_loss(model.generator, symbols, model.vocab_size, train=False, device=device)

    # Builds the trainer with the model and validation loss
    trainer = build_trainer(args, device_id, model, None, valid_loss)
    
    # Validates the model using the validation data iterator
    stats = trainer.validate(valid_iter, step)
    
    # Returns the cross-entropy loss from the validation statistics
    return stats.xent()


def test_abs(args, device_id, pt, step):
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

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()

    corpus = 'test'
    if args.use_valid_data:
        corpus = 'valid'

    print("Corpus type:", corpus)
    
    # chooses the appropriate tokenizer based on the language
    if args.language == "english":
        tokenizer_name = 'bert-base-uncased' 
    else:
        tokenizer_name = f'indobenchmark/indobert-{"large" if args.large else "base"}-{args.model_version}'
    token_un0 = '[unused0]' if args.language == "english" else '[UNUSED_0]'
    token_un1 = '[unused1]' if args.language == "english" else '[UNUSED_1]'
    token_un2 = '[unused2]' if args.language == "english" else '[UNUSED_2]'
    token_pad = '[PAD]'

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab[token_un0], 'EOS': tokenizer.vocab[token_un1],
               'PAD': tokenizer.vocab[token_pad], 'EOQ': tokenizer.vocab[token_un2]}

    logger.info(f"Tokenizer: {tokenizer.name_or_path}")
    logger.info(f"Tokens: {token_un0}, {token_un1}, {token_un2}, {token_pad}")
    logger.info(f"Symbols: {symbols}")

    logger.info(f"Vocab size: {model.vocab_size}")
    logger.info(f"Label smoothing: {args.label_smoothing}")

    test_iter = data_loader.Dataloader(args, load_dataset(args, corpus, shuffle=False),
                                       args.test_batch_size, device, tokenizer,
                                       shuffle=False, is_test=True)
    
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step, use_cache=args.dec_use_cache, use_topic_emb=args.use_topic_emb)