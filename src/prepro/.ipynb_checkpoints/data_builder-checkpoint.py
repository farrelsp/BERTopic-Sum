import gc
import os
import glob
import json
import re
import shutil
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
from prepro.utils import _get_word_ngrams

import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from transformers import BertTokenizer

class BertData():
    def __init__(self, args):
        self.args = args
        if self.args.language == "english":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            if self.args.large:
                self.tokenizer = BertTokenizer.from_pretrained(f'indobenchmark/indobert-large-{self.args.model_version}', do_lower_case=True)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(f'indobenchmark/indobert-base-{self.args.model_version}', do_lower_case=True)

        # print(f"Tokenizer name: {self.tokenizer.name_or_path}")
        
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]' if self.args.language == "english" else '[UNUSED_0]'
        self.tgt_eos = '[unused1]' if self.args.language == "english" else '[UNUSED_1]'
        self.tgt_sent_split = '[unused2]' if self.args.language == "english" else '[UNUSED_2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None
    
        original_src_txt = [' '.join(s) for s in src]

        # Bikin list index dari jumlah kalimat yang ada di teks
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]
        
        # Bikin list 0 dengan panjang sama dengan jumlah kalimat di teks
        _sent_labels = [0] * len(src)

        # Ubah isi list menjadi 1 kalau kalimat tersebut merupakan kalimat dalam oracle
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        
        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        
        src_subtokens = self.tokenizer.tokenize(text)        
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = f"{self.tgt_bos} " + f" {self.tgt_sent_split} ".join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt))) for tt in tgt]) + f" {self.tgt_eos}"
        
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

def format_raw(args):
    """ Format the raw file of XLSum dataset into a json with this
      kind of structure {id: ID, tgt: SUMMARY, src: TEXT_BODY} """
    
    for i in glob.glob(pjoin(args.raw_path, 'indonesian_*.jsonl')):
        is_train = True if "train" in i else False
        is_valid = True if "val" in i else False

        raw_formated = _format_raw(i)

        if is_train:
            file_name = "xlsum_train.json"
        elif is_valid:
            file_name = "xlsum_valid.json"
        else:
            file_name = "xlsum_test.json"
        
        json.dump(raw_formated, open(pjoin(args.save_path,file_name),"w"))

def _format_raw(raw_path):
    with open(raw_path, encoding="utf8") as f:
        d = [json.loads(line) for line in f]
        print(f"Total data: {len(d)}")

    json_list = []
    for i in range(len(d)):
        doc = {
            "id":d[i]['id'],
            "tgt":d[i]['summary'],
            "src":d[i]['text']
        }
        json_list.append(doc)
    
    return json_list

def tokenize(args):
    """ Convert XLSum dataset into json format and each file contains
      tokenized version (sentence-splitting and tokens) of src and tgt. """
    
    train_files, valid_files, test_files = [], [], []

    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]

        with open(f, "r") as read_json:
            data_file = json.load(read_json)
            
        if ('valid' in real_name):
            valid_files = data_file
        elif ('test' in real_name):
            test_files = data_file
        elif ('train' in real_name):
            train_files = data_file

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}

    for corpus_type in ['train', 'valid', 'test']:
        
        print(f"Preparing to tokenize {corpus_type} files...")
        dataset = []
        p_ct = 0

        for d in corpora[corpus_type]:
            d_formated = _tokenize(d)
            dataset.append(d_formated)
            if (len(dataset) > args.shard_size-1):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []
                    print(f"Saved tokenized docs in {pt_file}.")
        
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []
                print(f"Saved tokenized docs in {pt_file}.")

def _tokenize(json_element):
    json_element_split = {
        'src': sent_token_split(json_element['src']), 
        'tgt':sent_token_split(json_element['tgt'])
    }

    return json_element_split

def sent_token_split(doc):    
    sent_text = nltk.sent_tokenize(doc) 
    doc_split = []
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)
        doc_split.append(tokenized_text)
    
    return doc_split

def format_to_bert(args):
    """ Format the json files into a .pt file which contains a dict consists of
    src indices, tgt indices, sentence labels (for ext sum), segment ids, cls ids,
    src text, and tgt text"""

    datasets = ['train', 'valid', 'test']
    
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
            
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()

def _format_to_bert(params):
    corpus_type, json_file, args, save_file = params
    is_test = corpus_type == 'test'

    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    nones = 0
    for d in jobs:
        source, tgt = d['src'], d['tgt']

        # Find sentences that has the best rouge combined (oracle)
        sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)

        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
            
        b_data = bert.preprocess(source, tgt, sent_labels, is_test=is_test)

        if (b_data is None):
            nones += 1
            continue
            
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        
        b_data_dict = {"src": src_subtoken_idxs, 
                       "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, 
                       "segs": segments_ids, 
                       'clss': cls_ids,
                       'src_txt': src_txt, 
                       "tgt_txt": tgt_txt}
        
        datasets.append(b_data_dict)

    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    logger.info(f"Total nones in {save_file} is {nones}")
    torch.save(datasets, save_file)
    
    filename = save_file.replace('bert.pt', 'json')
    with open(filename, 'w') as save:
        save.write(json.dumps(datasets))
        
    datasets = []

    gc.collect()
            
def get_long_data(args):
    """ 
    Produce dataset (train, valid, test) with tokens greater than 512.
    This dataset must have the same data whether it is english or indonesian.
    Say that we have the source folder in english.
    First, it will look after documents with token > 512 in the english folder.
    Then, long docs in english folder will be saved.
    The indexes are saved so that the long docs in indonesian will be the same data as in english folder.
    Why do the data have to be the same?
    So that the test data will be the same then the model performance can be compared.
    """
    
    lang_1 = args.raw_path_1.split("/")[-1]
    lang_2 = args.raw_path_2.split("/")[-1]
    
    save_path_1 = f"{args.raw_path_1}_{lang_1}_long"
    save_path_2 = f"{args.raw_path_1}_{lang_2}_long"
    if os.path.exists(save_path_1):
        shutil.rmtree(save_path_1)
    if os.path.exists(save_path_2):
        shutil.rmtree(save_path_2)
    os.mkdir(save_path_1)
    os.mkdir(save_path_2)
    
    datasets = ['train', 'valid', 'test']
    
    long_data_idx = {}
    for corpus_type in datasets:
        logger.info(f"Looping through {args.raw_path_1} {corpus_type} dataset...")
        
        # Loop through the reference dataset
        for json_f in glob.glob(pjoin(args.raw_path_1, '*' + corpus_type + '.*.json')):
            dataset_name = json_f.split('/')[-1].split(".")[0]  # --> xlsum
            idx = json_f.split('/')[-1].split(".")[-2] # --> index: 1-19 (train), 1-3 (valid, test)
            filename = f"/{dataset_name}.{corpus_type}.{idx}.bert.pt"
            
            save_file = save_path_1 + filename
            
            with open(json_f) as json_f:
                data = json.load(json_f)
                datasets = []
    
                # Get data with total tokens > 512
                for i in range(len(data)):
                    src = data[i]['src']
                    if len(src) > 512:
                        datasets.append(data[i])
    
                        # Save the index (key) for data with long tokens
                        if f"{corpus_type}_{idx}" in long_data_idx.keys():
                            long_data_idx[f"{corpus_type}_{idx}"].append(i) 
                        else:
                            long_data_idx[f"{corpus_type}_{idx}"] = [i]
                        
                logger.info(f"Saving {len(datasets)} instances to {save_file}...")
                torch.save(datasets, save_file)
            
                save_file_json = save_file.replace('bert.pt', 'json')
                with open(save_file_json, 'w') as f:
                    f.write(json.dumps(datasets))
    
        
        logger.info(f"Looping through {args.raw_path_2} {corpus_type} dataset...")
        
        # Loop through the follower dataset
        for json_f in glob.glob(pjoin(args.raw_path_2, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1].split(".")[0]
            idx = json_f.split('/')[-1].split(".")[-2]
            filename = f"/{dataset_name}.{corpus_type}.{idx}.bert.pt"
            
            save_file = save_path_2 + filename
    
            with open(json_f) as json_f:
                data = json.load(json_f)
                datasets = []
    
                # Get data with the same index from the reference dataset
                for i in range(len(data)):
                    if i in long_data_idx[f"{corpus_type}_{idx}"]:
                        datasets.append(data[i])
                        
                logger.info(f"Saving {len(datasets)} instances to {save_file}...")
                torch.save(datasets, save_file)
    
                save_file_json = save_file.replace('bert.pt', 'json')
                with open(save_file_json, 'w') as f:
                    f.write(json.dumps(datasets))
                    
def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2

            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        if (cur_id == -1):
            return selected
        
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)