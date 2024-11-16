import os
import glob
import json
import re
import string
from os.path import join as pjoin

import torch

import rouge 

from others.logging import logger

def convert_pt_to_json(args):
    for file in os.listdir(args.raw_path):
        if "bert.pt" not in file:
            continue
        filename = args.raw_path + "/" + file
        print(filename)
        datasets = torch.load(filename)
        print(datasets[0])
        filename = filename.replace('bert.pt', 'json')
        with open(filename, 'w') as f:
            f.write(json.dumps(datasets))
            
def reorder(args):
    """Reorder summary results so that they are the same order as the original valid/test data. """
    
    logger.info("Reorder data...")
    
    for f in glob.glob(pjoin(args.result_path, 'ro.*.*')):
        if os.path.isfile(f):
            logger.info(f"Removing {f}...")
            os.remove(f)
    
    corpus_type = "valid" if args.use_valid_data else "test"
    logger.info(f"Corpus type: {corpus_type}")

    logger.info(f"Mapping {corpus_type} data to index...")
    
    # Map seluruh dokumen test set ke index 
    mapping_tgt = {}
    threshold = 0    
    for json_f in sorted(glob.glob(pjoin(args.raw_path, f'*{corpus_type}.*.json'))):
        with open(json_f) as json_f:
            data = json.load(json_f)
            threshold = threshold + len(data)
            for i in range(len(data)):
                j = i + threshold - len(data)
                if len(mapping_tgt) < threshold: 
                    tgt = data[i]['tgt_txt']
                    mapping_tgt[j] = tgt
              
    logger.info(f"Total {corpus_type} data: {len(mapping_tgt)}")

    keys = list(mapping_tgt.keys())
    vals = list(mapping_tgt.values())
    
    gold_docs = []
    cand_docs = []
    src_docs = []
    
    for f in glob.glob(pjoin(args.result_path, '*.*.gold')):
        real_name = f.split("/")[-1].split(".")[0]
        step = f.split("/")[-1].split(".")[1]
        with open(f) as file:
            for line in file: 
                line = line.strip()
                gold_docs.append(line) 
                
    for f in glob.glob(pjoin(args.result_path, '*.*.candidate')):
        with open(f) as file:
            for line in file: 
                line = line.strip() 
                cand_docs.append(line) 
                
    for f in glob.glob(pjoin(args.result_path, '*.*.raw_src')):
        with open(f) as file:
            for line in file: 
                line = line.strip() 
                src_docs.append(line) 
    
    save_gold = ["" for i in range(len(gold_docs))]
    save_cand = ["" for i in range(len(gold_docs))]
    save_src = ["" for i in range(len(gold_docs))]
    
    logger.info("Matching the data...")
    
    for i in range(len(gold_docs)):    # looping seluruh gold summary
        if gold_docs[i] in vals:       # jika gold doc nya ada di dalam test set
            pos = vals.index(gold_docs[i])
            save_gold[pos] = gold_docs[i]  # tempatkan gold sesuai key di dalam dict test set
            save_cand[pos] = cand_docs[i]
            save_src[pos] = src_docs[i]
    
    logger.info("Saving the reordered data...")
    
    with open(args.result_path + "/ro." + real_name + "." + step + ".gold", 'w') as f:
        for e in save_gold:
            f.write(f"{e}\n")
    
    with open(args.result_path + "/ro." + real_name + "." + step +".candidate", 'w') as f:
            for e in save_cand:
                f.write(f"{e}\n")
    
    with open(args.result_path + "/ro." + real_name + "." + step + ".raw_src", 'w') as f:
            for e in save_src:
                f.write(f"{e}\n")

def get_rouge_per_doc(args):
    """ Produce separated file containing ROUGE results for each doc in valid/test set. """
    
    logger.info("Getting ROUGE for each documents...")
    
    aggregator = "Individual"
    logger.info(f'ROUGE Evaluation with {aggregator}')
    
    candidates = []
    with open(args.file_path + ".candidate") as file:
        for line in file:
            line = line.strip()
            candidates.append(line)
            
    golds = []
    with open(args.file_path + ".gold") as file:
        for line in file:
            line = line.strip()
            golds.append(line)
    
    logger.info(f"Total candidate docs: {len(candidates)}")
    logger.info(f"Total gold docs: {len(golds)}")
    
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=3,
                           limit_length=False,
                           alpha=0.5, # Default F1_score
                           apply_avg=False,
                           apply_best=False,
                           weight_factor=1.0,
                           stemming=True)

    logger.info("Evaluating ROUGE scores...")
    scores = evaluator.get_scores(candidates, golds)
    
    results_list = [0 for i in range(len(candidates))]
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        for cand_id, results_per_ref in enumerate(results):
            if metric == 'rouge-1':
                results_list[cand_id] = [results_per_ref['f'][0]]
            else:
                results_list[cand_id].append(results_per_ref['f'][0])

    logger.info("Saving ROUGE file...")
    with open(args.file_path + ".rouge", "w") as file:
        for res in results_list:
            line = 'ROUGE-F(1/2/3/L): {:2.4f}/{:2.4f}/{:2.4f}/{:2.4f}'.format(res[0]*100,res[1]*100,res[2]*100,res[3]*100)
            file.write(f"{line}\n")
            # logger.info(line)

def handle_repetition(args):
    """ 
    Remove repetitive punctuation and words. 
    Example: halo halo bandung . . . --> halo bandung .
    """
    logger.info("Handling repetitive symbols and words...")
    
    files = []
    for file in os.listdir(args.result_path):
        if "ro." in file:
            files.append(file)
            
    for file in files:
        if "candidate" in file:
            lines_handled = []
            
            with open(f"{args.result_path}/{file}") as f:
                for line in f:      
                    text = line.split()
                    for i in range(len(text)):
                        if i > 0:
                            if text[i] == text[i - 1] and text[i] in string.punctuation:
                                text[i-1] = ""
                            if text[i] in string.punctuation and text[i-1] in string.punctuation:
                                text[i] = ""
                            if text[i] == text[i - 1]:
                                text[i-1] = ""
                    text = ' '.join(text)
                    text = ' '.join(text.split())
                    lines_handled.append(text)

            logger.info("Updating .candidate file...")
            with open(f"{args.result_path}/{file}", "w") as f:     
                for line in lines_handled:
                    f.write(f"{line}\n")
    
def truncate(args):
    """ 
    Truncate the text so that the total words is reduced 
    in accordance to the average total words of the target summary. 
    """
    logger.info("Truncating the documents...")
    
    files = []
    
    for file in os.listdir(args.result_path):
        if "ro." in file:
            files.append(file)
    
    lengths_gold = []
    lengths_cand = []
    for file in files:
        if "gold" in file:
            with open(f"{args.result_path}/{file}") as file:
                for line in file:
                    lengths_gold.append(len(line.split()))   
        elif "candidate" in file:
            with open(f"{args.result_path}/{file}") as file:
                for line in file:
                    lengths_cand.append(len(line.split())) 
    
    avg_words_gold = sum(lengths_gold) / len(lengths_gold)
    avg_words_cand = sum(lengths_cand) / len(lengths_cand)
    
    logger.info("Statistics")
    logger.info(f"Average words in gold: {avg_words_gold}")
    logger.info(f"Average words in candidate: {avg_words_cand}")
    logger.info(f"Max. words in gold: {max(lengths_gold)}")
    logger.info(f"Max. words in candidate: {max(lengths_cand)}")

    logger.info("Truncate to be average total words in gold summary...")
    
    if avg_words_cand > avg_words_gold + 5:
        max_words = round(avg_words_gold) + 5
        truncates = []
        for file in files:
            if "candidate" in file:
                with open(f"{args.result_path}/{file}") as file:
                    for line in file:
                        truncates.append(" ".join(line.split()[:max_words]))
                        
        truncate_filename = ".".join(file.name.split(".")[:-1]) + ".truncate"
        
        with open(f"{truncate_filename}", "w") as file:
            for trunc in truncates:
                file.write(f"{trunc}\n")

def get_final_rouge(args):
    """ Calculate final average ROUGE score from all documents. """
    
    logger.info("Calculate final ROUGE score...")
    
    golds = []
    with open(args.file_path + ".gold") as file:
        for line in file:
            line = line.strip()
            golds.append(line)
            
    extension = ".candidate"
    if args.use_truncate:
        extension = ".truncate"

    candidates = []
    
    with open(args.file_path + extension) as file:
        for line in file:
            line = line.strip()
            candidates.append(line)
    
    for aggregator in ['Individual']:
    
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                               max_n=3,
                               limit_length=False,
                               alpha=0.5, # Default F1_score
                               apply_avg=False,
                               apply_best=False,
                               weight_factor=1.0,
                               stemming=True)
        
        scores = evaluator.get_scores(candidates, golds)
        
        results_list = [0 for i in range(len(candidates))]
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            
            for cand_id, results_per_ref in enumerate(results):
                if metric == 'rouge-1':
                    results_list[cand_id] = [results_per_ref['f'][0]]
                else:
                    results_list[cand_id].append(results_per_ref['f'][0])
    
    r1 = []
    r2 = []
    r3 = []
    rl = []
    for res in results_list:
        r1.append(res[0])
        r2.append(res[1])
        r3.append(res[2])
        rl.append(res[3])
    
    logger.info("ROUGE Scores")
    logger.info(f"ROUGE-1: {sum(r1)/len(r1) * 100}")
    logger.info(f"ROUGE-2: {sum(r2)/len(r2) * 100}")
    logger.info(f"ROUGE-L: {sum(rl)/len(rl) * 100}")