import argparse
import time

from others.logging import init_logger
from prepro import data_builder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-mode", default='train', type=str)
    parser.add_argument("-language", default='english', type=str, choices=['english', 'indonesia'])
    parser.add_argument("-model_version", default='p1', type=str, choices=['p1', 'p2'])
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-select_mode", default='greedy', type=str) # to get oracle summary
    parser.add_argument("-map_path", default='../mapping')
    parser.add_argument("-raw_path", default='../raw_data')
    parser.add_argument("-save_path", default='../json_data')

    # Get long data
    parser.add_argument("-raw_path_1", default='../bert_data/en')
    parser.add_argument("-raw_path_2", default='../bert_data/en')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_src_nsents', default=3, type=int)
    parser.add_argument('-max_src_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('-min_tgt_ntokens', default=5, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)

    parser.add_argument('-log_file', default='../logs/preprocess.log')

    parser.add_argument('-n_cpus', default=2, type=int)

    args = parser.parse_args()
    
    init_logger(args.log_file)
    eval('data_builder.' + args.mode + '(args)')  # execute function 
    
    
