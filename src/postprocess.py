import argparse
import time

from others.logging import init_logger
from postpro import data_builder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-mode", default='reorder', type=str)
    parser.add_argument("-language", default='english', type=str, choices=['english', 'indonesia'])
    parser.add_argument("-raw_path", default='../bert_data/en')
    parser.add_argument("-result_path", default='../results/baseline')
    parser.add_argument("-file_path", default='../results/baseline/ro.result.9000')

    parser.add_argument("-use_valid_data", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-use_truncate", type=str2bool, nargs='?',const=True,default=False)
    
    parser.add_argument("-is_lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-is_long", type=str2bool, nargs='?',const=True,default=False)  
    
    parser.add_argument('-log_file', default='../logs/postprocess.log')

    parser.add_argument('-n_cpus', default=2, type=int)

    args = parser.parse_args()
    init_logger(args.log_file)
    
    eval('data_builder.' + args.mode + '(args)')  # execute function 
