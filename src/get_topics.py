import os
import argparse
import time

from others.logging import init_logger
from topics import topic_builder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-model", default='bertopic', type=str, choices=['bertopic', 'lda'])
    parser.add_argument("-source_path", default='../bert_data')
    parser.add_argument("-save_path", default='../topic_data')
    parser.add_argument("-topic_model_path", default='../topic_models')
    parser.add_argument("-is_scoring", type=str2bool, nargs='?',const=True,default=False)
    
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    
    parser.add_argument("-n_topics", default=5, type=int)
    parser.add_argument("-top_n_words", default=10, type=int)
    
    parser.add_argument("-sbert", default='id', type=str, choices=['id', 'en'])
    parser.add_argument("-umap_components", default=5, type=int)
    parser.add_argument("-min_cluster_size", default=10, type=int)
    
    parser.add_argument('-log_file', default='../logs/preprocess.log')

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    
    init_logger(args.log_file)
    eval('topic_builder.get_topic_dist(args)')  
    