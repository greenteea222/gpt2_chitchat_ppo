
from PPO import PPO

import numpy as np
import torch.backends.cudnn as cudnn
import torch

import argparse
import logging
import time
import os
import sys

parser = argparse.ArgumentParser('parser')

#模型参数
parser.add_argument('--bert_model_path', type=str, default='BERT_chinese/model/')
parser.add_argument('--bert_tokenizer_path', type=str, default='BERT_chinese/model/vocab.txt')
parser.add_argument('--origin_model_path', type=str, default='dialogue_model_train/from_initial_GPT2/model_epoch10/')
parser.add_argument('--mmi_model_path', type=str, default='origin_model/mmi_model/')
parser.add_argument('--tokenizer_path', type=str, default='vocabulary/vocab_small.txt')

#RL参数
parser.add_argument('--rl_epochs', type=int, default=100)
parser.add_argument('--rl_lr', type=float, default=3.5e-4)
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--entropy_weight', type=float, default=1e-5)
parser.add_argument('--baseline_weight', type=float, default=0.95)
parser.add_argument('--embedding_size', type=int, default=32)
parser.add_argument('--algorithm', type=str, choices=['PPO', 'PG', 'RS'], default='PPO')
parser.add_argument('--lambda_easy_answer', type=int, default=0.25)
parser.add_argument('--lambda_novel', type=int, default=0.25)
parser.add_argument('--lambda_mutual_inform', type=int, default=0.5)

#PPO
parser.add_argument('--ppo_epochs', type=int, default=10)
parser.add_argument('--clip_epsilon', type=float, default=0.2)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--work_dir', type=str, default='rl_work_folder')
#parser.add_argument('--log_path', type=str, default='rl_work_folder/ppo_training.log')
args = parser.parse_args()

def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """

    exp_dir_name = 'search_{}_{}'.format(args.algorithm, time.strftime("%Y%m%d-%H%M%S"))
    exp_dir = os.path.join(args.work_dir, exp_dir_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')


    log_file_name = os.path.join( exp_dir, 'log.txt')
    #log_file_name = os.path.join(args.work_dir, 'log.txt')
    f = open(log_file_name, 'w')
    f.close()

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename= os.path.join(log_file_name))

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger, exp_dir

def main(logger, exp_dir):
    # exp_dir = 'search_{}_{}'.format(args.algorithm, time.strftime("%Y%m%d-%H%M%S"))
    # if not os.path.exists(exp_dir):
    #     os.mkdir(exp_dir)
    # log_format = '%(asctime)s %(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    logger.info('args = %s', args)

    if args.algorithm == 'PPO' or args.algorithm == 'PG':
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(str(args.gpu)))
            cudnn.benchmark = True
            cudnn.enable = True
            logger.info('using gpu : {}'.format(args.gpu))
            torch.cuda.manual_seed(args.seed)
        else:
            device = torch.device('cpu')
            logger.info('using cpu')

        if args.algorithm == 'PPO':
            ppo = PPO(args, device, logger, exp_dir)
            ppo.solve_environment()
        elif args.algorithm == 'PG':
            pass



if __name__ == '__main__':
    global logger
    logger, exp_dir = create_logger(args)
    main(logger, exp_dir)