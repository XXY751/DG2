# main_5.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from timeit import default_timer as timer
from utils import *
import random
from datasets.dataset import LoadDataset
from trainer import Trainer
import os
import shutil
from datetime import datetime

CUDA_VISIBLE_DEVICES=0,1,2,3

datasets = [
    'sleep-edfx',
    'HMC',
    'ISRUC',
    'SHHS1',
    'P2018',
]

def main():
    torch.set_float32_matmul_precision('high')
    seed = 0
    setup_seed(seed)

    # --- 新增：创建带时间戳的日志和模型保存目录 ---
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = os.path.join('results', now)
    os.makedirs(model_dir, exist_ok=True)

    # --- 新增：备份代码 ---
    code_backup_dir = os.path.join(model_dir, 'code_backup')
    os.makedirs(code_backup_dir, exist_ok=True)
    for filename in os.listdir('.'):
        if filename.endswith('.py'):
            shutil.copy(filename, code_backup_dir)
    for dirname in ['models', 'losses', 'datasets', 'utils', 'prepare_datasets']:
        if os.path.isdir(dirname):
            shutil.copytree(dirname, os.path.join(code_backup_dir, dirname))


    accs, f1s = [], []
    print("--- GPU Diagnosis ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print("---------------------")

    for dataset_name in datasets:
        parser = argparse.ArgumentParser(description='SleepDG')
        parser.add_argument('--target_domains', type=str, default=dataset_name, help='target_domains')
        parser.add_argument('--seed', type=int, default=443, help='random seed (default: 0)')
        parser.add_argument('--cuda', type=int, default=4, help='cuda number (default: 1)')
        parser.add_argument('--epochs', type=int, default=200, help='number of epochs (default: 5)')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size for training (default: 32)')
        parser.add_argument('--num_of_classes', type=int, default=5, help='number of classes')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
        parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='dropout')
        parser.add_argument('--datasets_dir', type=str, default='/data/lijinyang/SleepSLeep/datasets_dir', help='datasets_dir')
        # --- 修改：将新的 model_dir 传递给参数 ---
        parser.add_argument('--model_dir', type=str, default=model_dir, help='model_dir')
        parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
        parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')

        parser.add_argument('--projection_type', type=str, default='diag')
        parser.add_argument('--lowrank_rank', type=int, default=32)
        parser.add_argument('--enable_stats_alignment', type=int, default=1)
        parser.add_argument('--anchor_momentum', type=float, default=0.9)
        parser.add_argument('--lambda_caa', type=float, default=1.0)
        parser.add_argument('--lambda_stat', type=float, default=0.2)
        parser.add_argument('--lambda_Areg', type=float, default=0.1)
        parser.add_argument('--lambda_ae', type=float, default=1.0)
        parser.add_argument('--lambda_coral', type=float, default=0.0)
        parser.add_argument('--num_domains', type=int, default=4)

        params = parser.parse_args()
        print(params)

        trainer = Trainer(params)
        test_acc, test_f1 = trainer.train()
        accs.append(test_acc)
        f1s.append(test_f1)

    print(accs)
    print(f1s)
    print(np.mean(accs), np.mean(f1s))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()