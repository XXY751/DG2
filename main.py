# main_5.py

import argparse
import os
import shutil
import random
from datetime import datetime
from pathlib import Path
# ===== 线程数限制（必须在导入 numpy 前）=====
import os
_DEFAULT_THREADS = "8"
for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ.setdefault(k, _DEFAULT_THREADS)

import numpy as np
import torch

from newtraining import Trainer

# =============== 可按需修改的 target 列表 ===============
datasets = [
    'sleep-edfx',
    'HMC',
    'ISRUC',
    'SHHS1',
    'P2018',
]


def setup_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def backup_code(dst_root: Path):
    """备份当前项目中关键代码到 results/<ts>/code_backup/"""
    code_backup_dir = dst_root / 'code_backup'
    code_backup_dir.mkdir(parents=True, exist_ok=True)

    # 备份顶层 .py
    for filename in os.listdir('.'):
        if filename.endswith('.py'):
            shutil.copy(filename, code_backup_dir / filename)

    # 备份常见代码目录
    for dirname in ['models', 'losses', 'datasets', 'utils', 'prepare_datasets', 'scripts']:
        if os.path.isdir(dirname):
            dst_dir = code_backup_dir / dirname
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(dirname, dst_dir)


def build_argparser():
    parser = argparse.ArgumentParser(description='SleepDG Runner (multi-target driver)')

    # ==== 基础训练参数 ====
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--num_of_classes', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--datasets_dir', type=str, default='/data/lijinyang/SleepSLeep/datasets_dir')

    # ==== 设备/种子 ====
    parser.add_argument('--seed', type=int, default=443, help='random seed')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='CUDA_VISIBLE_DEVICES, e.g. "0,1" or "0" or "" for CPU')

    # ==== 目录 ====
    parser.add_argument('--results_root', type=str, default='results', help='根结果目录（会自动加时间戳子目录）')

    # ==== 训练控制：数据抽样比例 ====
    parser.add_argument('--data_ratio', type=float, default=0.01,
                        help='训练集采样比例 (0,1]；例如 0.1 表示只用 10% 训练数据加速调试')
    parser.add_argument('--eval_ratio', type=float, default=1.0,
                        help='验证/测试集采样比例 (0,1]；一般保持 1.0，不建议减少（目前 Trainer 仅使用 data_ratio）')

    # ==== 跨折（由 Trainer 负责将结果保存到 fold{fold}/）====
    parser.add_argument('--fold', type=int, default=0, help='当前折编号（0-based）')
    parser.add_argument('--run_name', type=str, default='exp', help='本次运行命名（写入聚合CSV）')

    # ==== 你已有的超参 ====
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

    # ==== per-target 参数（会在循环里覆盖 default）====
    parser.add_argument('--target_domains', type=str, default='', help='在循环里覆盖为每个数据集的名字')

    return parser


def main():
    # -------- 解析一次全局入参 --------
    parser = build_argparser()
    args = parser.parse_args()

    # 校验比例
    if not (0 < args.data_ratio <= 1.0):
        raise ValueError(f"data_ratio 必须在 (0,1]，当前={args.data_ratio}")
    if not (0 < args.eval_ratio <= 1.0):
        raise ValueError(f"eval_ratio 必须在 (0,1]，当前={args.eval_ratio}")

    # 设定 GPU（或 CPU）
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # 随机数种子
    setup_seed(args.seed)

    # 生成时间戳根目录
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = Path(args.results_root) / ts
    model_dir.mkdir(parents=True, exist_ok=True)

    # 备份代码
    backup_code(model_dir)

    print("--- GPU Diagnosis ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print("---------------------")
    print(f"[INFO] Results root: {model_dir}\n")

    accs, f1s = [], []

    # -------- 多 target 循环（每个 target_domains 一次训练/测试）--------
    for dataset_name in datasets:
        # 为该数据集构建一份 params（复用 argparse 的 Namespace，再覆盖 default）
        # 注意：这里不重新 parse_args（避免多次解析命令行），而是复制并覆盖
        from types import SimpleNamespace
        params = SimpleNamespace(**vars(args))

        # 覆盖本轮 target domain 与输出根目录
        params.target_domains = dataset_name
        params.model_dir = str(model_dir)   # 传给 Trainer（其内部会创建 fold{fold}/ 和 allfold/）

        # 建议给 run_name 加上数据集信息，便于 allfold/aggregate_results.csv 里区分
        params.run_name = f"{args.run_name}_{dataset_name}"

        # 打印本轮参数摘要
        print("===== PARAMS FOR TARGET =====")
        print(f"target_domains = {params.target_domains}")
        print(f"fold           = {params.fold}")
        print(f"run_name       = {params.run_name}")
        print(f"data_ratio     = {params.data_ratio}")
        print(f"epochs         = {params.epochs}")
        print(f"lr             = {params.lr}")
        print(f"batch_size     = {params.batch_size}")
        print(f"model_dir      = {params.model_dir}")
        print("=============================\n")

        # 训练与测试
        trainer = Trainer(params)
        test_acc, test_f1 = trainer.train()
        accs.append(test_acc)
        f1s.append(test_f1)

    # -------- 多 target 的总览 --------
    print("Per-target acc:", accs)
    print("Per-target f1 :", f1s)
    print("Mean acc, f1  :", float(np.mean(accs)), float(np.mean(f1s)))


if __name__ == '__main__':
    main()
