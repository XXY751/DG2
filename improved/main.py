# main.py (修改版 - 标准 LODO-CV 流程)

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
import pandas as pd
from types import SimpleNamespace

# 假设 Trainer 类在 newtraining.py 中，并且 train() 方法返回一个字典
from newtraining import Trainer
# 假设 write_aggregate_row 在 utils/allutils.py 中
from utils.allutils import write_aggregate_row

# =============== 数据集列表 ===============

datasets = [
    'sleep-edfx', # LODO Fold 0
    'HMC',        # LODO Fold 1
    'ISRUC',      # LODO Fold 2
    'SHHS1',      # LODO Fold 3
    'P2018',      # LODO Fold 4
    'ABC',
]

def setup_seed(seed: int = 0):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # 可能影响性能，但为了复现性开启

def backup_code(dst_root: Path):
    """备份当前项目中关键代码到 results/<ts>/code_backup/"""
    code_backup_dir = dst_root / 'code_backup'
    code_backup_dir.mkdir(parents=True, exist_ok=True)

    # 备份顶层 .py 文件
    for filename in os.listdir('.'):
        if filename.endswith('.py'):
            try:
                shutil.copy(filename, code_backup_dir / filename)
            except Exception as e:
                print(f"Warning: Could not back up {filename}: {e}")

    # 备份常见代码目录
    for dirname in ['models', 'losses', 'datasets', 'utils', 'prepare_datasets', 'scripts']:
        if os.path.isdir(dirname):
            dst_dir = code_backup_dir / dirname
            try:
                shutil.copytree(dirname, dst_dir, dirs_exist_ok=True)
            except Exception as e:
                 print(f"Warning: Could not back up directory {dirname}: {e}")

def build_argparser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='SleepDG Runner (Leave-One-Dataset-Out Evaluation)')

    # ==== 基础训练参数 ====
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--num_of_classes', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=16, help='dataloader workers, adjust based on system memory')
    parser.add_argument('--datasets_dir', type=str, default='/data/lijinyang/SleepSLeep/datasets_dir', help='path to preprocessed datasets')

    # ==== 设备/种子 ====
    parser.add_argument('--seed', type=int, default=443, help='random seed')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='CUDA_VISIBLE_DEVICES, e.g. "0,1" or "0" or "" for CPU')

    # ==== 目录 ====
    parser.add_argument('--results_root', type=str, default='results', help='root directory for saving results (timestamped subfolder will be created)')

    # ==== 训练控制：数据抽样比例 ====
    parser.add_argument('--data_ratio', type=float, default=1,
                        help='fraction of training data to use (0,1]; e.g., 0.1 for debugging with 10% data')
    parser.add_argument('--eval_ratio', type=float, default=1,
                        help='fraction of validation/test data to use (0,1]; typically 1.0')

    # ==== 运行命名 ====
    parser.add_argument('--run_name', type=str, default='lodo_exp', help='base name for this LODO evaluation run')

    # ==== 模型超参数 (与 newtraining.py 中的 Trainer 保持一致) ====
    parser.add_argument('--projection_type', type=str, default='diag', choices=['diag', 'lowrank'], help='type of domain projection')
    parser.add_argument('--lowrank_rank', type=int, default=32, help='rank for lowrank projection')
    parser.add_argument('--enable_stats_alignment', type=int, default=1, choices=[0, 1], help='enable lightweight statistics alignment (1=True, 0=False)')
    parser.add_argument('--anchor_momentum', type=float, default=0.9, help='momentum for EMA anchor updates')
    parser.add_argument('--lambda_caa', type=float, default=0.3, help='weight for CAA loss')
    parser.add_argument('--lambda_stat', type=float, default=0.2, help='weight for statistics alignment loss')
    parser.add_argument('--lambda_Areg', type=float, default=0.1, help='weight for projection matrix regularization')
    parser.add_argument('--lambda_ae', type=float, default=1.0, help='weight for AE reconstruction loss')
    parser.add_argument('--lambda_coral', type=float, default=1.0, help='weight for CORAL loss')
    # num_domains 在 LODO 中通常是 len(datasets) - 1
    parser.add_argument('--num_domains', type=int, default=len(datasets)-1, help='number of source domains (automatically set based on datasets list)')
    # ==== 目标域和 Fold 参数 (由脚本内部设置，命令行无需关心) ====
    # parser.add_argument('--target_domains', type=str, default='') # 内部设置
    # parser.add_argument('--fold', type=int, default=0) # 内部设置

    return parser

# ----- 计算平均值和标准差的辅助函数 -----
def calculate_stats(results_list: list, key: str) -> tuple[float, float]:
    """从结果字典列表中提取指定 key 的值，计算均值和标准差"""
    values = [res[key] for res in results_list if key in res and isinstance(res[key], (int, float))]
    if not values:
        return 0.0, 0.0 # 返回 0 均值和 0 标准差
    mean_val = np.mean(values)
    std_val = np.std(values)
    return mean_val, std_val

def main():
    parser = build_argparser()
    args = parser.parse_args()

    # -------- 参数校验 --------
    if not (0 < args.data_ratio <= 1.0):
        raise ValueError(f"data_ratio 必须在 (0,1]，当前={args.data_ratio}")
    if not (0 < args.eval_ratio <= 1.0):
        raise ValueError(f"eval_ratio 必须在 (0,1]，当前={args.eval_ratio}")
    # 自动设置 num_domains，确保与 LODO 匹配
    expected_num_source_domains = len(datasets) - 1
    if args.num_domains != expected_num_source_domains:
        print(f"[INFO] Adjusting num_domains from {args.num_domains} to {expected_num_source_domains} based on datasets list for LODO.")
        args.num_domains = expected_num_source_domains

    # -------- 环境设置 --------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    setup_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # -------- 结果目录与代码备份 --------
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_basedir = Path(args.results_root) / f"{args.run_name}_{ts}" # 目录名包含 run_name
    results_basedir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved under: {results_basedir}")
    backup_code(results_basedir)
    print("[INFO] Code backup completed.")

    # -------- GPU 诊断 --------
    print("\n--- GPU Diagnosis ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("---------------------\n")

    # -------- 初始化结果存储 --------
    # 存储 LODO 中每次留一的详细结果字典
    all_lodo_results = []
    # 定义需要计算平均值的指标键名列表
    metrics_to_average = [
        'test_acc', 'test_f1', 'test_kappa',
        'wake_f1', 'n1_f1', 'n2_f1', 'n3_f1', 'rem_f1'
    ]

    # -------- LODO 循环 (外层循环现在代表不同的 LODO 折) --------
    # 使用 enumerate 获取折索引 (lodo_fold_index) 和目标数据集名称
    for lodo_fold_index, target_dataset_name in enumerate(datasets):
        print(f"\n{'='*15} LODO Fold {lodo_fold_index}/{len(datasets)-1}: Target = {target_dataset_name} {'='*15}")

        # -- 为本次 LODO 运行构建 params --
        params = SimpleNamespace(**vars(args))
        params.target_domains = target_dataset_name # 设置当前目标域
        params.model_dir = str(results_basedir) # Trainer 会在此目录下创建 fold{lodo_fold_index}/
        params.fold = lodo_fold_index # !! 使用 LODO 循环的索引作为 fold !!
        # run_name 包含目标域信息和 LODO 折信息
        params.run_name = f"{args.run_name}_target_{target_dataset_name}_lodo{lodo_fold_index}"

        # -- 打印参数摘要 --
        print("===== PARAMS FOR THIS LODO FOLD =====")
        print(f"target_domains = {params.target_domains}")
        print(f"fold (LODO index) = {params.fold}")
        print(f"run_name       = {params.run_name}")
        # ... 其他参数打印 ...
        print(f"epochs         = {params.epochs}")
        print(f"lr             = {params.lr}")
        print(f"batch_size     = {params.batch_size}")
        print(f"model_dir root = {params.model_dir}")
        print(f"num_workers    = {params.num_workers}")
        print(f"num_source_domains = {params.num_domains}")
        print("====================================\n")

        try:
            trainer = Trainer(params)
            # train() 返回包含所有测试指标的字典
            lodo_result_dict = trainer.train()

            if lodo_result_dict and isinstance(lodo_result_dict, dict):
                # 将本次 LODO 运行的结果添加到总列表中
                all_lodo_results.append(lodo_result_dict)
                # 打印关键指标
                acc = lodo_result_dict.get('test_acc', float('nan'))
                f1 = lodo_result_dict.get('test_f1', float('nan'))
                kappa = lodo_result_dict.get('test_kappa', float('nan'))
                print(f"LODO Fold {lodo_fold_index} (Target={target_dataset_name}) Test -> Acc: {acc:.5f}, F1: {f1:.5f}, Kappa: {kappa:.5f}")
            else:
                print(f"[WARN] Trainer for LODO Fold {lodo_fold_index} (Target={target_dataset_name}) returned invalid or empty results.")
        except Exception as e:
             print(f"[ERROR] An error occurred during LODO Fold {lodo_fold_index} (Target={target_dataset_name}): {e}")
             # import traceback
             # traceback.print_exc() # 打印详细错误堆栈以供调试
             # 决定是否中断或继续下一个 LODO 运行
             continue # 继续下一个 LODO 折

    # -------- 所有 LODO 运行处理完毕后的总平均计算与记录 --------
    print(f"\n======== Overall Mean Results Across {len(datasets)} LODO Folds ========")
    aggregate_csv_path = results_basedir / "allfold" / "aggregate_results.csv" # 聚合文件路径

    if all_lodo_results and len(all_lodo_results) == len(datasets): # 确保所有折都成功运行
        print("Metric               | Mean ± Std across LODO folds")
        print("---------------------|--------------------------")
        # 存储平均结果
        avg_row = {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "run_id": f"{args.run_name}_LODO_avg", # 标记为 LODO 平均行
            "fold": "mean", # 标记折为 'mean'
            "best_val_acc": "", # LODO 平均的最佳验证指标意义不大
            "best_val_f1": "",
            "model_path": "N/A" # 平均行没有具体的模型路径
        }
        # 计算并打印每个指标的均值和标准差
        for key in metrics_to_average:
            mean_val, std_val = calculate_stats(all_lodo_results, key)
            print(f"{key:<20} | {mean_val:.5f} ± {std_val:.5f}")
            avg_row[key] = f"{mean_val:.5f} +/- {std_val:.5f}" # 存储格式化字符串
        try:
            # 再次确认 utils/allutils.py 的 write_aggregate_row 表头包含所有 key
            write_aggregate_row(aggregate_csv_path, row=avg_row)
            print(f"\nAppended LODO average results to {aggregate_csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to append LODO average results to CSV: {e}")
    elif all_lodo_results:
         print(f"[WARN] Only {len(all_lodo_results)} out of {len(datasets)} LODO folds completed successfully. Average results might be unreliable and were not written to the aggregate CSV.")
    else:
        print("No LODO runs completed successfully, cannot calculate average results.")
    print("======================================================")
    print(f"\n[INFO] All LODO runs finished. Results are saved in: {results_basedir}")


if __name__ == '__main__':
    main()