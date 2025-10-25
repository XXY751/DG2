# main.py
import argparse
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import pandas as pd
import importlib # 用于动态导入











# --- 全局设置 ---
# (可选) 限制线程数
_DEFAULT_THREADS = "8"
for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"]:
    os.environ.setdefault(k, _DEFAULT_THREADS)

# --- 数据集列表 (LODO) ---
datasets_lodo_order = [
    'sleep-edfx', # LODO Fold 0
    'HMC',        # LODO Fold 1
    'ISRUC',      # LODO Fold 2
    'SHHS1',      # LODO Fold 3
    'P2018',      # LODO Fold 4
]

# --- 辅助函数 ---
def setup_seed(seed: int = 0):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def backup_code(dst_root: Path):
    """备份当前项目中关键代码到 results/<ts>/code_backup/"""
    code_backup_dir = dst_root / 'code_backup'
    code_backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Backing up code to {code_backup_dir}")

    # 备份顶层 .py 文件和 configs 目录
    for item in os.listdir('.'):
        source_path = Path(item)
        dest_path = code_backup_dir / source_path.name
        if source_path.is_file() and source_path.suffix == '.py':
            try:
                shutil.copy(str(source_path), str(dest_path))
            except Exception as e:
                print(f"Warning: Could not back up {item}: {e}")
        elif source_path.is_dir() and item == 'configs':
             try:
                shutil.copytree(str(source_path), str(dest_path), dirs_exist_ok=True)
             except Exception as e:
                print(f"Warning: Could not back up directory {item}: {e}")


    # 备份核心代码目录 (original, improved, models, losses, datasets, utils)
    for dirname in ['original', 'improved', 'models', 'losses', 'datasets', 'utils']:
        source_dir = Path(dirname)
        if source_dir.is_dir():
            dst_dir = code_backup_dir / dirname
            try:
                # 忽略 __pycache__ 目录
                shutil.copytree(str(source_dir), str(dst_dir), dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            except Exception as e:
                 print(f"Warning: Could not back up directory {dirname}: {e}")

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file {config_path} did not load as a dictionary.")
    return config

def calculate_stats(results_list: list, key: str) -> tuple[float, float]:
    """从结果字典列表中提取指定 key 的值，计算均值和标准差"""
    values = [res[key] for res in results_list if key in res and isinstance(res[key], (int, float))]
    if not values:
        return 0.0, 0.0
    mean_val = np.mean(values)
    std_val = np.std(values)
    return mean_val, std_val

# --- 从 allutils 导入 write_aggregate_row ---
# 确保 utils/allutils.py 文件存在且包含 write_aggregate_row 函数
try:
    from utils.allutils import write_aggregate_row
except ImportError:
    print("[ERROR] Could not import 'write_aggregate_row' from 'utils.allutils'. "
          "Make sure the file and function exist.")
    # 定义一个空函数以避免程序崩溃，但聚合结果将无法保存
    def write_aggregate_row(path, row):
        print(f"[WARN] write_aggregate_row not available. Skipping writing row to {path}: {row}")


def main():
    parser = argparse.ArgumentParser(description='SleepDG Runner with YAML Configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # -------- 加载配置 --------
    try:
        config = load_config(args.config)
        print("[INFO] Loaded configuration:")
        print(yaml.dump(config, indent=2))
    except (FileNotFoundError, TypeError, yaml.YAMLError) as e:
        print(f"[ERROR] Failed to load or parse config file '{args.config}': {e}")
        return

    # -------- 环境设置 --------
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('gpus', "")
    setup_seed(config.get('seed', 42))
    torch.backends.cudnn.benchmark = True
    # torch.set_float32_matmul_precision('high') # 取消注释如果你用的是较新 PyTorch 版本

    # -------- 结果目录与代码备份 --------
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_basedir = Path(config.get('results_root', './results')) / f"{config.get('run_name', 'run')}_{ts}"
    results_basedir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved under: {results_basedir}")
    # 备份代码
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

    # -------- 动态加载 Trainer --------
    model_version = config.get('model_version', 'improved') # 默认为改进版
    try:
        if model_version == 'original':
            trainer_module = importlib.import_module('original.trainer')
            TrainerClass = trainer_module.Trainer
            print("[INFO] Using ORIGINAL Trainer.")
        elif model_version == 'improved':
            trainer_module = importlib.import_module('improved.trainer')
            TrainerClass = trainer_module.Trainer # 假设改进版的类名也是 Trainer
            print("[INFO] Using IMPROVED Trainer.")
        else:
            raise ValueError(f"Unknown model_version: {model_version}. Must be 'original' or 'improved'.")
    except ImportError as e:
        print(f"[ERROR] Failed to import trainer for model_version '{model_version}': {e}")
        return
    except AttributeError as e:
         print(f"[ERROR] Trainer class not found in module for model_version '{model_version}': {e}")
         return

    # -------- 初始化结果存储 --------
    all_lodo_results = []
    # 从 config 中读取 model_type 以便区分指标
    model_type = config.get('model_type', 'unknown')
    metrics_to_average = [
        'test_acc', 'test_f1',
        # 改进版返回了 kappa
        'test_kappa' if model_version == 'improved' else None,
        'wake_f1', 'n1_f1', 'n2_f1', 'n3_f1', 'rem_f1'
    ]
    metrics_to_average = [m for m in metrics_to_average if m is not None] # 移除 None

    # -------- LODO 循环 --------
    num_total_datasets = len(datasets_lodo_order)
    expected_num_source = num_total_datasets - 1
    if config.get('num_domains', expected_num_source) != expected_num_source:
         print(f"[WARN] Config 'num_domains' ({config.get('num_domains')}) does not match LODO setup ({expected_num_source}). Using {expected_num_source}.")
         config['num_domains'] = expected_num_source


    for lodo_fold_index, target_dataset_name in enumerate(datasets_lodo_order):
        print(f"\n{'='*15} LODO Fold {lodo_fold_index}/{num_total_datasets-1}: Target = {target_dataset_name} {'='*15}")

        # -- 为本次 LODO 运行构建 params 对象 --
        fold_config = config.copy() # 复制基础配置
        fold_config['target_domains'] = target_dataset_name
        fold_config['model_dir'] = str(results_basedir) # Trainer 会在此下创建 fold{i}
        fold_config['fold'] = lodo_fold_index
        # 更新 run_name 以包含 fold 信息
        base_run_name = config.get('run_name', 'run')
        fold_config['run_name'] = f"{base_run_name}_target_{target_dataset_name}_lodo{lodo_fold_index}"

        # 将配置字典转换为 SimpleNamespace 对象，方便 Trainer 使用 . 访问
        params = SimpleNamespace(**fold_config)

        # -- 打印参数摘要 --
        print("===== PARAMS FOR THIS LODO FOLD =====")
        print(f"model_version  = {params.model_version}")
        print(f"target_domains = {params.target_domains}")
        print(f"fold (LODO index) = {params.fold}")
        print(f"run_name       = {params.run_name}")
        print(f"model_dir root = {params.model_dir}")
        print(f"epochs         = {params.epochs}")
        print(f"lr             = {params.lr}")
        print(f"batch_size     = {params.batch_size}")
        print(f"num_workers    = {params.num_workers}")
        print(f"num_domains    = {params.num_domains}") # num_domains 指的是源域数量
        # ... 可以添加更多你想打印的参数 ...
        print("====================================\n")

        try:
            # 实例化选择的 Trainer
            trainer = TrainerClass(params)
            # train() 返回测试结果字典
            lodo_result_dict = trainer.train() # 假设 train 方法返回包含测试指标的字典

            if lodo_result_dict and isinstance(lodo_result_dict, dict):
                all_lodo_results.append(lodo_result_dict)
                # 打印关键指标
                acc = lodo_result_dict.get('test_acc', float('nan'))
                f1 = lodo_result_dict.get('test_f1', float('nan'))
                kappa_str = f", Kappa: {lodo_result_dict.get('test_kappa', 'N/A'):.5f}" if 'test_kappa' in lodo_result_dict else ""
                print(f"LODO Fold {lodo_fold_index} (Target={target_dataset_name}) Test -> Acc: {acc:.5f}, F1: {f1:.5f}{kappa_str}")
            else:
                print(f"[WARN] Trainer for LODO Fold {lodo_fold_index} returned invalid results.")
        except Exception as e:
             print(f"[ERROR] An error occurred during LODO Fold {lodo_fold_index}: {e}")
             import traceback
             traceback.print_exc() # 打印详细错误堆栈
             continue # 继续下一个 LODO 折

    # -------- LODO 结束后的总结 --------
    print(f"\n======== Overall Mean Results Across {len(datasets_lodo_order)} LODO Folds ({model_type}) ========")
    # 聚合 CSV 文件路径在 allfold 目录下
    aggregate_csv_path = results_basedir / "allfold" / "aggregate_results.csv"
    # 确保 allfold 目录存在 (Trainer 可能已创建，但以防万一)
    aggregate_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if all_lodo_results and len(all_lodo_results) == num_total_datasets:
        print("Metric               | Mean ± Std across LODO folds")
        print("---------------------|--------------------------")

        avg_row = {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "run_id": f"{config.get('run_name', 'run')}_LODO_avg",
            "fold": "mean",
            "best_val_acc": "N/A",
            "best_val_f1": "N/A",
            "model_path": "N/A"
        }

        # 计算并打印均值和标准差
        for key in metrics_to_average:
            mean_val, std_val = calculate_stats(all_lodo_results, key)
            print(f"{key:<20} | {mean_val:.5f} ± {std_val:.5f}")
            # CSV 中保存数值，或者你可以选择保存格式化字符串
            avg_row[key] = f"{mean_val:.5f} +/- {std_val:.5f}" # 保存格式化字符串便于阅读

        try:
            write_aggregate_row(aggregate_csv_path, row=avg_row)
            print(f"\nAppended LODO average results to {aggregate_csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to append LODO average results to CSV: {e}")
    elif all_lodo_results:
         print(f"[WARN] Only {len(all_lodo_results)} out of {num_total_datasets} LODO folds completed successfully. Average results not calculated or saved.")
    else:
        print("No LODO runs completed successfully.")
    print("======================================================")
    print(f"\n[INFO] All LODO runs finished. Results are in: {results_basedir}")

if __name__ == '__main__':
    main()