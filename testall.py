# test_all.py (新文件)

import torch
import numpy as np
import random
import os
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import re

# 从您的项目中导入必要的模块
from datasets.dataset import LoadDataset
from models.model import Model
from evaluator import Evaluator


# --- 辅助函数 ---
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ==============================================================================
# 唯一需要您修改的区域
# ==============================================================================
class Config:
    # !! 必须 !!: 指向包含所有折训练结果的时间戳总目录
    # 例如: 'results/2025-10-14_09-42-44'
    results_dir = '/data/lijinyang/SleepSLeep/results/2025-10-14_09-42-44'

    # -- GPU 设置 --
    gpus = '0'

    # -- 数据加载参数 --
    datasets_dir = '/data/lijinyang/SleepSLeep/datasets_dir'
    batch_size = 512
    num_workers = 4

    # -- 模型架构参数 (必须与训练时保持一致) --
    num_of_classes = 5
    dropout = 0.1
    projection_type = 'diag'
    lowrank_rank = 32
    num_domains = 4  # 源域数量

    # -- 其他参数 --
    enable_stats_alignment = 1
    anchor_momentum = 0.9
    label_smoothing = 0.0


# ==============================================================================
# 主程序逻辑 (通常无需修改)
# ==============================================================================
def find_model_path_for_target(results_dir: Path, target_domain: str) -> Path:
    """在总结果目录中，为指定的目标域自动查找对应的模型文件路径。"""
    print(f"\n---> Searching for model trained with target_domain='{target_domain}'...")

    # 模型文件名通常包含 'tacc' 和 '.pth'
    for model_path in results_dir.rglob('*.pth'):
        # 从路径中提取 run_name 来判断目标域
        # 路径示例: .../run_exp_sleep-edfx__fold0_.../fold0/fold0_...
        match = re.search(r'run_exp_([a-zA-Z0-9_-]+)__', str(model_path))
        if match:
            domain_in_path = match.group(1)
            if domain_in_path == target_domain:
                print(f"    [Found Model] {model_path.name}")
                return model_path

    raise FileNotFoundError(
        f"Could not automatically find a model file for target domain '{target_domain}' in '{results_dir}'")


def main():
    # 将 Config 类的属性转换为 SimpleNamespace 对象
    base_params = SimpleNamespace(**{k: v for k, v in Config.__dict__.items() if not k.startswith('__')})

    setup_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = base_params.gpus
    torch.set_float32_matmul_precision('high')

    # 定义所有需要被评估的目标域
    all_target_datasets = ['sleep-edfx', 'HMC', 'ISRUC', 'SHHS1', 'P2018']

    # 用于存储所有结果的字典
    all_results = {}

    for target_domain in all_target_datasets:
        print(f"\n{'=' * 20} EVALUATING TARGET: {target_domain.upper()} {'=' * 20}")

        try:
            # 1. 自动查找模型路径
            model_path = find_model_path_for_target(Path(base_params.results_dir), target_domain)

            # 2. 准备本轮评估的参数
            params = SimpleNamespace(**vars(base_params))
            params.target_domains = target_domain

            # 3. 加载对应的数据集
            print(f"    Loading test data for '{target_domain}'...")
            data_loader, _ = LoadDataset(params).get_data_loader()
            test_loader = data_loader['test']

            # 4. 初始化模型并加载权重
            print("    Initializing model...")
            model = Model(params)

            state_dict = torch.load(model_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model.cuda()

            # 5. 运行评估
            print("    Running evaluation...")
            evaluator = Evaluator(params, test_loader)

            results = evaluator.get_accuracy(model)
            all_results[target_domain] = results  # 保存所有返回值

        except FileNotFoundError as e:
            print(f"    [SKIPPING] {e}")
            continue
        except Exception as e:
            print(f"    [ERROR] An unexpected error occurred while processing {target_domain}: {e}")
            continue

    # --- 打印最终的总结报告 ---
    print(f"\n\n{'=' * 25} FINAL SUMMARY {'=' * 25}")
    if not all_results:
        print("No models were evaluated.")
        return

    summary_data = []
    for domain, metrics in all_results.items():
        acc, f1, _, _, _, _, _, _, kappa, _ = metrics
        summary_data.append([domain, acc, f1, kappa])

    print("{:<15} | {:<15} | {:<15} | {:<15}".format("Target Domain", "Accuracy", "Macro F1", "Kappa"))
    print("-" * 65)
    for row in summary_data:
        print("{:<15} | {:<15.4f} | {:<15.4f} | {:<15.4f}".format(row[0], row[1], row[2], row[3]))

    # 计算并打印平均值
    if len(summary_data) > 0:
        mean_acc = np.mean([row[1] for row in summary_data])
        mean_f1 = np.mean([row[2] for row in summary_data])
        mean_kappa = np.mean([row[3] for row in summary_data])
        print("-" * 65)
        print("{:<15} | {:<15.4f} | {:<15.4f} | {:<15.4f}".format("AVERAGE", mean_acc, mean_f1, mean_kappa))

    print(f"{'=' * 65}\n")


if __name__ == '__main__':
    main()