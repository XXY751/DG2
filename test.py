# test.py (重写版本 - 手动配置)

import torch
import torch.nn as nn
import numpy as np
import random
import os
from collections import OrderedDict
from types import SimpleNamespace

from datasets.dataset import LoadDataset
from models.model import Model
from evaluator import Evaluator
from utils.visualize import Visualization


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
    # --- 核心参数 ---
    # !! 必须 !!: 指向您训练好的 .pth 模型文件的路径
    model_path = '/data/lijinyang/SleepSLeep/results/2025-10-14_09-42-44/run__fold0__20251014-094245/fold0_tacc_0.77675_tf1_0.72045_run__fold0__20251014-094245.pth'

    # !! 必须 !!: 指定您想评估的目标数据集名称
    target_domains = 'sleep-edfx'

    # -- GPU 设置 --
    gpus = '0'

    # -- 数据加载参数 --
    datasets_dir = '/data/lijinyang/SleepSLeep/datasets_dir'
    batch_size = 512
    # !! 重要 !!: 如果程序无声退出，请将此值减小！建议从 4 开始尝试。
    num_workers = 4

    # -- 模型架构参数 (必须与训练时保持一致) --
    num_of_classes = 5
    dropout = 0.1
    projection_type = 'diag'
    lowrank_rank = 32
    num_domains = 4  # 源域数量

    # -- 其他参数 (保持与训练时一致) --
    enable_stats_alignment = 1
    anchor_momentum = 0.9
    label_smoothing = 0.0


# ==============================================================================
# 主程序逻辑 (通常无需修改)
# ==============================================================================
def main():
    # 将 Config 类的属性转换为 SimpleNamespace 对象，以模拟 argparse 的行为
    params = SimpleNamespace(**{k: v for k, v in Config.__dict__.items() if not k.startswith('__')})

    print("--- Evaluation Parameters ---")
    for k, v in params.__dict__.items():
        print(f"{k}: {v}")
    print("---------------------------\n")

    # --- 环境设置 ---
    setup_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpus
    torch.set_float32_matmul_precision('high')

    # --- 加载测试数据 ---
    print(f"Loading test data for target domain: {params.target_domains}...")
    data_loader, _ = LoadDataset(params).get_data_loader()
    test_loader = data_loader['test']
    print(f"Test data loaded. Number of batches: {len(test_loader)}")

    # --- 初始化模型并加载权重 ---
    print("Initializing model...")
    model = Model(params)

    print(f"Loading model weights from: {params.model_path}")
    try:
        state_dict = torch.load(params.model_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.cuda()

    # --- 运行评估 ---
    print("\n--- Running Evaluation ---")
    evaluator = Evaluator(params, test_loader)

    test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
        test_n3_f1, test_rem_f1, test_kappa, test_report = evaluator.get_accuracy(model)

    print("*************************** Test Results ***************************")
    print(f"Test Accuracy: {test_acc:.5f}")
    print(f"Test Macro F1-Score: {test_f1:.5f}")
    print(f"Test Cohen's Kappa: {test_kappa:.5f}")
    print("\nConfusion Matrix:")
    print(test_cm)
    print("\nPer-class F1 Scores:")
    print(f"  Wake: {test_wake_f1:.5f}")
    print(f"  N1:   {test_n1_f1:.5f}")
    print(f"  N2:   {test_n2_f1:.5f}")
    print(f"  N3:   {test_n3_f1:.5f}")
    print(f"  REM:  {test_rem_f1:.5f}")
    print("\nClassification Report:")
    print(test_report)
    print("******************************************************************")


if __name__ == '__main__':
    main()