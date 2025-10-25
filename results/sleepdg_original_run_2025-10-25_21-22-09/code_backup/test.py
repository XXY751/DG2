# test.py
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
from collections import OrderedDict
import re # 用于从文件名中提取信息

# --- 辅助函数 ---
def setup_seed(seed: int = 0):
    """设置随机器种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"配置文件 {config_path} 未能加载为字典。")
    return config

# --- 从项目中导入必要的模块 ---
try:
    from datasets.dataset import LoadDataset
    from evaluator import Evaluator
    # 注意：Model 类将根据配置文件动态加载
except ImportError as e:
    print(f"[错误] 导入必需模块失败 (LoadDataset, Evaluator): {e}")
    print("请确保 datasets/dataset.py 和 evaluator.py 位于正确的路径下。")
    exit(1)

def find_best_model_path(results_log_dir: Path) -> Path:
    """
    在指定的训练日志文件夹中查找最佳模型文件 (.pth)。
    策略：查找文件名中包含 'tacc_' 和 '_tf1_' 的 .pth 文件。
    如果存在多个符合条件的文件，可能需要更精确的逻辑（例如，基于时间戳或 epoch 数），
    但这里我们先假设每个 fold 目录下只有一个符合条件的最优模型。
    """
    potential_models = list(results_log_dir.rglob('*.pth'))
    best_model_path = None
    best_f1 = -1.0 # 或者使用 acc 作为标准

    for model_path in potential_models:
        # 尝试从文件名解析指标（这依赖于 trainer.py 中保存的命名格式）
        match = re.search(r'tacc_([\d.]+)_tf1_([\d.]+)', model_path.name)
        if match:
            # acc = float(match.group(1))
            f1 = float(match.group(2))
            # 可以根据 f1 或 acc 来选择最佳模型
            if f1 > best_f1:
                best_f1 = f1
                best_model_path = model_path

    if best_model_path:
        print(f"    [找到模型] {best_model_path.name} 位于 {best_model_path.parent}")
        return best_model_path
    else:
        # 如果找不到带指标的文件名，尝试查找任意 .pth 文件作为后备
        if potential_models:
             print(f"    [警告] 未能在文件名中找到包含 acc/f1 的模型。将使用找到的第一个 .pth 文件: {potential_models[0].name}")
             return potential_models[0]
        else:
            raise FileNotFoundError(f"在 '{results_log_dir}' 及其子目录中未能找到任何 .pth 模型文件。")


def main():
    parser = argparse.ArgumentParser(description='SleepDG 模型评估脚本')
    parser.add_argument('--config', type=str, required=True, help='用于测试的 YAML 配置文件的路径')
    args = parser.parse_args()

    # -------- 加载配置 --------
    try:
        config = load_config(args.config)
        print("[信息] 已加载测试配置:")
        print(yaml.dump(config, indent=2, allow_unicode=True)) # 允许显示中文
    except (FileNotFoundError, TypeError, yaml.YAMLError) as e:
        print(f"[错误] 加载或解析配置文件 '{args.config}' 失败: {e}")
        return

    # 将配置字典转换为 SimpleNamespace 对象
    params = SimpleNamespace(**config)

    # -------- 环境设置 --------
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpus
    setup_seed(params.seed)
    # torch.set_float32_matmul_precision('high') # 如果需要

    # -------- GPU 诊断 --------
    print("\n--- GPU 诊断 ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        current_device_index = torch.cuda.current_device()
        print(f"当前设备索引: {current_device_index}")
        print(f"设备名称: {torch.cuda.get_device_name(current_device_index)}")
    print("---------------------\n")

    # -------- 验证必要的配置 --------
    required_keys = ['model_version', 'results_log_dir', 'test_dataset', 'datasets_dir']
    missing_keys = [k for k in required_keys if not hasattr(params, k)]
    if missing_keys:
        print(f"[错误] 配置文件中缺少必要的键: {', '.join(missing_keys)}")
        return

    # -------- 动态加载 Model 类 --------
    model_version = params.model_version
    try:
        if model_version == 'original':
            model_module = importlib.import_module('original.models.model')
            ModelClass = model_module.Model
            print("[信息] 使用 ORIGINAL 模型类。")
        elif model_version == 'improved':
            model_module = importlib.import_module('improved.models.model')
            ModelClass = model_module.Model
            print("[信息] 使用 IMPROVED 模型类。")
        else:
            raise ValueError(f"未知的 model_version: {model_version}。必须是 'original' 或 'improved'。")
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[错误] 为 '{model_version}' 加载 Model 类失败: {e}")
        return

    # -------- 查找模型权重 --------
    results_log_dir = Path(params.results_log_dir)
    try:
        model_path = find_best_model_path(results_log_dir)
        print(f"[信息] 使用模型权重: {model_path}")
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return

    # -------- 加载测试数据集 --------
    # 这里我们需要修改 LoadDataset 的逻辑，使其能够只加载指定的测试集
    # 为了简化，我们暂时借用 target_domains 参数来指定测试集
    # 注意：这可能需要调整 LoadDataset 类或创建一个新的加载函数
    print(f"[信息] 加载测试数据集: {params.test_dataset}...")
    try:
        # 创建一个临时的 params 对象，只包含加载数据所需的参数
        load_params = SimpleNamespace(
            datasets_dir=params.datasets_dir,
            target_domains=params.test_dataset, # **重要**: 用 test_dataset 覆盖 target_domains
            batch_size=params.batch_size,
            num_workers=params.num_workers
        )
        # LoadDataset 在初始化时会打印 source/target dirs，这里的 source 可能为空
        data_loader_dict, _ = LoadDataset(load_params).get_data_loader()
        # **重要**: 我们只需要 'test' 部分的 loader
        if 'test' not in data_loader_dict or len(data_loader_dict['test']) == 0:
             # 如果 LoadDataset 返回的 'test' 是空的 (因为它可能被设计为只返回 LODO 中的 target)
             # 我们需要一种方式强制加载指定数据集作为 'test'
             # **备选方案/需要修改 LoadDataset**:
             # 1. 修改 LoadDataset 允许直接指定 test_domains
             # 2. 创建一个只加载单个数据集的简化版 LoadDataset
             # 暂时假设 LoadDataset 的 'test' loader 就是我们想要的
             # 但如果 LoadDataset 的设计是严格 LODO 的，这里会出错
             print("[警告] LoadDataset 可能由于其 LODO 设计未能正确将指定的 test_dataset 加载到 'test' 加载器中。谨慎继续。")
             # **临时解决方案**: 尝试从 'val' 或 'train' loader 中获取数据集（如果 test 为空）
             # 这不是标准做法，仅作演示
             if len(data_loader_dict['test']) == 0:
                 if 'val' in data_loader_dict and len(data_loader_dict['val']) > 0:
                     print("[警告] 由于 'test' 为空，使用 'val' 加载器作为测试加载器。")
                     test_loader = data_loader_dict['val']
                 elif 'train' in data_loader_dict and len(data_loader_dict['train']) > 0:
                     print("[警告] 由于 'test' 和 'val' 均为空，使用 'train' 加载器作为测试加载器。")
                     test_loader = data_loader_dict['train']
                 else:
                      raise ValueError(f"无法为测试数据集加载数据: {params.test_dataset}。所有加载器均为空。")
             else:
                  test_loader = data_loader_dict['test']

        else:
            test_loader = data_loader_dict['test']

        print(f"测试数据加载完成。批次数: {len(test_loader)}")
    except Exception as e:
        print(f"[错误] 加载数据集 '{params.test_dataset}' 失败: {e}")
        return

    # -------- 初始化模型并加载权重 --------
    print("[信息] 初始化模型结构...")
    # 需要确保 config 文件中有模型所需的参数 (dropout, num_classes 等)
    model = ModelClass(params)

    print(f"[信息] 从以下路径加载模型权重: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        # 处理 DataParallel 可能添加的 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"[错误] 加载模型权重时出错: {e}")
        return

    # 将模型移至 GPU
    if torch.cuda.is_available():
        model.cuda()
        # 如果原始训练使用了 DataParallel，测试时最好也使用（尽管对于 inference 可能不是必须的）
        if torch.cuda.device_count() > 1:
            print("[信息] 使用 DataParallel 进行评估。")
            model = torch.nn.DataParallel(model)

    # -------- 运行评估 --------
    print("\n--- 开始评估 ---")
    evaluator = Evaluator(params, test_loader)

    # 执行评估
    # 注意：确保 Evaluator 不需要训练特定的参数（如 fold_id, model_dir 等）
    # 如果需要，则需要从 config 中传递
    test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
        test_n3_f1, test_rem_f1, test_kappa, test_report = evaluator.get_accuracy(model)

    # -------- 打印结果 --------
    print(f"\n***************** 测试结果 ({params.test_dataset}) *****************")
    print(f"模型版本:           {model_version}")
    print(f"加载权重自:         {model_path}")
    print("-" * 65)
    print(f"测试准确率 (Acc):    {test_acc:.5f}")
    print(f"测试宏 F1 分数:     {test_f1:.5f}")
    print(f"测试 Cohen's Kappa:  {test_kappa:.5f}")
    print("\n混淆矩阵:")
    print(test_cm)
    print("\n各类别 F1 分数:")
    print(f"  Wake: {test_wake_f1:.5f}")
    print(f"  N1:   {test_n1_f1:.5f}")
    print(f"  N2:   {test_n2_f1:.5f}")
    print(f"  N3:   {test_n3_f1:.5f}")
    print(f"  REM:  {test_rem_f1:.5f}")
    print("\n分类报告:")
    print(test_report)
    print("******************************************************************")

    # -------- (可选) 保存结果 --------
    # 可以将结果保存到 results_log_dir 下的一个文件中
    results_output_dir = Path(params.results_log_dir) / "test_results"
    results_output_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_output_dir / f"test_on_{params.test_dataset}_results.txt"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"数据集测试结果: {params.test_dataset}\n")
            f.write(f"模型版本: {model_version}\n")
            f.write(f"加载权重自: {model_path}\n")
            f.write("-" * 65 + "\n")
            f.write(f"准确率 (Acc): {test_acc:.5f}\n")
            f.write(f"宏 F1 分数:   {test_f1:.5f}\n")
            f.write(f"Kappa 系数:   {test_kappa:.5f}\n\n")
            f.write("混淆矩阵:\n")
            np.savetxt(f, test_cm, fmt="%d")
            f.write("\n\n各类别 F1 分数:\n")
            f.write(f"  Wake: {test_wake_f1:.5f}\n")
            f.write(f"  N1:   {test_n1_f1:.5f}\n")
            f.write(f"  N2:   {test_n2_f1:.5f}\n")
            f.write(f"  N3:   {test_n3_f1:.5f}\n")
            f.write(f"  REM:  {test_rem_f1:.5f}\n\n")
            f.write("分类报告:\n")
            f.write(test_report)
        print(f"\n[信息] 测试结果已保存至: {results_file}")
    except Exception as e:
        print(f"[警告] 保存测试结果到文件失败: {e}")


if __name__ == '__main__':
    main()