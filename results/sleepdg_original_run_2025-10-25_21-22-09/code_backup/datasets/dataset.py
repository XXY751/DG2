# datasets/dataset.py (修改版)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# 确保 utils.util 存在且包含 to_tensor
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from types import SimpleNamespace # 确保导入 SimpleNamespace
from pathlib import Path # <-- 修复：在这里添加导入
try:
    from utils.util import to_tensor
except ImportError:
    print("[警告] 无法从 utils.util 导入 to_tensor，将使用本地定义。")
    # 如果导入失败，提供一个备用定义
    def to_tensor(array):
        return torch.from_numpy(np.array(array)).float()
import os
import random
from types import SimpleNamespace # 确保导入 SimpleNamespace

class CustomDataset(Dataset):
    def __init__(
            self,
            seqs_labels_path_pair
    ):
        super(CustomDataset, self).__init__()
        # 存储 [(seq_path, label_path, subject_id), ...]
        self.seqs_labels_path_pair = seqs_labels_path_pair

    def __len__(self):
        return len(self.seqs_labels_path_pair)

    def __getitem__(self, idx):
        seq_path, label_path, subject_id = self.seqs_labels_path_pair[idx]
        # print(seq_path) # 调试时可以取消注释
        try:
            # 加载 EEG 和 EOG 通道
            # 注意：这里假设 .npy 文件保存的是 (num_epochs_in_seq, num_channels, sequence_length)
            # 并且 EEG 是第 0 通道，EOG 是第 1 通道
            data = np.load(seq_path) # 预期形状 (20, 2, 3000)
            seq_eeg = data[:, :1, :]
            seq_eog = data[:, 1:2, :]
            seq = np.concatenate((seq_eeg, seq_eog), axis=1) # 形状仍为 (20, 2, 3000)

            label = np.load(label_path) # 预期形状 (20,)

            # 数据类型转换（可选，但推荐）
            seq = seq.astype(np.float32)
            label = label.astype(np.int64) # CrossEntropyLoss 需要 Long 类型

        except Exception as e:
            print(f"[错误] 加载数据失败: seq_path='{seq_path}', label_path='{label_path}'. 错误: {e}")
            # 返回占位符或引发异常，避免程序崩溃在后续处理上
            # 这里我们返回 None，并在 collate 函数中处理
            return None, None, None

        return seq, label, subject_id

    def collate(self, batch):
        # 过滤掉加载失败的样本 (None)
        batch = [item for item in batch if item[0] is not None]
        if not batch: # 如果整个批次都加载失败
            return None, None, None # 或者根据需要返回空的 tensor

        # 从有效样本中提取数据
        x_seq = np.array([x[0] for x in batch])     # (batch_size, 20, 2, 3000)
        y_label = np.array([x[1] for x in batch])   # (batch_size, 20)
        z_label = np.array([x[2] for x in batch])   # (batch_size,) subject id

        # 转换为 Tensor
        x_tensor = to_tensor(x_seq)
        y_tensor = to_tensor(y_label).long() # 确保标签是 Long 类型
        z_tensor = to_tensor(z_label).long()

        return x_tensor, y_tensor, z_tensor


class LoadDataset(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params
        # 数据集名称到 ID 的映射（用于分配 subject_id）
        self.datasets_map = {
            'sleep-edfx': 0,
            'HMC': 1,
            'ISRUC': 2,
            'SHHS1': 3,
            'P2018': 4,
        }
        # 如果 datasets_map 中没有找到 params.datasets_dir 中的数据集名称，可以给一个默认的基础 ID
        self.unknown_dataset_base_id = max(self.datasets_map.values()) + 1

        # --- 新增：判断是 LODO 训练模式还是独立测试模式 ---
        if hasattr(params, 'test_dataset') and params.test_dataset:
            self.mode = 'test'
            self.test_dataset_name = params.test_dataset
            self.test_dir = f'{self.params.datasets_dir}/{self.test_dataset_name}'
            print(f"[信息] LoadDataset 初始化为 '测试' 模式，测试数据集: {self.test_dataset_name}")
            if not os.path.isdir(self.test_dir):
                 raise FileNotFoundError(f"指定的测试数据集目录不存在: {self.test_dir}")
            self.source_dirs = []
            self.targets_dirs = [] # LODO 的 target 在测试模式下不用
        else:
            self.mode = 'lodo'
            # 按原 LODO 逻辑区分源域和目标域
            target_domains_list = params.target_domains.split(',') if isinstance(params.target_domains, str) else [params.target_domains] # 处理单个或多个目标域
            self.targets_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets_map.keys() if item in target_domains_list]
            self.source_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets_map.keys() if item not in target_domains_list]
            print("[信息] LoadDataset 初始化为 'LODO' 模式")
            print(f"  目标域 (LODO中的测试集): {self.targets_dirs}")
            print(f"  源域 (训练/验证集): {self.source_dirs}")


    def get_data_loader(self):
        # --- 根据模式加载数据 ---
        if self.mode == 'test':
            # --- 测试模式：只加载指定的测试数据集 ---
            print(f"[信息] 加载测试数据从: {self.test_dir}")
            test_pairs, _ = self.load_path([self.test_dir], start_subject_id=0) # 测试模式 id 从 0 开始
            if not test_pairs:
                print(f"[警告] 未能在 {self.test_dir} 中找到任何有效的 seq/label 文件对。")
                test_set = CustomDataset([]) # 创建空数据集避免错误
            else:
                print(f"  找到 {len(test_pairs)} 个测试样本。")
                test_set = CustomDataset(test_pairs)

            # 只返回 test loader
            data_loader_dict = {
                'train': None, # 或者返回一个空的 DataLoader
                'val': None,   # 或者返回一个空的 DataLoader
                'test': DataLoader(
                    test_set,
                    batch_size=self.params.batch_size,
                    collate_fn=test_set.collate,
                    shuffle=False, # 测试时通常不需要打乱
                    num_workers=self.params.num_workers,
                    pin_memory=True,
                )
            }
            # 测试模式下，返回的 subject_id 通常无意义，可以返回 -1 或 None
            return data_loader_dict, -1

        elif self.mode == 'lodo':
            # --- LODO 模式：按原来的逻辑加载 ---
            print("[信息] 加载源域数据...")
            source_domains_pairs, next_subject_id = self.load_path(self.source_dirs, start_subject_id=0)
            print(f"  共加载 {len(source_domains_pairs)} 个源域样本，下一个 subject_id 为 {next_subject_id}")

            print("[信息] 加载目标域数据...")
            target_domains_pairs, final_subject_id = self.load_path(self.targets_dirs, start_subject_id=next_subject_id)
            print(f"  共加载 {len(target_domains_pairs)} 个目标域样本，最终 subject_id 为 {final_subject_id -1}")

            if not source_domains_pairs:
                print("[警告] 未加载到任何源域数据，训练和验证集将为空。")
                train_pairs, val_pairs = [], []
            else:
                train_pairs, val_pairs = self.split_dataset(source_domains_pairs)
            print(f"  源域数据分割: {len(train_pairs)} 训练样本, {len(val_pairs)} 验证样本")

            if not target_domains_pairs:
                print("[警告] 未加载到任何目标域数据，测试集将为空。")
                test_set = CustomDataset([])
            else:
                 test_set = CustomDataset(target_domains_pairs) # LODO 模式下目标域是测试集

            train_set = CustomDataset(train_pairs)
            val_set = CustomDataset(val_pairs)

            # 返回包含 train, val, test 的 loader 字典
            data_loader_dict = {
                'train': DataLoader(
                    train_set,
                    batch_size=self.params.batch_size,
                    collate_fn=train_set.collate,
                    shuffle=True, # 训练时打乱
                    num_workers=self.params.num_workers,
                    pin_memory=True,
                ),
                'val': DataLoader(
                    val_set,
                    batch_size=self.params.batch_size,
                    collate_fn=val_set.collate,
                    shuffle=False, # 验证时通常不打乱
                    num_workers=self.params.num_workers,
                    pin_memory=True,
                ),
                'test': DataLoader(
                    test_set,
                    batch_size=self.params.batch_size,
                    collate_fn=test_set.collate,
                    shuffle=False, # 测试时通常不打乱
                    num_workers=self.params.num_workers,
                    pin_memory=True,
                ),
            }
            # LODO 模式下返回最后一个分配的 subject_id
            return data_loader_dict, final_subject_id

        else:
            raise ValueError(f"未知的 LoadDataset 模式: {self.mode}")


    def load_path(self, domains_dirs, start_subject_id):
        """
        加载指定目录列表下的所有 seq 和 label 文件路径对。
        为每个数据集目录分配一个连续的 subject_id。
        返回: [(seq_path, label_path, subject_id), ...], next_subject_id
        """
        all_pairs = []
        current_subject_id = start_subject_id
        print(f"  开始加载路径，起始 subject_id: {current_subject_id}")

        for dataset_dir in domains_dirs:
            dataset_name = Path(dataset_dir).name
            # 尝试从 self.datasets_map 获取 ID，否则分配一个新的 ID
            subject_id = self.datasets_map.get(dataset_name, current_subject_id)
            # 如果是新分配的 ID，确保 current_subject_id 递增
            if subject_id == current_subject_id:
                print(f"    数据集 '{dataset_name}' 不在预定义映射中，分配新 ID: {subject_id}")
                current_subject_id += 1 # 准备下一个新 ID
            else:
                 print(f"    加载数据集 '{dataset_name}' (ID: {subject_id}) 从: {dataset_dir}")
                 # 如果使用了预定义 ID，确保 current_subject_id 不小于已分配的最大 ID + 1
                 current_subject_id = max(current_subject_id, max(self.datasets_map.values()) + 1)


            seq_base_path = os.path.join(dataset_dir, 'seq')
            label_base_path = os.path.join(dataset_dir, 'labels')

            if not os.path.isdir(seq_base_path) or not os.path.isdir(label_base_path):
                print(f"    [警告] 在 {dataset_dir} 中未找到 'seq' 或 'labels' 子目录，跳过此数据集。")
                continue

            # 遍历 seq 子目录下的文件夹 (例如 ISRUC-group1-1, ISRUC-group1-2, ...)
            try:
                sub_dirs = sorted([d for d in os.listdir(seq_base_path) if os.path.isdir(os.path.join(seq_base_path, d))])
            except FileNotFoundError:
                 print(f"    [警告] 无法访问目录 {seq_base_path}，跳过此数据集。")
                 continue

            if not sub_dirs:
                 print(f"    [警告] 在 {seq_base_path} 中未找到子目录，将直接在此目录下查找 .npy 文件。")
                 sub_dirs = ["."] # 如果没有子目录，就直接在 seq/ 目录下查找

            found_files_in_dataset = False
            for sub_dir in sub_dirs:
                current_seq_dir = os.path.join(seq_base_path, sub_dir)
                current_label_dir = os.path.join(label_base_path, sub_dir) # 假设 label 目录结构与 seq 一致

                if not os.path.isdir(current_label_dir):
                    print(f"    [警告] 未找到对应的标签目录 {current_label_dir}，跳过子目录 {sub_dir}。")
                    continue

                try:
                    seq_files = sorted([f for f in os.listdir(current_seq_dir) if f.endswith('.npy')])
                    label_files = sorted([f for f in os.listdir(current_label_dir) if f.endswith('.npy')])
                except FileNotFoundError:
                     print(f"    [警告] 无法访问目录 {current_seq_dir} 或 {current_label_dir}，跳过子目录 {sub_dir}。")
                     continue


                # 按文件名匹配 seq 和 label 文件 (假设它们的文件名除了路径外一一对应)
                label_files_dict = {f: os.path.join(current_label_dir, f) for f in label_files}

                for seq_file in seq_files:
                    if seq_file in label_files_dict:
                        seq_path = os.path.join(current_seq_dir, seq_file)
                        label_path = label_files_dict[seq_file]
                        all_pairs.append((seq_path, label_path, subject_id))
                        found_files_in_dataset = True
                    else:
                        print(f"    [警告] 未找到序列文件 '{seq_file}' 对应的标签文件，已跳过。")

            if not found_files_in_dataset and sub_dirs == ["."]:
                print(f"    [警告] 在目录 {seq_base_path} 和 {label_base_path} 中未找到任何 .npy 文件对。")


        print(f"  完成加载路径，共 {len(all_pairs)} 个文件对，下一个可用的 subject_id: {current_subject_id}")
        return all_pairs, current_subject_id

    def split_dataset(self, source_domain_pairs, val_ratio=0.2, seed=None):
        """
        将源域数据分割为训练集和验证集。
        默认按 80%/20% 分割。
        可以传入 seed 以保证可复现性。
        """
        if seed is not None:
            random.seed(seed) # 设置随机种子

        # 复制列表以避免修改原始列表
        shuffled_pairs = source_domain_pairs[:]
        random.shuffle(shuffled_pairs)

        split_num = int(len(shuffled_pairs) * (1 - val_ratio))
        train_pairs = shuffled_pairs[:split_num]
        val_pairs = shuffled_pairs[split_num:]

        if seed is not None:
             random.seed() # 恢复随机状态 (可选)

        return train_pairs, val_pairs