# %%
from mne.io import read_raw_edf
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ------------ 路径配置（按需修改） ------------
dir_path = r'/public_data/hmc-sleep-staging/DESTINATION/recordings/'

seq_dir = r'/data/lijinyang/SleepSLeep/SleepDG-main/dataa/HMC/seq'
label_dir = r'/data/lijinyang/SleepSLeep/SleepDG-main/dataa/HMC/labels'

os.makedirs(seq_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# ------------ 采样与通道配置（按需修改） ------------
target_sfreq = 100.0
# 这里与你原脚本一致：只取两个通道
signal_name = ['EEG F4-M1', 'EOG E1-M2']

# ------------ 标签映射（W,N1,N2,N3,R） ------------
label2id = {
    'Sleep stage W': 0,
    'Sleep stage N1': 1,
    'Sleep stage N2': 2,
    'Sleep stage N3': 3,
    'Sleep stage R': 4,
}
# 只保留上述睡眠事件，直接在 events_from_annotations 中指定
sleep_event_id = {
    'Sleep stage W': 1,
    'Sleep stage N1': 2,
    'Sleep stage N2': 3,
    'Sleep stage N3': 4,
    'Sleep stage R': 5,
}

# -----------------------------------------------------
# 1) 构造 PSG/标注文件配对
# -----------------------------------------------------
f_names = os.listdir(dir_path)
psg_f_names, label_f_names = [], []
for f_name in f_names:
    if 'sleepscoring.edf' in f_name:
        label_f_names.append(f_name)
    elif f_name.endswith('.edf'):
        psg_f_names.append(f_name)
psg_f_names.sort()
label_f_names.sort()

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:5] == label_f_name[:5]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))

print(f'共配对到 {len(psg_label_f_pairs)} 个文件：')
print(psg_label_f_pairs)

# -----------------------------------------------------
# 2) 逐被试读取 → 切 epoch(30s) → 生成样本与标签
#    注：去除了 get_annotations_per_epoch 的错误用法
# -----------------------------------------------------
num_seqs = 0
num_labels = 0

for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):

    # 2.1 读取原始 PSG
    raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True, verbose='ERROR')
    # 只保留所需通道，若通道名不在文件中会报错，可改为 pick_channels(..., error='ignore') 自行容错
    raw.pick_channels(signal_name)
    # 重采样到 target_sfreq（30s -> 30*100=3000 点）
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(sfreq=target_sfreq)

    # 2.2 读取并设置注释（hypnogram）
    annotation = mne.read_annotations(os.path.join(dir_path, label_f_name))
    raw.set_annotations(annotation, emit_warning=False)

    # 2.3 从注释生成事件（只保留睡眠阶段；自动忽略 Lights on/off 等）
    events, event_id = mne.events_from_annotations(
        raw,
        event_id=sleep_event_id,       # 只解析睡眠阶段
        chunk_duration=30.0            # 30s 一个 epoch
    )
    # 若没有可用事件，跳过
    if len(events) == 0:
        print(f'[WARN] {psg_f_name} 无有效睡眠事件，跳过')
        continue

    # 2.4 构造 Epochs（每个事件切一个 30s epoch）
    tmax = 30.0 - 1.0 / raw.info['sfreq']  # tmax 为包含端点
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,                    # 直接加载到内存，便于后续 numpy 操作
        verbose='ERROR'
    )

    # 2.5 标签获取：用 epochs.events[:,2] 的编码，反查成标签名，再映射到整数ID
    inv_event_id = {v: k for k, v in epochs.event_id.items()}  # 反向字典：编码->名字
    codes = epochs.events[:, 2]                                 # 每个 epoch 的事件编码
    label_names = [inv_event_id[int(c)] for c in codes]         # 字符串标签
    labels_np = np.array([label2id[nm] for nm in label_names], dtype=np.int64)

    # 2.6 取出数据：形状 (N, C, T)
    X = epochs.get_data()  # ndarray，(n_epochs, n_channels, n_times)
    # 容错：若标签数与 epoch 数不一致（理论不会发生）
    assert X.shape[0] == labels_np.shape[0], "数据与标签数量不一致"

    # -------------------------------------------------
    # 3) 标准化（与你原逻辑一致：先展平到(-1, C)，对每个通道做全局标准化）
    # -------------------------------------------------
    N, C, T = X.shape  # 期望 T=3000
    X_flat = X.transpose(0, 2, 1).reshape(-1, C)   # (N,T,C) -> (N*T, C)
    std = StandardScaler()
    X_flat = std.fit_transform(X_flat)             # 按通道标准化
    X_std = X_flat.reshape(N, T, C).transpose(0, 2, 1)  # -> (N, C, T)

    # -------------------------------------------------
    # 4) 组成长度为 20 的序列（不足部分截掉）
    # -------------------------------------------------
    n_keep = (N // 20) * 20
    if n_keep == 0:
        print(f'[INFO] {psg_f_name} 有效 epoch={N} < 20，跳过保存序列')
        continue
    X_std = X_std[:n_keep]
    labels_np = labels_np[:n_keep]

    X_seq = X_std.reshape(-1, 20, C, T)           # (M, 20, C, T)
    y_seq = labels_np.reshape(-1, 20)             # (M, 20)
    print(f'{psg_f_name}: X_seq={X_seq.shape}, y_seq={y_seq.shape}')

    # -------------------------------------------------
    # 5) 保存到目标目录（文件名前缀用前5字符以与你现逻辑保持一致）
    # -------------------------------------------------
    pid = psg_f_name[:5]
    # seq
    pid_seq_dir = os.path.join(seq_dir, pid)
    os.makedirs(pid_seq_dir, exist_ok=True)
    for seq in X_seq:
        out_path = os.path.join(pid_seq_dir, f'{pid}-{num_seqs}.npy')
        with open(out_path, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    # labels
    pid_lab_dir = os.path.join(label_dir, pid)
    os.makedirs(pid_lab_dir, exist_ok=True)
    for lab in y_seq:
        out_path = os.path.join(pid_lab_dir, f'{pid}-{num_labels}.npy')
        with open(out_path, 'wb') as f:
            np.save(f, lab)
        num_labels += 1

print(f'完成：保存序列 {num_seqs} 个，标签 {num_labels} 个')
# %%
