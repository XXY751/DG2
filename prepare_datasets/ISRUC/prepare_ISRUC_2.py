# %%
import os
import numpy as np
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from mne.io import concatenate_raws  # 保留
from edf import read_raw_edf
import mne

# =============== 日志配置 =============== #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ISRUC-Preprocess")

# =============== 路径配置 =============== #
dir_path = r'/public_data/1_ISRUC/Subgroup2/'
seq_dir = r'/data/lijinyang/SleepSLeep/SleepDG-main/dataa/ISRUC/seq'
label_dir = r'/data/lijinyang/SleepSLeep/SleepDG-main/dataa/ISRUC/labels'
os.makedirs(seq_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# =============== 文件列表收集 =============== #
psg_f_names = []
label_f_names = []
for i in range(1, 8):
    numstr = str(i)
    psg_f_names.append(f'{dir_path}/{numstr}/1/1.rec')
    label_f_names.append(f'{dir_path}/{numstr}/1/1_1.txt')
    psg_f_names.append(f'{dir_path}/{numstr}/2/2.rec')
    label_f_names.append(f'{dir_path}/{numstr}/2/2_1.txt')

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:-4] == label_f_name[:-6]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))

for item in psg_label_f_pairs:
    logger.info(f"匹配到文件对: PSG={item[0]} | Label={item[1]}")

label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 4}
logger.info(f"标签映射: {label2id}")

# =============== 处理参数 =============== #
FS_TARGET = 100
LOWCUT, HIGHCUT = 0.3, 35.0
EPOCH_SEC = 30
SEQ_EPOCHS = 20
N_CH = 2

# =============== 计数器 =============== #
num_seqs_global = 0
num_labels_global = 0

# %%
m = 0
for psg_f_name, label_f_name in tqdm(psg_label_f_pairs, desc="文件对处理中"):
    n = m // 2 + 1
    m += 1

    group_seq_dir = os.path.join(seq_dir, f'ISRUC-group2-{str(n)}')
    group_label_dir = os.path.join(label_dir, f'ISRUC-group2-{str(n)}')
    os.makedirs(group_seq_dir, exist_ok=True)
    os.makedirs(group_label_dir, exist_ok=True)

    try:
        # ---------- 读取 ----------
        src_psg_path = os.path.join(dir_path, psg_f_name)
        src_label_path = os.path.join(dir_path, label_f_name)

        logger.info(f"[开始] 读取PSG: {src_psg_path}")
        raw = read_raw_edf(src_psg_path, preload=True)
        logger.info(f"原始信息: n_chan={raw.info['nchan']}, sfreq={raw.info['sfreq']}")

        # 重采样 + 滤波
        raw.resample(sfreq=FS_TARGET)
        logger.info(f"已重采样至 {FS_TARGET} Hz")
        raw.filter(LOWCUT, HIGHCUT, fir_design='firwin')
        logger.info(f"已带通滤波: {LOWCUT}-{HIGHCUT} Hz")

        # 转为 numpy，去掉时间列
        psg_array = raw.to_data_frame().values
        logger.info(f"DataFrame->ndarray 形状(含时间列)：{psg_array.shape}")
        psg_array = psg_array[:, 1:]
        logger.info(f"去除时间列后形状：{psg_array.shape}")

        # 通道选择（保持与你原代码一致）
        eeg_array = psg_array[:, 5:6]  # 第6列
        eog_array = psg_array[:, 0:1]  # 第1列
        psg_array = np.concatenate((eeg_array, eog_array), axis=1)  # (T, 2)
        logger.info(f"选择EEG+EOG后形状：(T, C)={psg_array.shape}")

        # 标准化
        std = StandardScaler()
        psg_array = std.fit_transform(psg_array)

        # 对齐到整epoch
        samples_per_epoch = EPOCH_SEC * FS_TARGET  # 3000
        rem_samples = psg_array.shape[0] % samples_per_epoch
        if rem_samples > 0:
            psg_array = psg_array[:-rem_samples, :]
            logger.info(f"对齐到整epoch: 去掉最后 {rem_samples} 个采样点")
        logger.info(f"对齐后形状：{psg_array.shape}")

        # (N_epoch, 3000, 2)
        psg_array = psg_array.reshape(-1, samples_per_epoch, N_CH)
        logger.info(f"按epoch重塑形状：(N_epoch, 3000, 2)={psg_array.shape}")

        # 对齐到整序列（20个epoch）
        rem_epoch = psg_array.shape[0] % SEQ_EPOCHS
        if rem_epoch > 0:
            psg_array = psg_array[:-rem_epoch, :, :]
            logger.info(f"对齐到整序列: 去掉最后 {rem_epoch} 个epoch")
        logger.info(f"整序列对齐后形状：(N_epoch, 3000, 2)={psg_array.shape}")

        # (N_seq, 20, 3000, 2) -> (N_seq, 20, 2, 3000)
        psg_array = psg_array.reshape(-1, SEQ_EPOCHS, samples_per_epoch, N_CH)
        epochs_seq = psg_array.transpose(0, 1, 3, 2)  # ✅ 正确的4维转置
        logger.info(f"最终序列形状 (N_seq, {SEQ_EPOCHS}, {N_CH}, {samples_per_epoch})：{epochs_seq.shape}")

        # ---------- 读取标签并对齐 ----------
        labels_list = []
        logger.info(f"[开始] 读取标签: {src_label_path}")
        with open(src_label_path, 'r') as f:
            for line in f:
                s = line.strip()
                if s:
                    if s not in label2id:
                        logger.warning(f"发现未映射标签 '{s}'，跳过该行")
                        continue
                    labels_list.append(label2id[s])
        labels_array = np.array(labels_list, dtype=int)

        # 先对齐到 PSG 的 epoch 数
        total_epochs_psg = epochs_seq.shape[0] * SEQ_EPOCHS  # N_seq * 20
        if labels_array.shape[0] < total_epochs_psg:
            logger.error(f"标签不足：labels={labels_array.shape[0]} < 需要的={total_epochs_psg}，跳过该文件对")
            continue
        if labels_array.shape[0] > total_epochs_psg:
            logger.info(f"标签多于数据epoch，将截断到 {total_epochs_psg}")
            labels_array = labels_array[:total_epochs_psg]

        # 切成 (N_seq, 20)
        labels_seq = labels_array.reshape(-1, SEQ_EPOCHS)
        logger.info(f"标签序列形状：{labels_seq.shape}")

        # 一致性检查
        if epochs_seq.shape[0] != labels_seq.shape[0]:
            logger.error(f"样本数不一致：数据序列={epochs_seq.shape[0]}，标签序列={labels_seq.shape[0]}。跳过该文件对。")
            continue

        # ---------- 逐条保存 ----------
        n_seq = epochs_seq.shape[0]
        logger.info(f"[保存] 将保存 {n_seq} 条序列（每条含 {SEQ_EPOCHS} 个epoch）")

        for idx in range(n_seq):
            seq = epochs_seq[idx]   # (20, 2, 3000)
            lab = labels_seq[idx]   # (20,)

            seq_name = os.path.join(group_seq_dir, f'ISRUC-group2-{str(n)}-{str(num_seqs_global)}.npy')
            label_name = os.path.join(group_label_dir, f'ISRUC-group2-{str(n)}-{str(num_labels_global)}.npy')

            try:
                with open(seq_name, 'wb') as f_seq:
                    np.save(f_seq, seq)
                with open(label_name, 'wb') as f_lab:
                    np.save(f_lab, lab)
                logger.info(f"已保存 | 序列: {seq_name} | 标签: {label_name} | 形状: seq{seq.shape}, label{lab.shape}")
                num_seqs_global += 1
                num_labels_global += 1
            except Exception as e_save:
                logger.error(f"保存失败: {e_save} | 目标文件: {seq_name}, {label_name}")

        logger.info(f"[完成] 当前文件对处理结束：PSG={psg_f_name}, Label={label_f_name}")

    except Exception as e:
        logger.error(f"[异常] 处理失败：PSG={psg_f_name}, Label={label_f_name} | 错误：{e}")
        continue

logger.info(f"全部结束：总序列数={num_seqs_global}，总标签序列数={num_labels_global}")
