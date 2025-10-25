# -*- coding: utf-8 -*-
"""
utils/allutils.py
一个工具文件，集成：
- CSV 指标记录与读取
- 训练/验证曲线绘图（loss/acc/f1/lr等）
- t-SNE 执行与可视化（原始 vs 网络输出）
- metric 计算（acc/f1/confusion）
- 按比例抽样构建 DataLoader（data_ratio）
- 5折综合图与CSV汇总（allfold）

依赖：matplotlib, seaborn, numpy, pandas, scikit-learn, torch
"""
import os
import csv
import math
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =========================
# 基础IO工具
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(d: dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
def _now() -> str:  # <-- 添加这个函数
    """获取 'YYYY-MM-DD HH:MM:SS' 格式的当前时间字符串"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# =========================
# Metrics 计算
# =========================
def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))

def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    return float(f1_score(y_true, y_pred, average=average))

def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

# =========================
# 训练日志 CSV 记录器
# =========================
class MetricsLogger:
    """
    每折一个 logger。每个 epoch 追加写一行到 CSV。
    列：time, fold, epoch, lr, train_loss, train_acc?, train_f1?, val_acc, val_f1, 各类F1...
    （train_acc/train_f1 可选，你有的话就传；没有就传None）
    """
    def __init__(self, fold_dir: Path, fold_id: int):
        self.fold_dir = ensure_dir(fold_dir)
        self.fold_id = fold_id
        self.csv_path = self.fold_dir / "metrics.csv"
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time", "fold", "epoch", "lr",
                    "train_loss", "train_acc", "train_f1",
                    "val_acc", "val_f1",
                    "wake_f1", "n1_f1", "n2_f1", "n3_f1", "rem_f1"
                ])

    def log_epoch(self,
                  time_str: str,
                  epoch: int,
                  lr: float,
                  train_loss: float,
                  val_acc: float,
                  val_f1: float,
                  wake_f1: Optional[float] = None,
                  n1_f1: Optional[float] = None,
                  n2_f1: Optional[float] = None,
                  n3_f1: Optional[float] = None,
                  rem_f1: Optional[float] = None,
                  train_acc: Optional[float] = None,
                  train_f1: Optional[float] = None):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time_str, self.fold_id, epoch, lr,
                float(train_loss) if train_loss is not None else "",
                float(train_acc) if train_acc is not None else "",
                float(train_f1) if train_f1 is not None else "",
                float(val_acc) if val_acc is not None else "",
                float(val_f1) if val_f1 is not None else "",
                float(wake_f1) if wake_f1 is not None else "",
                float(n1_f1) if n1_f1 is not None else "",
                float(n2_f1) if n2_f1 is not None else "",
                float(n3_f1) if n3_f1 is not None else "",
                float(rem_f1) if rem_f1 is not None else "",
            ])

    def path(self) -> Path:
        return self.csv_path

# =========================
# DataLoader 按比例抽样
# =========================
def build_ratio_loader(orig_loader: DataLoader,
                       ratio: float = 1.0,
                       seed: int = 42) -> DataLoader:
    """复制一个新的 DataLoader，但数据集为原 dataset 的子集(比例为 ratio)。"""
    if ratio >= 1.0:
        return orig_loader

    dataset = orig_loader.dataset
    N = len(dataset)
    n_use = max(1, int(N * ratio))

    rng = random.Random(seed)
    indices = list(range(N))
    rng.shuffle(indices)
    pick = indices[:n_use]

    subset = Subset(dataset, pick)
    # 尽量复用原 loader 的参数
    new_loader = DataLoader(
        subset,
        batch_size=orig_loader.batch_size,
        shuffle=True,  # 子集内再shuffle
        num_workers=orig_loader.num_workers,
        drop_last=getattr(orig_loader, "drop_last", False),
        pin_memory=getattr(orig_loader, "pin_memory", False),
        collate_fn=getattr(orig_loader, "collate_fn", None)
    )
    return new_loader

# =========================
# 绘图：指标曲线
# =========================
def _setup_style():
    sns.set(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

def plot_curves_for_fold(csv_path: Path, out_dir: Path):
    """读取单折 CSV，绘制 loss/acc/f1/lr 曲线，输出到该折目录"""
    ensure_dir(out_dir)
    df = pd.read_csv(csv_path)

    _setup_style()

    # 1) Loss
    plt.figure()
    sns.lineplot(x="epoch", y="train_loss", data=df, marker="o")
    plt.title("Train Loss per Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.savefig(out_dir / "loss_curve.pdf")
    plt.close()

    # 2) Val Accuracy
    if "val_acc" in df.columns:
        plt.figure()
        sns.lineplot(x="epoch", y="val_acc", data=df, marker="o")
        plt.title("Validation Accuracy per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "val_acc_curve.png", dpi=180)
        plt.savefig(out_dir / "val_acc_curve.pdf")
        plt.close()

    # 3) Val F1
    if "val_f1" in df.columns:
        plt.figure()
        sns.lineplot(x="epoch", y="val_f1", data=df, marker="o")
        plt.title("Validation F1 per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("F1 (macro)")
        plt.tight_layout()
        plt.savefig(out_dir / "val_f1_curve.png", dpi=180)
        plt.savefig(out_dir / "val_f1_curve.pdf")
        plt.close()

    # 4) LR
    if "lr" in df.columns:
        plt.figure()
        sns.lineplot(x="epoch", y="lr", data=df, marker="o")
        plt.title("Learning Rate per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("LR")
        plt.tight_layout()
        plt.savefig(out_dir / "lr_curve.png", dpi=180)
        plt.savefig(out_dir / "lr_curve.pdf")
        plt.close()

def plot_curves_allfold(fold_csv_paths: List[Path], out_dir: Path):
    """读取多折 CSV，计算均值并绘制综合曲线到 allfold/"""
    ensure_dir(out_dir)
    _setup_style()

    # 合并
    dfs = []
    for p in fold_csv_paths:
        if p.exists():
            df = pd.read_csv(p)
            df["__fold__"] = p.parent.name
            dfs.append(df)
    if not dfs:
        return
    ALL = pd.concat(dfs, ignore_index=True)

    # 对相同epoch聚合均值
    metrics = ["train_loss", "val_acc", "val_f1", "lr"]
    for m in metrics:
        if m not in ALL.columns:
            continue
        mean_df = ALL.groupby("epoch")[m].mean().reset_index()

        plt.figure()
        # 各折
        for name, g in ALL.dropna(subset=[m]).groupby("__fold__"):
            sns.lineplot(x="epoch", y=m, data=g, alpha=0.35, label=name)
        # 平均
        sns.lineplot(x="epoch", y=m, data=mean_df, linewidth=3, marker="o", label="mean")
        ttl = f"All-Folds {m} per Epoch"
        plt.title(ttl)
        plt.xlabel("Epoch"); plt.ylabel(m)
        plt.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / f"allfold_{m}_curve.png", dpi=180)
        plt.savefig(out_dir / f"allfold_{m}_curve.pdf")
        plt.close()

# =========================
# t-SNE 可视化
# =========================
# utils/allutils.py 中，完整替换这个函数
@torch.no_grad()
def extract_features_for_tsne(model: torch.nn.Module,
                              loader: DataLoader,
                              device: torch.device,
                              take: str = "mu_tilde") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从测试集提取：
      RAW: [N, L] 展平后的原始输入
      REP: [N, D] 选取 'mu_tilde'/'mu'/'logits' 作为表征
      Y  : [N]    标签
    """
    model.eval()
    raw_list, rep_list, lab_list = [], [], []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, y, z = batch
            elif len(batch) == 2:
                x, y = batch; z = None
            else:
                x = batch[0]; y = batch[1]; z = None
        else:
            # 非标准 batch（很少见）
            x, y, z = batch["x"], batch.get("y", None), batch.get("z", None)

        x = x.to(device, non_blocking=True)
        if y is not None:
            y = y.to(device, non_blocking=True).long()
        if z is not None:
            z = z.to(device, non_blocking=True).long()

        # 原始展平
        raw = x.detach().float().cpu().numpy().reshape(x.shape[0], -1)
        raw_list.append(raw)

        # 前向
        out = model(x, labels=y, domain_ids=z)
        if isinstance(out, (list, tuple)) and len(out) >= 4:
            logits, recon, mu, mu_tilde = out[:4]
        else:
            raise RuntimeError("模型forward返回不含预期的4个主输出 (logits, recon, mu, mu_tilde)")

        if take == "mu_tilde":
            rep = mu_tilde
        elif take == "mu":
            rep = mu
        elif take == "logits":
            rep = logits.mean(dim=1) if logits.dim() == 3 else logits
        else:
            raise ValueError(f"unknown take={take}")

        rep = rep.detach().float().cpu().numpy().reshape(rep.shape[0], -1)
        rep_list.append(rep)
        lab_list.append(y.detach().cpu().numpy())

    RAW = np.concatenate(raw_list, axis=0)
    REP = np.concatenate(rep_list, axis=0)
    Y = np.concatenate(lab_list, axis=0)
    return RAW, REP, Y


def _tsne_2d(X: np.ndarray, perplexity: int = 80, pca_dim: int = 50, seed: int = 42) -> np.ndarray:
    """PCA 预降维 + t-SNE 到2D"""
    Xp = X
    if X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed, learning_rate="auto", n_iter=1200)
    Z = tsne.fit_transform(Xp)
    return Z
def _to_1d(a: np.ndarray) -> np.ndarray:
    """确保是一维 (N,)，若是 (N,1) / (N,T) / one-hot 则化到 (N,)。"""
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    # 如果是 one-hot 或 (N,T) 多列标签，取最后一维 argmax
    if a.ndim >= 2:
        return a.argmax(axis=-1).reshape(-1)
    return a.reshape(-1)

def _make_df(Z2: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """把二维坐标和标签打包成 DataFrame，列都保证 1D。"""
    Z2 = np.asarray(Z2)
    assert Z2.ndim == 2 and Z2.shape[1] == 2, f"Z2 shape must be (N,2), got {Z2.shape}"
    x1 = np.asarray(Z2[:, 0]).reshape(-1)
    x2 = np.asarray(Z2[:, 1]).reshape(-1)
    y1d = _to_1d(labels)
    # 强制长度一致
    N = min(len(x1), len(x2), len(y1d))
    df = pd.DataFrame({
        "x1": x1[:N],
        "x2": x2[:N],
        "label": y1d[:N].astype(int)
    })
    return df

def tsne_compare_plot(raw_X: np.ndarray,
                      rep_X: np.ndarray,
                      y: np.ndarray,
                      out_dir: Path,
                      title_prefix: str = "Test t-SNE",
                      filename_prefix: str = "tsne",
                      palette: Optional[List[str]] = None,
                      max_points: Optional[int] = None,
                      seed: int = 42):
    """
    绘制 原始输入(raw) vs 表征(rep) t-SNE 对比图。
    - 自动将 labels 压成一维；若为 one-hot 或 (N,T) 会取 argmax。
    - 为避免 seaborn 报错，先构造 DataFrame 再绘图。
    - max_points: 若样本很多，可设置下采样数量（如 10000）。
    """
    ensure_dir(out_dir)
    _setup_style()
    rng = np.random.default_rng(seed)

    # 颜色
    if palette is None:
        palette = sns.color_palette("tab10", n_colors=10)

    # 计算 t-SNE 嵌入
    Z_raw = _tsne_2d(raw_X)
    Z_rep = _tsne_2d(rep_X)

    # 组装 DF（确保全是 1D）
    df_raw = _make_df(Z_raw, y)
    df_rep = _make_df(Z_rep, y)

    # 可选下采样
    def _subsample(df: pd.DataFrame) -> pd.DataFrame:
        if max_points is None or len(df) <= max_points:
            return df
        idx = rng.choice(len(df), size=max_points, replace=False)
        return df.iloc[idx]
    df_raw = _subsample(df_raw)
    df_rep = _subsample(df_rep)

    # 确保 palette 足够长
    n_classes = int(df_raw["label"].nunique())
    if len(palette) < n_classes:
        palette = sns.color_palette("tab20", n_colors=n_classes)

    # 原始
    plt.figure()
    sns.scatterplot(data=df_raw, x="x1", y="x2", hue="label",
                    palette=palette[:n_classes], s=14, linewidth=0, alpha=0.75)
    plt.title(f"{title_prefix} — Raw")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"{filename_prefix}_raw.png", dpi=180)
    plt.savefig(out_dir / f"{filename_prefix}_raw.pdf")
    plt.close()

    # 表征
    plt.figure()
    sns.scatterplot(data=df_rep, x="x1", y="x2", hue="label",
                    palette=palette[:n_classes], s=14, linewidth=0, alpha=0.75)
    plt.title(f"{title_prefix} — Representation")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"{filename_prefix}_rep.png", dpi=180)
    plt.savefig(out_dir / f"{filename_prefix}_rep.pdf")
    plt.close()


# =========================
# 5折汇总（allfold）
# =========================
def write_aggregate_row(allfold_csv: Path,
                        row: Dict):
    """向 allfold/aggregate_results.csv 追加一行；无文件则写表头"""
    ensure_dir(allfold_csv.parent)
    write_header = not allfold_csv.exists()
    with open(allfold_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "time", "run_id", "fold",
            "best_val_acc", "best_val_f1",
            "test_acc", "test_f1",
            "wake_f1", "n1_f1", "n2_f1", "n3_f1", "rem_f1",
            "model_path"
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def collect_and_plot_allfold(fold_dirs: List[Path], out_dir: Path):
    """收集各 fold/metrics.csv，绘制 allfold 综合图"""
    csvs = [d / "metrics.csv" for d in fold_dirs]
    plot_curves_allfold(csvs, out_dir=out_dir)
