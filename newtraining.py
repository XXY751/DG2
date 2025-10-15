# trainer.py

import os
import copy
import logging
from datetime import datetime
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import Model
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from losses.double_alignment import CORAL
from losses.ae_loss import AELoss

# === 工具 ===
from utils.allutils import (
    ensure_dir, MetricsLogger, build_ratio_loader,
    plot_curves_for_fold, extract_features_for_tsne, tsne_compare_plot,
    write_aggregate_row
)

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Trainer(object):
    def __init__(self, params):
        self.params = params

        # --- 目录：每折单独一个 fold{K} 子目录；allfold 为根聚合 ---
        self.model_dir = Path(getattr(self.params, "model_dir", "./outputs"))
        self.fold_id = int(getattr(self.params, "fold", 0))
        self.fold_dir = ensure_dir(self.model_dir / f"fold{self.fold_id}")
        self.allfold_dir = ensure_dir(self.model_dir / "allfold")
        self.allfold_csv = self.allfold_dir / "aggregate_results.csv"

        # --- Data ---
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()
        # data_ratio：仅对训练集抽样
        self.data_ratio = float(getattr(self.params, "data_ratio", 1.0))
        if 0 < self.data_ratio < 1.0:
            self.data_loader['train'] = build_ratio_loader(self.data_loader['train'], self.data_ratio, seed=42)

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        # --- Model ---
        model = Model(params)
        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.cuda()

        # --- Loss/Lambda ---
        self.lambda_ae = getattr(self.params, "lambda_ae", 1.0)
        self.lambda_coral = getattr(self.params, "lambda_coral", 0.0)
        self.ce_loss = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        self.coral_loss = CORAL().cuda()
        self.ae_loss = AELoss().cuda()
        self.lmb_caa = getattr(self.params, "lambda_caa", 1.0)
        self.lmb_stat = getattr(self.params, "lambda_stat", 0.2)
        self.lmb_Areg = getattr(self.params, "lambda_Areg", 0.1)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.params.lr,
                                          weight_decay=self.params.lr / 10)

        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length
        )

        print(self.model)
        self._frozen_uni_built = False

        # --- 日志与CSV ---
        self.logger = self._setup_logger()
        self.metrics_logger = MetricsLogger(self.fold_dir, self.fold_id)
        self.best_model_states = None

        # 文本结果（追加）
        self.result_txt = self.fold_dir / "results.txt"
        with open(self.result_txt, "a", encoding="utf-8") as f:
            f.write(f"[{_now()}] Start fold{self.fold_id}, data_ratio={self.data_ratio}\n\n")

    def _setup_logger(self):
        logger = logging.getLogger(f"TrainerFold{self.fold_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | fold=%(fold)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        fh = logging.FileHandler(self.fold_dir / "run.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | fold=%(fold)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(ch); logger.addHandler(fh)

        # 注入 fold 变量
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.fold = self.fold_id
            return record
        logging.setLogRecordFactory(record_factory)
        return logger

    def _log_txt(self, msg: str):
        print(msg)
        with open(self.result_txt, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def train(self):
        acc_best = 0.0
        f1_best = 0.0
        best_f1_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y, z in tqdm(self.data_loader['train'], mininterval=10,
                                desc=f"Fold {self.fold_id} | Epoch {epoch+1}"):
                self.optimizer.zero_grad()
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True).long()
                z = z.cuda(non_blocking=True).long()

                logits, recon, mu, mu_tilde, reg_A = self.model(x, labels=y, domain_ids=z)

                # 任务损失
                loss_task = self.ce_loss(logits.permute(0, 2, 1), y)

                # 锚点/统计
                model_to_update = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                with torch.no_grad():
                    model_to_update.anchors.update(mu_tilde.detach(), y, z)
                loss_caa = model_to_update.anchors.caa_loss()
                loss_stat = model_to_update.anchors.stats_align_loss(mu_tilde, z)

                # 组合
                loss = (loss_task +
                        self.lmb_caa * loss_caa +
                        self.lmb_stat * loss_stat +
                        self.lmb_Areg * reg_A.mean())

                if self.lambda_ae != 0.0:
                    loss = loss + self.ae_loss(x, recon) * self.lambda_ae
                if self.lambda_coral != 0.0:
                    loss = loss + self.coral_loss(mu, z) * self.lambda_coral

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.scheduler.step()
                losses.append(float(loss.detach().cpu().item()))

            optim_state = self.optimizer.state_dict()
            lr_now = optim_state['param_groups'][0]['lr']
            time_min = (timer() - start_time) / 60.0

            # 验证
            with torch.no_grad():
                model_to_eval = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                model_to_eval.freeze_unified_projection(strategy="avg")
                acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = self.val_eval.get_accuracy(self.model)

            # 日志 + CSV
            msg = (f"Epoch {epoch+1:03d} | train_loss={np.mean(losses):.5f} | "
                   f"val_acc={acc:.5f} | val_f1={f1:.5f} | "
                   f"wake={wake_f1:.5f} n1={n1_f1:.5f} n2={n2_f1:.5f} n3={n3_f1:.5f} rem={rem_f1:.5f} | "
                   f"lr={lr_now:.6f} | {time_min:.2f} min")
            self._log_txt(msg)
            self.metrics_logger.log_epoch(
                time_str=_now(),
                epoch=epoch+1,
                lr=lr_now,
                train_loss=float(np.mean(losses)),
                val_acc=float(acc),
                val_f1=float(f1),
                wake_f1=float(wake_f1),
                n1_f1=float(n1_f1),
                n2_f1=float(n2_f1),
                n3_f1=float(n3_f1),
                rem_f1=float(rem_f1),
            )

            # 保存最优
            if acc > acc_best:
                best_f1_epoch = epoch + 1
                acc_best = acc
                f1_best = f1
                self.best_model_states = copy.deepcopy(self.model.state_dict())
                self._log_txt(f"[BEST@{best_f1_epoch:03d}] val_acc={acc_best:.5f} | val_f1={f1_best:.5f}")

        self._log_txt(f"Best@Epoch {best_f1_epoch:03d} -> val_acc={acc_best:.5f}, val_f1={f1_best:.5f}")

        # 单折曲线图
        plot_curves_for_fold(self.metrics_logger.path(), out_dir=self.fold_dir)

        test_acc, test_f1 = self.test(best_val_acc=acc_best, best_val_f1=f1_best)
        return test_acc, test_f1

    def test(self, best_val_acc: float, best_val_f1: float):
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            model_to_test = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model_to_test.freeze_unified_projection(strategy="avg")
            self._log_txt("*************************** Test ***************************")
            test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
                test_n3_f1, test_rem_f1 = self.test_eval.get_accuracy(self.model)

            self._log_txt(f"Test: acc={test_acc:.5f}, f1={test_f1:.5f}")
            self._log_txt(str(test_cm))
            self._log_txt(
                ("Class F1 -> "
                 f"wake={test_wake_f1:.5f}, n1={test_n1_f1:.5f}, n2={test_n2_f1:.5f}, "
                 f"n3={test_n3_f1:.5f}, rem={test_rem_f1:.5f}")
            )

            # 保存模型（带指标）
            model_path = self.fold_dir / (
                f"fold{self.fold_id}_tacc_{test_acc:.5f}_tf1_{test_f1:.5f}.pth"
            )
            torch.save(self.best_model_states, model_path)
            self._log_txt("Model saved -> " + str(model_path))

            # ===== t-SNE: 原始 vs 表征（mu_tilde） =====
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            RAW, REP, Y = extract_features_for_tsne(model_to_test, self.data_loader['test'], device, take="mu_tilde")
            tsne_compare_plot(
                raw_X=RAW, rep_X=REP, y=Y,
                out_dir=self.fold_dir,
                title_prefix=f"Fold{self.fold_id} Test t-SNE",
                filename_prefix="tsne"
            )

            # allfold 聚合CSV 追加
            write_aggregate_row(
                self.allfold_dir / "aggregate_results.csv",
                row=dict(
                    time=_now(),
                    run_id=getattr(self.params, "run_name", f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                    fold=self.fold_id,
                    best_val_acc=float(best_val_acc),
                    best_val_f1=float(best_val_f1),
                    test_acc=float(test_acc),
                    test_f1=float(test_f1),
                    wake_f1=float(test_wake_f1),
                    n1_f1=float(test_n1_f1),
                    n2_f1=float(test_n2_f1),
                    n3_f1=float(test_n3_f1),
                    rem_f1=float(test_rem_f1),
                    model_path=str(model_path)
                )
            )

        return test_acc, test_f1
