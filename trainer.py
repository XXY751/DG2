# trainer.py

import os
import copy
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

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


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()

        # ---- meta: run/fold/dirs ----
        self.fold_id = int(getattr(self.params, "fold", 0))
        # run_name 可选；没有则自动生成
        base_run = getattr(self.params, "run_name", None)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{base_run}__fold{self.fold_id}__{ts}" if base_run else f"run__fold{self.fold_id}__{ts}"

        # model_dir 作为本次运行的输出目录（每次运行都独立子目录）
        self.model_dir = Path(getattr(self.params, "model_dir", "./outputs"))
        self.run_dir = Path(_ensure_dir(self.model_dir / self.run_id))
        # 共享聚合文件放在 model_dir 根目录，所有折追加
        self.aggregate_csv = self.model_dir / "aggregate_results.csv"

        # ---- evaluators ----
        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.best_model_states = None

        # ---- model & optim ----
        model = Model(params)
        self._logger_setup()
        self.logger.info("Compiling/initializing model（未开启 torch.compile）...")
        self.model = model  # 如需 torch.compile，可在此替换

        if torch.cuda.device_count() > 1:
            self.logger.info(f"Detected {torch.cuda.device_count()} GPUs -> using DataParallel")
            self.model = nn.DataParallel(self.model)

        self.model.cuda()
        self.lambda_ae = getattr(self.params, "lambda_ae", 1.0)
        self.lambda_coral = getattr(self.params, "lambda_coral", 0.0)

        self.ce_loss = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        self.coral_loss = CORAL().cuda()
        self.ae_loss = AELoss().cuda()
        self.lmb_caa = getattr(self.params, "lambda_caa", 1.0)
        self.lmb_stat = getattr(self.params, "lambda_stat", 0.2)
        self.lmb_Areg = getattr(self.params, "lambda_Areg", 0.1)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.lr / 10
        )

        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length
        )

        self.logger.info(f"\n{self.model}\n")
        self._frozen_uni_built = False

        # ---- 日志与结果文件（全部改为“追加”写）----
        self.result_txt = self.run_dir / "results.txt"           # 本次运行的详细文本日志
        self.run_log = self.run_dir / "run.log"                  # logging.FileHandler 已写入
        self.epoch_csv = self.run_dir / f"train_log_fold{self.fold_id}.csv"  # 逐 epoch 指标
        self.test_cm_file = self.run_dir / f"test_confusion_fold{self.fold_id}.txt"
        self.meta_json = self.run_dir / "meta.json"

        # 写入 meta 信息
        meta = {
            "run_id": self.run_id,
            "fold": self.fold_id,
            "params": {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v))
                       for k, v in vars(self.params).items()}
        }
        with open(self.meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 逐 epoch CSV 文件：如不存在则写表头
        if not self.epoch_csv.exists():
            with open(self.epoch_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time", "fold", "epoch", "lr",
                    "train_loss",
                    "val_acc", "val_f1",
                    "wake_f1", "n1_f1", "n2_f1", "n3_f1", "rem_f1"
                ])

        # 共享聚合 CSV：如不存在则写表头
        if not self.aggregate_csv.exists():
            with open(self.aggregate_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time", "run_id", "fold",
                    "best_val_acc", "best_val_f1",
                    "test_acc", "test_f1",
                    "wake_f1", "n1_f1", "n2_f1", "n3_f1", "rem_f1",
                    "model_path"
                ])

        # 在 results.txt 留下 run 信息（追加）
        with open(self.result_txt, "a", encoding="utf-8") as f:
            f.write(f"[{_now()}] New run: {self.run_id}\n")
            f.write("Training and Evaluation Results\n\n")

    # ---------- 日志初始化 ----------
    def _logger_setup(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        # 避免重复 handler
        self.logger.handlers.clear()

        # 控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)5s | fold=%(fold)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        # 文件
        _ensure_dir(self.model_dir)
        _ensure_dir(self.params.model_dir) if hasattr(self.params, "model_dir") else None
        # 还没有 run_dir 时，先临时输出到 model_dir 下的一个通用日志，再在 __init__ 完成后切到 run_dir
        # 为简单起见，这里直接等 __init__ 结束再用 run_dir 的 FileHandler。先用空 handler 占位。
        fh = logging.NullHandler()

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        # 动态给日志上下文注入 fold 信息
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            # 默认 0，等 __init__ 设置 self.fold_id 后，会生效
            record.fold = getattr(self, "fold_id", 0)
            return record

        logging.setLogRecordFactory(record_factory)

    def _attach_file_handler(self):
        # 把文件 handler 指向 run_dir/run.log（追加）
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)
        fh = logging.FileHandler(self.run_dir / "run.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)5s | fold=%(fold)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

    def _log_txt(self, msg: str):
        """同时写 results.txt（便于你保留老格式）"""
        self.logger.info(msg)
        with open(self.result_txt, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # ---------- 训练 ----------
    def train(self):
        # 将文件日志绑定到 run 目录
        self._attach_file_handler()

        acc_best = 0.0
        f1_best = 0.0
        best_f1_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            pbar = tqdm(self.data_loader['train'], mininterval=10, desc=f"Fold {self.fold_id} | Epoch {epoch+1}")
            for x, y, z in pbar:
                self.optimizer.zero_grad()
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True).long()
                z = z.cuda(non_blocking=True).long()

                logits, recon, mu, mu_tilde, reg_A = self.model(x, labels=y, domain_ids=z)

                # 1) 任务损失
                loss_task = self.ce_loss(logits.permute(0, 2, 1), y)

                # 2) 锚点/统计约束
                model_to_update = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                with torch.no_grad():
                    model_to_update.anchors.update(mu_tilde.detach(), y, z)
                loss_caa = model_to_update.anchors.caa_loss()
                loss_stat = model_to_update.anchors.stats_align_loss(mu_tilde, z)

                # 3) 组合损失
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

                losses.append(loss.detach().cpu().item())
                pbar.set_postfix(loss=np.mean(losses))

            optim_state = self.optimizer.state_dict()

            # ---- 验证 ----
            with torch.no_grad():
                model_to_eval = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                model_to_eval.freeze_unified_projection(strategy="avg")
                acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = self.val_eval.get_accuracy(self.model)

            lr_now = optim_state['param_groups'][0]['lr']
            time_min = (timer() - start_time) / 60.0

            # 控制台 + 文件日志（结构化）
            self._log_txt(
                (f"Epoch {epoch+1:03d} | train_loss={np.mean(losses):.5f} | "
                 f"val_acc={acc:.5f} | val_f1={f1:.5f} | "
                 f"wake={wake_f1:.5f} n1={n1_f1:.5f} n2={n2_f1:.5f} n3={n3_f1:.5f} rem={rem_f1:.5f} | "
                 f"lr={lr_now:.6f} | {time_min:.2f} min")
            )

            # 逐 epoch CSV（便于画图/对比）
            with open(self.epoch_csv, "a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([
                    _now(), self.fold_id, epoch + 1, lr_now,
                    float(np.mean(losses)),
                    float(acc), float(f1),
                    float(wake_f1), float(n1_f1), float(n2_f1), float(n3_f1), float(rem_f1)
                ])

            # 追踪最佳
            if acc > acc_best:
                best_f1_epoch = epoch + 1
                acc_best = acc
                f1_best = f1
                self.best_model_states = copy.deepcopy(self.model.state_dict())
                self._log_txt(f"[BEST@{best_f1_epoch:03d}] val_acc={acc_best:.5f} | val_f1={f1_best:.5f}")

        self._log_txt(f"Best@Epoch {best_f1_epoch:03d} -> val_acc={acc_best:.5f}, val_f1={f1_best:.5f}")
        test_acc, test_f1 = self.test(best_val_acc=acc_best, best_val_f1=f1_best)
        return test_acc, test_f1

    # ---------- 测试 ----------
    def test(self, best_val_acc: float, best_val_f1: float):
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            model_to_test = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            model_to_test.freeze_unified_projection(strategy="avg")
            self._frozen_uni_built = True

            self._log_txt("*************************** Test ***************************")
            test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
                test_n3_f1, test_rem_f1 = self.test_eval.get_accuracy(self.model)

            self._log_txt(f"Test: acc={test_acc:.5f}, f1={test_f1:.5f}")
            # 保存混淆矩阵到独立文件
            with open(self.test_cm_file, "w", encoding="utf-8") as fcm:
                fcm.write(str(test_cm) + "\n")

            self._log_txt(
                ("Class F1 -> "
                 f"wake={test_wake_f1:.5f}, n1={test_n1_f1:.5f}, n2={test_n2_f1:.5f}, "
                 f"n3={test_n3_f1:.5f}, rem={test_rem_f1:.5f}")
            )

            # 模型文件：包含测试指标与 fold/run，避免覆盖
            model_path = self.run_dir / (
                f"fold{self.fold_id}_"
                f"tacc_{test_acc:.5f}_tf1_{test_f1:.5f}_"
                f"{self.run_id}.pth"
            )
            torch.save(self.best_model_states, model_path)
            self._log_txt("Model saved -> " + str(model_path))

            # 写入共享聚合 CSV（所有折共用）
            with open(self.aggregate_csv, "a", newline="", encoding="utf-8") as faggr:
                writer = csv.writer(faggr)
                writer.writerow([
                    _now(), self.run_id, self.fold_id,
                    float(best_val_acc), float(best_val_f1),
                    float(test_acc), float(test_f1),
                    float(test_wake_f1), float(test_n1_f1), float(test_n2_f1),
                    float(test_n3_f1), float(test_rem_f1),
                    str(model_path)
                ])

        return test_acc, test_f1