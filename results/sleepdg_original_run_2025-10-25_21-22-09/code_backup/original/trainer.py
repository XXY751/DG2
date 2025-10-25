# original/trainer.py
# (基于 wjq-learning/sleepdg/SleepDG-01d209d4fda874933a277a88bccdca6e3835e884/trainer.py 修改)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# Remove 'import torch' again if already imported
from types import SimpleNamespace # Import SimpleNamespace
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import csv
import copy
from timeit import default_timer as timer
import importlib # For dynamic model loading

# --- CORRECTED IMPORTS relative to project root ---
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from losses.double_alignment import CORAL
from losses.ae_loss import AELoss
from utils.allutils import ensure_dir, MetricsLogger, build_ratio_loader, _now, plot_curves_for_fold
# Note: No 'from models.model import Model' here anymore

class Trainer(object):
    def __init__(self, params: SimpleNamespace): # Receive SimpleNamespace
        self.params = params

        # --- Directory setup ---
        self.model_dir = Path(params.model_dir)
        self.fold_id = params.fold
        self.fold_dir = ensure_dir(self.model_dir / f"fold{self.fold_id}")
        self.allfold_dir = ensure_dir(self.model_dir / "allfold")

        # --- Data ---
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()
        self.data_ratio = getattr(params, "data_ratio", 1.0)
        if 0 < self.data_ratio < 1.0:
            print(f"[INFO] Using {self.data_ratio*100:.1f}% of training data.")
            self.data_loader['train'] = build_ratio_loader(
                self.data_loader['train'],
                self.data_ratio,
                seed=getattr(params, 'seed', 42)
            )
        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.best_model_states = None

        # --- Model (Corrected Dynamic Loading) ---
        try:
            # Use relative import path within the 'original' package
            model_module = importlib.import_module('.models.model', package='original') # Relative import
            ModelClass = model_module.Model
            print(f"[INFO] Using ORIGINAL Model class from original.models.model.")
            model = ModelClass(params)
        except (ImportError, AttributeError, ValueError) as e:
            print(f"[ERROR] Failed to load ORIGINAL model class: {e}")
            raise

        self.model = model
        # GPU setup
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.cuda()

        # --- Loss ---
        self.ce_loss = CrossEntropyLoss(label_smoothing=getattr(params, "label_smoothing", 0.0)).cuda()
        self.lambda_coral = getattr(params, "lambda_coral", 1.0)
        self.lambda_ae = getattr(params, "lambda_ae", 1.0)
        self.coral_loss = CORAL().cuda() if self.lambda_coral > 0 else None
        self.ae_loss = AELoss().cuda() if self.lambda_ae > 0 else None

        # --- Optimizer & Scheduler ---
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=getattr(params, 'weight_decay', params.lr / 10)
        )
        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=params.epochs * self.data_length
        )

        print(self.model)

        # --- Logging ---
        self.logger = self._setup_logger()
        self.metrics_logger = MetricsLogger(self.fold_dir, self.fold_id)
        self.result_txt = self.fold_dir / "results.txt"
        self.test_cm_file = self.fold_dir / f"test_confusion_fold{self.fold_id}.txt"

    # --- _setup_logger and _log_txt methods (same as improved version) ---
    def _setup_logger(self):
        """Sets up logger pointing to the fold directory."""
        logger = logging.getLogger(f"TrainerFold{self.fold_id}")
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False

        log_format = "%(asctime)s | %(levelname)s | fold=%(fold)d | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.fold_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.fold_dir / "run.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.fold = self.fold_id
            return record
        logging.setLogRecordFactory(record_factory)
        return logger

    def _log_txt(self, msg: str):
        """Logs to both logger and results.txt."""
        self.logger.info(msg)
        try:
            with open(self.result_txt, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to results.txt: {e}")

    # --- train method (similar structure, different losses) ---
    def train(self) -> Dict:
        try:
            with open(self.result_txt, "a", encoding="utf-8") as f:
                f.write(f"[{_now()}] Start fold {self.fold_id}, Target: {getattr(self.params, 'target_domains', 'N/A')}\n")
                f.write(f"Run ID: {getattr(self.params, 'run_name', 'N/A')}\n")
                f.write(f"Data ratio: {self.data_ratio}\n\n")
        except Exception as e:
             self.logger.error(f"Failed to write initial info to results.txt: {e}")

        self._log_txt(f"===== [START] Training Fold {self.fold_id} / Target: {getattr(self.params, 'target_domains', 'N/A')} (Original Model) =====")

        acc_best = 0.0
        f1_best = 0.0
        best_f1_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            ce_losses, coral_losses, ae_losses = [], [], []

            pbar_desc = f"Fold {self.fold_id} | Epoch {epoch+1}/{self.params.epochs} (Original)"
            pbar = tqdm(self.data_loader['train'], mininterval=5, desc=pbar_desc, leave=False)

            for x, y, z in pbar:
                self.optimizer.zero_grad()
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True).long()
                z = z.cuda(non_blocking=True).long() # domain id

                # Original model forward
                pred, recon, mu = self.model(x)

                loss_ce = self.ce_loss(pred.transpose(1, 2), y)
                ce_losses.append(loss_ce.item())

                loss_coral = torch.tensor(0.0).cuda()
                if self.lambda_coral > 0 and self.coral_loss is not None:
                    loss_coral = self.coral_loss(mu, z) # Apply CORAL on mu
                coral_losses.append(loss_coral.item())

                loss_ae = torch.tensor(0.0).cuda()
                if self.lambda_ae > 0 and self.ae_loss is not None:
                    loss_ae = self.ae_loss(x, recon) # Apply AE loss
                ae_losses.append(loss_ae.item())

                loss = loss_ce + self.lambda_coral * loss_coral + self.lambda_ae * loss_ae
                losses.append(loss.item())

                loss.backward()
                clip_val = getattr(self.params, 'clip_value', 0)
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                self.optimizer.step()
                self.scheduler.step()

                pbar.set_postfix(loss=np.mean(losses[-100:]), ce=loss_ce.item(), refresh=False)

            # --- Epoch End ---
            optim_state = self.optimizer.state_dict()
            lr_now = optim_state['param_groups'][0]['lr']
            time_min = (timer() - start_time) / 60.0

            # --- Validation ---
            val_acc, val_f1, val_cm, val_wake_f1, val_n1_f1, val_n2_f1, \
                val_n3_f1, val_rem_f1, val_kappa, val_report = 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""
            if self.val_eval and len(self.data_loader['val']) > 0:
                with torch.no_grad():
                    model_to_eval = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    model_to_eval.eval() # Set to eval mode

                    # Original model doesn't have freeze_unified_projection
                    val_acc, val_f1, val_cm, val_wake_f1, val_n1_f1, val_n2_f1, \
                        val_n3_f1, val_rem_f1, val_kappa, val_report = self.val_eval.get_accuracy(self.model)
            else:
                 self.logger.warning("Validation loader is empty or not provided. Skipping validation.")


            # --- Logging ---
            avg_loss = np.mean(losses) if losses else 0.0
            avg_ce = np.mean(ce_losses) if ce_losses else 0.0
            avg_coral = np.mean(coral_losses) if coral_losses else 0.0
            avg_ae = np.mean(ae_losses) if ae_losses else 0.0
            loss_detail = f"ce={avg_ce:.3f}, coral={avg_coral:.3f}, ae={avg_ae:.3f}"

            msg = (f"Epoch {epoch+1:03d} | train_loss={avg_loss:.5f} ({loss_detail}) | "
                   f"val_acc={val_acc:.5f} | val_f1={val_f1:.5f} | val_kappa={val_kappa:.5f} | "
                   f"W={val_wake_f1:.3f} N1={val_n1_f1:.3f} N2={val_n2_f1:.3f} N3={val_n3_f1:.3f} R={val_rem_f1:.3f} | "
                   f"lr={lr_now:.6f} | {time_min:.2f} min")
            self._log_txt(msg)

            self.metrics_logger.log_epoch(
                time_str=_now(), epoch=epoch+1, lr=lr_now, train_loss=avg_loss,
                train_acc=None, train_f1=None,
                val_acc=val_acc, val_f1=val_f1,
                wake_f1=val_wake_f1, n1_f1=val_n1_f1, n2_f1=val_n2_f1,
                n3_f1=val_n3_f1, rem_f1=val_rem_f1
            )

            # --- Best Model Tracking ---
            if val_acc > acc_best:
                best_f1_epoch = epoch + 1
                acc_best = val_acc
                f1_best = val_f1
                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                self.best_model_states = copy.deepcopy(model_state)
                self._log_txt(f"[BEST@{best_f1_epoch:03d}] val_acc={acc_best:.5f} | val_f1={f1_best:.5f}")

        # --- Training End ---
        self._log_txt(f"Training finished. Best val @ Epoch {best_f1_epoch:03d} -> val_acc={acc_best:.5f}, val_f1={f1_best:.5f}")

        try:
            plot_curves_for_fold(self.metrics_logger.path(), out_dir=self.fold_dir)
            self.logger.info(f"Curves plotted for fold {self.fold_id} in {self.fold_dir}")
        except Exception as e:
            self.logger.error(f"Failed to plot curves for fold {self.fold_id}: {e}")

        if self.best_model_states is None and self.params.epochs > 0 :
             self.logger.warning("No best model state saved. Using last epoch model for testing.")
             last_epoch_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
             self.best_model_states = copy.deepcopy(last_epoch_state)
        elif self.best_model_states is None:
             self.logger.error("No model state available for testing.")
             return {"error": "No model state available for testing."}

        test_results_dict = self.test(best_val_acc=acc_best, best_val_f1=f1_best)
        return test_results_dict


    def test(self, best_val_acc: float, best_val_f1: float) -> Dict:
        if self.best_model_states is None:
             self._log_txt("[ERROR] No model state available for testing.")
             return {"error": "No model state available for testing."}

        # --- Load state ---
        try:
            model_module = importlib.import_module('.models.model', package='original')
            ModelClass = model_module.Model
            temp_model = ModelClass(self.params)
            temp_model.load_state_dict(self.best_model_states)
            self.logger.info("Best model state loaded into temporary model for testing.")
            if torch.cuda.device_count() > 1 and isinstance(self.model, nn.DataParallel):
                model_to_test = nn.DataParallel(temp_model).cuda()
            else:
                 model_to_test = temp_model.cuda()

        except (RuntimeError, ImportError, AttributeError) as e:
            self.logger.error(f"Failed to load best model state or re-initialize model: {e}")
            return {"error": f"Failed to load best model state: {e}"}

        test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
            test_n3_f1, test_rem_f1, test_kappa, test_report = 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "N/A"

        if self.test_eval and len(self.data_loader['test']) > 0:
            with torch.no_grad():
                model_to_test.eval()
                self._log_txt("*************************** Test ***************************")

                test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
                    test_n3_f1, test_rem_f1, test_kappa, test_report = self.test_eval.get_accuracy(model_to_test)

                self._log_txt("***************************Test results************************")
                self._log_txt(f"Test Evaluation: acc: {test_acc:.5f}, f1: {test_f1:.5f}")
                self._log_txt(f"Cohen's Kappa: {test_kappa:.5f}")
                self._log_txt("Confusion Matrix:\n" + str(test_cm))
                self._log_txt(
                    ("Class F1 -> W={:.3f}, N1={:.3f}, N2={:.3f}, N3={:.3f}, R={:.3f}"
                     .format(test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1))
                )
                self._log_txt("\nClassification Report:\n" + str(test_report))

                try:
                    with open(self.test_cm_file, "w", encoding="utf-8") as fcm:
                        fcm.write("Confusion Matrix:\n")
                        np.savetxt(fcm, test_cm, fmt="%d")
                        fcm.write("\n\nClassification Report:\n")
                        fcm.write(str(test_report))
                except Exception as e:
                     self.logger.error(f"Failed to save confusion matrix/report: {e}")
        else:
            self.logger.warning("Test loader is empty or not provided. Skipping testing.")

        # --- Save Model ---
        model_filename = f"fold{self.fold_id}_tacc_{test_acc:.5f}_tf1_{test_f1:.5f}.pth"
        model_path = self.fold_dir / model_filename
        try:
            torch.save(self.best_model_states, model_path)
            self._log_txt("the model is save in " + str(model_path))
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            model_path = Path("N/A")

        # --- No t-SNE for original model in this script ---

        # --- Prepare Results Dict ---
        result_dict = {
            "time": _now(),
            "run_id": getattr(self.params, "run_name", f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            "fold": self.fold_id,
            "best_val_acc": float(best_val_acc), "best_val_f1": float(best_val_f1),
            "test_acc": float(test_acc), "test_f1": float(test_f1),
            "test_kappa": float(test_kappa), # Include kappa
            "wake_f1": float(test_wake_f1), "n1_f1": float(test_n1_f1),
            "n2_f1": float(test_n2_f1), "n3_f1": float(test_n3_f1),
            "rem_f1": float(test_rem_f1),
            "model_path": str(model_path)
        }

        return result_dict