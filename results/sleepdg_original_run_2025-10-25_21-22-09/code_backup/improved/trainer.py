# improved/trainer.py
# (基于 xxy751/sleepdg-1/SleepDG-1-ffaed6f5c03de5f43d87f2c73a9fb7dabb05ccf0/newtraining.py 修改)

import os
import copy
import logging
from datetime import datetime
from timeit import default_timer as timer
from pathlib import Path
from typing import Dict, Tuple, Optional
from types import SimpleNamespace
import importlib # Still needed for dynamic model loading within __init__

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- CORRECTED IMPORTS relative to project root ---
# These modules are in the top-level directory or packages
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from losses.double_alignment import CORAL
from losses.ae_loss import AELoss
from utils.allutils import (
    ensure_dir, MetricsLogger, build_ratio_loader,
    plot_curves_for_fold, extract_features_for_tsne, tsne_compare_plot,
    write_aggregate_row, _now # Make sure _now is imported
)
# Note: No 'from models.model import Model' here anymore

class Trainer(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params

        # --- Directory setup (remains the same) ---
        self.model_dir = Path(params.model_dir)
        self.fold_id = params.fold
        self.fold_dir = ensure_dir(self.model_dir / f"fold{self.fold_id}")
        self.allfold_dir = ensure_dir(self.model_dir / "allfold")

        # --- Data (remains the same) ---
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

        # --- Model (Corrected Dynamic Loading) ---
        # No change needed here if main.py correctly selected this Trainer
        # We need to load the *correct* Model class associated with this trainer
        try:
            # 使用新的文件夹名称 'improved_models'
            model_module = importlib.import_module('improved.models.model')  # <-- 修改这里
            ModelClass = model_module.Model
            print(f"[INFO] Using IMPROVED Model class from improved.models.model.")  # <-- 修改这里
            model = ModelClass(params)
        except (ImportError, AttributeError, ValueError) as e:
            print(f"[ERROR] Failed to load IMPROVED model class: {e}")
            raise

        self.model = model
        # --- GPU setup (remains the same) ---
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.cuda()

        # --- Loss/Lambda (remains the same) ---
        self.lambda_ae = getattr(params, "lambda_ae", 1.0)
        self.lambda_coral = getattr(params, "lambda_coral", 0.0)
        self.ce_loss = CrossEntropyLoss(label_smoothing=getattr(params, "label_smoothing", 0.1)).cuda()
        self.coral_loss = CORAL().cuda() if self.lambda_coral > 0 else None
        self.ae_loss = AELoss().cuda() if self.lambda_ae > 0 else None
        self.lmb_caa = getattr(params, "lambda_caa", 0.3)
        self.lmb_stat = getattr(params, "lambda_stat", 0.2)
        self.lmb_Areg = getattr(params, "lambda_Areg", 0.1)

        # --- Optimizer & Scheduler (remains the same) ---
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
        self._frozen_uni_built = False

        # --- Logging and Results (remains the same) ---
        self.logger = self._setup_logger()
        self.metrics_logger = MetricsLogger(self.fold_dir, self.fold_id)
        self.best_model_states = None
        self.result_txt = self.fold_dir / "results.txt"
        self.test_cm_file = self.fold_dir / f"test_confusion_fold{self.fold_id}.txt" # Define test_cm_file

    # --- _setup_logger and _log_txt methods remain the same as previous version ---
    def _setup_logger(self):
        """Sets up logger pointing to the fold directory."""
        logger = logging.getLogger(f"TrainerFold{self.fold_id}")
        # Check if logger already has handlers to prevent duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        # Prevent propagating to root logger if desired
        logger.propagate = False

        log_format = "%(asctime)s | %(levelname)s | fold=%(fold)d | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler (append mode)
        # Ensure directory exists before creating handler
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.fold_dir / "run.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Inject fold variable (keep if it works)
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

    # --- train method remains largely the same, ensure params access ---
    def train(self) -> Dict:
        try:
            with open(self.result_txt, "a", encoding="utf-8") as f:
                 f.write(f"[{_now()}] Start fold {self.fold_id}, Target: {getattr(self.params, 'target_domains', 'N/A')}\n")
                 f.write(f"Run ID: {getattr(self.params, 'run_name', 'N/A')}\n")
                 f.write(f"Data ratio: {self.data_ratio}\n\n")
        except Exception as e:
             self.logger.error(f"Failed to write initial info to results.txt: {e}")

        self._log_txt(f"===== [START] Training Fold {self.fold_id} / Target: {getattr(self.params, 'target_domains', 'N/A')} =====")
        acc_best = 0.0
        f1_best = 0.0
        best_f1_epoch = 0

        for epoch in range(self.params.epochs): # Use params.epochs
            self.model.train()
            start_time = timer()
            losses = []
            task_losses, caa_losses, stat_losses, areg_losses, ae_losses, coral_losses = [], [], [], [], [], []

            pbar_desc = f"Fold {self.fold_id} | Epoch {epoch+1}/{self.params.epochs}"
            pbar = tqdm(self.data_loader['train'], mininterval=5, desc=pbar_desc, leave=False) # leave=False might be cleaner

            for x, y, z in pbar:
                self.optimizer.zero_grad()
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True).long()
                z = z.cuda(non_blocking=True).long()

                logits, recon, mu, mu_tilde, reg_A = self.model(x, labels=y, domain_ids=z)
                loss_task = self.ce_loss(logits.permute(0, 2, 1), y)
                task_losses.append(loss_task.item())

                model_to_update = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                loss_caa = torch.tensor(0.0).cuda()
                loss_stat = torch.tensor(0.0).cuda()
                if hasattr(model_to_update, 'anchors'):
                    with torch.no_grad():
                        model_to_update.anchors.update(mu_tilde.detach(), y, z)
                    loss_caa = model_to_update.anchors.caa_loss()
                    loss_stat = model_to_update.anchors.stats_align_loss(mu_tilde, z)
                caa_losses.append(loss_caa.item())
                stat_losses.append(loss_stat.item())
                # Ensure reg_A is a scalar tensor or use .mean() safely
                current_reg_A = reg_A.mean() if isinstance(reg_A, torch.Tensor) else torch.tensor(reg_A).cuda()
                areg_losses.append(current_reg_A.item())


                loss = (loss_task +
                        self.lmb_caa * loss_caa +
                        self.lmb_stat * loss_stat +
                        self.lmb_Areg * current_reg_A) # Use the scalar tensor

                loss_ae_val = torch.tensor(0.0).cuda()
                if self.lambda_ae > 0 and self.ae_loss is not None:
                    loss_ae_val = self.ae_loss(x, recon)
                    loss = loss + loss_ae_val * self.lambda_ae
                ae_losses.append(loss_ae_val.item())

                loss_coral_val = torch.tensor(0.0).cuda()
                if self.lambda_coral > 0 and self.coral_loss is not None:
                    loss_coral_val = self.coral_loss(mu, z)
                    loss = loss + loss_coral_val * self.lambda_coral
                coral_losses.append(loss_coral_val.item())

                loss.backward()
                clip_val = getattr(self.params, 'clip_value', 0)
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
                self.optimizer.step()
                self.scheduler.step()

                losses.append(loss.item())
                pbar.set_postfix(loss=np.mean(losses[-100:]), task=loss_task.item(), refresh=False) # Show recent avg loss

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
                    model_to_eval.eval() # Set to eval mode for validation
                    if hasattr(model_to_eval, 'freeze_unified_projection'):
                        model_to_eval.freeze_unified_projection(strategy="avg")
                        self._frozen_uni_built = True
                    else:
                        self._frozen_uni_built = False

                    val_acc, val_f1, val_cm, val_wake_f1, val_n1_f1, val_n2_f1, \
                        val_n3_f1, val_rem_f1, val_kappa, val_report = self.val_eval.get_accuracy(self.model)
            else:
                 self.logger.warning("Validation loader is empty or not provided. Skipping validation.")

            # --- Logging ---
            avg_loss = np.mean(losses) if losses else 0.0
            avg_task = np.mean(task_losses) if task_losses else 0.0
            avg_caa = np.mean(caa_losses) if caa_losses else 0.0
            avg_stat = np.mean(stat_losses) if stat_losses else 0.0
            avg_areg = np.mean(areg_losses) if areg_losses else 0.0
            avg_ae = np.mean(ae_losses) if ae_losses else 0.0
            avg_coral = np.mean(coral_losses) if coral_losses else 0.0
            loss_detail = (f"task={avg_task:.3f}, caa={avg_caa:.3f}, stat={avg_stat:.3f}, "
                           f"Areg={avg_areg:.3f}, ae={avg_ae:.3f}, coral={avg_coral:.3f}")

            msg = (f"Epoch {epoch+1:03d} | train_loss={avg_loss:.5f} ({loss_detail}) | "
                   f"val_acc={val_acc:.5f} | val_f1={val_f1:.5f} | val_kappa={val_kappa:.5f} | "
                   f"W={val_wake_f1:.3f} N1={val_n1_f1:.3f} N2={val_n2_f1:.3f} N3={val_n3_f1:.3f} R={val_rem_f1:.3f} | "
                   f"lr={lr_now:.6f} | {time_min:.2f} min")
            self._log_txt(msg)

            self.metrics_logger.log_epoch(
                time_str=_now(), epoch=epoch+1, lr=lr_now, train_loss=avg_loss,
                train_acc=None, train_f1=None,
                val_acc=val_acc, val_f1=val_f1, # Add val_kappa if logger supports it
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

        # --- Load state into a temporary model instance on CPU first ---
        # Re-initialize the model structure without DataParallel wrapper
        try:
            model_module = importlib.import_module('.models.model', package='improved')
            ModelClass = model_module.Model
            temp_model = ModelClass(self.params)
            temp_model.load_state_dict(self.best_model_states)
            self.logger.info("Best model state loaded into temporary model for testing.")
            # Move model to GPU(s)
            if torch.cuda.device_count() > 1 and isinstance(self.model, nn.DataParallel):
                # If training used DataParallel, wrap the loaded model again for eval
                model_to_test = nn.DataParallel(temp_model).cuda()
                self.logger.info("Wrapping loaded model with DataParallel for evaluation.")
            else:
                 model_to_test = temp_model.cuda() # Single GPU or CPU

        except (RuntimeError, ImportError, AttributeError) as e:
            self.logger.error(f"Failed to load best model state or re-initialize model: {e}")
            return {"error": f"Failed to load best model state: {e}"}

        test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
            test_n3_f1, test_rem_f1, test_kappa, test_report = 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "N/A"

        if self.test_eval and len(self.data_loader['test']) > 0:
            with torch.no_grad():
                model_to_test.eval() # Set to eval mode

                # Check and potentially freeze unified projection on the *actual* model instance
                eval_model_instance = model_to_test.module if isinstance(model_to_test, nn.DataParallel) else model_to_test
                if not self._frozen_uni_built and hasattr(eval_model_instance, 'freeze_unified_projection'):
                     self.logger.warning("Unified projection not built during training. Building with 'avg'.")
                     eval_model_instance.freeze_unified_projection(strategy="avg")
                     self._frozen_uni_built = True

                self._log_txt("*************************** Test ***************************")
                test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
                    test_n3_f1, test_rem_f1, test_kappa, test_report = self.test_eval.get_accuracy(model_to_test)

                # --- Logging ---
                self._log_txt(f"Test: acc={test_acc:.5f}, f1={test_f1:.5f}")
                self._log_txt(f"Cohen's Kappa: {test_kappa:.5f}")
                self._log_txt("Confusion Matrix:\n" + str(test_cm))
                self._log_txt(
                    ("Class F1 -> W={:.3f}, N1={:.3f}, N2={:.3f}, N3={:.3f}, R={:.3f}"
                     .format(test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1))
                )
                self._log_txt("\nClassification Report:\n" + str(test_report))
                # --- Save CM and Report ---
                try:
                    with open(self.test_cm_file, "w", encoding="utf-8") as fcm:
                        fcm.write("Confusion Matrix:\n")
                        np.savetxt(fcm, test_cm, fmt="%d") # Save numpy array properly
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
            self._log_txt("Model saved -> " + str(model_path))
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            model_path = Path("N/A") # Use Path object for consistency

        # --- t-SNE ---
        try:
            device = next(model_to_test.parameters()).device # Get device model is on
            # Need the non-DP model for feature extraction if DP was used
            model_for_tsne = model_to_test.module if isinstance(model_to_test, nn.DataParallel) else model_to_test
            model_for_tsne.eval()

            RAW, REP, Y = extract_features_for_tsne(
                model_for_tsne, self.data_loader['test'], device, take="mu_tilde" # Assuming improved uses mu_tilde
            )
            tsne_compare_plot(
                raw_X=RAW, rep_X=REP, y=Y, out_dir=self.fold_dir,
                title_prefix=f"Fold{self.fold_id} Test t-SNE", filename_prefix="tsne"
            )
            self.logger.info(f"t-SNE plots generated in {self.fold_dir}")
        except Exception as e:
            self._log_txt(f"[WARN] Failed to generate t-SNE plot: {e}")
            self.logger.warning(f"t-SNE plot generation failed: {e}", exc_info=True)


        # --- Prepare Results Dict ---
        result_dict = {
            "time": _now(),
            "run_id": getattr(self.params, "run_name", f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
            "fold": self.fold_id,
            "best_val_acc": float(best_val_acc), "best_val_f1": float(best_val_f1),
            "test_acc": float(test_acc), "test_f1": float(test_f1),
            "test_kappa": float(test_kappa),
            "wake_f1": float(test_wake_f1), "n1_f1": float(test_n1_f1),
            "n2_f1": float(test_n2_f1), "n3_f1": float(test_n3_f1),
            "rem_f1": float(test_rem_f1),
            "model_path": str(model_path)
        }

        return result_dict