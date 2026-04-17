import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
import config as cfg


class Trainer:
    """
    Handles training loop, validation, checkpointing, and resumption.

    Checkpoint structure (per fold):
        {exp_name}_fold{k}/
            checkpoint.pt     -> model weights, optimizer, scheduler, epoch, best_val_loss
            train_history.json -> per-epoch metrics
            status.json       -> {"completed": bool, "current_epoch": int}
    """

    def __init__(self, model, train_loader, val_loader, class_weights,
                 exp_name="exp0", fold=0, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.exp_name = exp_name
        self.fold = fold

        self.save_dir = os.path.join(cfg.CHECKPOINT_DIR, f"{exp_name}_fold{fold}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Weighted cross-entropy (ignore index -1 for padded positions)
        weight_tensor = torch.from_numpy(class_weights).float().to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=cfg.LR_PATIENCE, factor=cfg.LR_FACTOR
        )

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "history": self.history,
        }
        torch.save(state, os.path.join(self.save_dir, "checkpoint.pt"))
        if is_best:
            torch.save(state, os.path.join(self.save_dir, "best_model.pt"))

        status = {"completed": False, "current_epoch": epoch}
        with open(os.path.join(self.save_dir, "status.json"), "w") as f:
            json.dump(status, f)

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.save_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            return False

        print(f"  [RESUME] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt["best_val_loss"]
        self.patience_counter = ckpt["patience_counter"]
        self.history = ckpt["history"]
        print(f"  [RESUME] Resuming from epoch {self.start_epoch}, best_val_loss={self.best_val_loss:.4f}")
        return True

    def is_fold_complete(self):
        status_path = os.path.join(self.save_dir, "status.json")
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                status = json.load(f)
            return status.get("completed", False)
        return False

    def mark_complete(self):
        status = {"completed": True, "current_epoch": self.history["train_loss"].__len__() - 1}
        with open(os.path.join(self.save_dir, "status.json"), "w") as f:
            json.dump(status, f)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        n_batches = len(self.train_loader)

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)  # (B, S, C)
            logits_flat = logits.reshape(-1, cfg.NUM_CLASSES)
            y_flat = y.reshape(-1)

            loss = self.criterion(logits_flat, y_flat)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            mask = y_flat != -1
            if mask.any():
                preds = logits_flat[mask].argmax(dim=1)
                total_correct += (preds == y_flat[mask]).sum().item()
                total_count += mask.sum().item()

            if (batch_idx + 1) % max(1, n_batches // 5) == 0 or batch_idx == n_batches - 1:
                running_acc = total_correct / max(total_count, 1) * 100
                print(f"    batch {batch_idx+1}/{n_batches} | "
                      f"loss: {loss.item():.4f} | acc: {running_acc:.1f}%", flush=True)

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_acc = total_correct / max(total_count, 1)
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            logits_flat = logits.reshape(-1, cfg.NUM_CLASSES)
            y_flat = y.reshape(-1)

            loss = self.criterion(logits_flat, y_flat)
            total_loss += loss.item() * x.size(0)

            mask = y_flat != -1
            if mask.any():
                preds = logits_flat[mask].argmax(dim=1)
                total_correct += (preds == y_flat[mask]).sum().item()
                total_count += mask.sum().item()

        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_acc = total_correct / max(total_count, 1)
        return avg_loss, avg_acc

    def train(self):
        """Full training loop with early stopping, checkpointing, and resumption."""
        if self.is_fold_complete():
            history_path = os.path.join(self.save_dir, "train_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    self.history = json.load(f)
            print(f"  [SKIP] {self.exp_name} fold {self.fold} already complete.")
            return self.history

        resumed = self.load_checkpoint()
        if not resumed:
            print(f"  [START] Training {self.exp_name} fold {self.fold} from scratch")

        for epoch in range(self.start_epoch, cfg.MAX_EPOCHS):
            t0 = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n  Epoch {epoch+1}/{cfg.MAX_EPOCHS} | lr={current_lr:.2e}")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step(val_loss)

            elapsed = time.time() - t0
            print(f"  => train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | {elapsed:.0f}s")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ** New best val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{cfg.EARLY_STOP_PATIENCE}")

            self.save_checkpoint(epoch, is_best=is_best)

            if self.patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"  [EARLY STOP] No improvement for {cfg.EARLY_STOP_PATIENCE} epochs.")
                break

        self.mark_complete()
        print(f"  [DONE] {self.exp_name} fold {self.fold} complete. Best val_loss={self.best_val_loss:.4f}")

        # Save final history
        with open(os.path.join(self.save_dir, "train_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def load_best_model(self):
        best_path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  [LOAD] Best model loaded (val_loss={ckpt['best_val_loss']:.4f})")
        else:
            print(f"  [WARN] No best_model.pt found, using current weights")
