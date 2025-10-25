# /data/lijinyang/sleep/DG1/improved/models/regulator.py
# (替换为以下完整修正内容)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from typing import Tuple


# ==========================================================
# 1. 梯度反转层 (Gradient Reverse Layer - GRL)
# ==========================================================
class GradientReverseFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ==========================================================
# 2. 域不变投影 (LDP - Latent Domain Projection)
# ==========================================================
class DomainProjectionLDP(nn.Module):
    """
    实现 LDP (Latent Domain Projection)
    - 针对 N 个源域，学习 N 个投影矩阵 A_i
    - 投影 F_i(mu) = A_i @ mu
    - 学习一个统一投影 A_uni (用于推理)
    - 计算 A_i 之间的正则化损失
    """

    def __init__(self, dim: int, num_domains: int, projection_type: str = "diag", lowrank_rank: int = 32,
                 dropout_p: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_domains = num_domains
        self.projection_type = projection_type.lower()
        self.lowrank_rank = lowrank_rank
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        print(f"[LDP] Init LDP: {projection_type} (dim={dim}, num_domains={num_domains}, r={lowrank_rank})")

        # 1. 定义 N 个域的投影
        if self.projection_type == "diag":
            # (N, D) - N 个对角矩阵 (存储为向量)
            # 使用 nn.Parameter 来存储这 N 个向量
            self.diag_projections = nn.Parameter(torch.ones(num_domains, dim))  # 初始化为 1
        elif self.projection_type == "lowrank":
            # (N, D, r) 和 (N, r, D) - N 个低秩矩阵 (A_i = L_i @ R_i)
            # 使用 ModuleList 存储 ModuleDict
            self.projections_A = nn.ModuleList()
            for _ in range(num_domains):
                L_i = nn.Parameter(torch.randn(dim, lowrank_rank) * 0.01)
                R_i = nn.Parameter(torch.randn(lowrank_rank, dim) * 0.01)
                self.projections_A.append(nn.ModuleDict({"L": L_i, "R": R_i}))
        elif self.projection_type == "full":
            # (N, D, D) - N 个全矩阵
            # 使用 ModuleList 存储 Linear 层
            self.projections_A = nn.ModuleList()
            for _ in range(num_domains):
                self.projections_A.append(nn.Linear(dim, dim, bias=False))
        elif self.projection_type == "none":
            pass  # 不做任何投影
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

        # 2. 注册统一投影矩阵 A_uni (用于推理)
        self.register_buffer("A_uni_diag", torch.ones(dim))  # (D,)
        self.register_buffer("A_uni_L", torch.zeros(dim, lowrank_rank))  # (D, r)
        self.register_buffer("A_uni_R", torch.zeros(lowrank_rank, dim))  # (r, D)
        self.register_buffer("A_uni_full", torch.eye(dim))  # (D, D)
        self.A_uni_built = False

    def _apply_proj(self, x: torch.Tensor, i: int, use_uni: bool = False) -> torch.Tensor:
        """对 (B, D) 的输入 x 应用第 i 个投影或 A_uni"""
        if self.projection_type == "diag":
            # A_i 是 (D,)
            A_i = self.A_uni_diag if use_uni else self.diag_projections[i]  # 从 (N, D) 中取出第 i 行
            return x * A_i  # (B, D) * (D,) -> (B, D)
        elif self.projection_type == "lowrank":
            L = self.A_uni_L if use_uni else self.projections_A[i]["L"]
            R = self.A_uni_R if use_uni else self.projections_A[i]["R"]
            # x @ (L @ R) -> (B, D) @ (D,r) @ (r,D) -> (B, D)
            return (x @ L) @ R
        elif self.projection_type == "full":
            A_i_module = self.projections_A[i]
            # A_uni_full 是 (D,D) eye, A_i_module.weight 是 (D,D), .t() 转置
            A = self.A_uni_full if use_uni else A_i_module.weight.t()
            return x @ A
        elif self.projection_type == "none":
            return x
        return x

    def forward(self, mu: torch.Tensor, domain_ids: torch.Tensor, use_uni: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        mu: (B, D) 特征
        domain_ids: (B,) 域标签
        use_uni: 是否强制使用 A_uni (推理时)
        """
        if self.projection_type == "none":
            return mu, torch.tensor(0.0).to(mu.device)  # 返回原特征和 0 损失

        if use_uni:
            if not self.A_uni_built:
                print("[WARN] A_uni not built, using default (Identity). Call build_A_uni() first.")
            mu_tilde = self._apply_proj(mu, 0, use_uni=True)
            return self.dropout(mu_tilde), torch.tensor(0.0).to(mu.device)  # 推理时无正则化损失

        # --- 训练时 ---
        mu_tilde = torch.zeros_like(mu)
        for i in range(self.num_domains):
            mask = (domain_ids == i)
            if mask.any():
                mu_i = mu[mask]
                mu_tilde[mask] = self._apply_proj(mu_i, i, use_uni=False)

        # 计算正则化损失 (A_i 之间的差异)
        reg_loss = self.calculate_A_reg_loss(mu.device)

        return self.dropout(mu_tilde), reg_loss

    def calculate_A_reg_loss(self, device) -> torch.Tensor:
        """计算投影矩阵之间的正则化损失 L_Areg"""
        if self.num_domains <= 1 or self.projection_type == "none":
            return torch.tensor(0.0).to(device)

        losses = []
        if self.projection_type == "diag":
            # self.diag_projections 是 (N, D)
            As = self.diag_projections
            A_mean = As.mean(dim=0)  # (D,)
            for i in range(self.num_domains):
                losses.append(F.mse_loss(As[i], A_mean))
        elif self.projection_type == "lowrank":
            # A_i = L_i @ R_i
            As = []
            for i in range(self.num_domains):
                As.append(self.projections_A[i]["L"] @ self.projections_A[i]["R"])
            As = torch.stack(As)  # (N, D, D)
            A_mean = As.mean(dim=0)  # (D, D)
            for A_i_matrix in As:
                losses.append(F.mse_loss(A_i_matrix, A_mean))
        elif self.projection_type == "full":
            # (N, D, D)
            As = torch.stack([A.weight for A in self.projections_A])
            A_mean = As.mean(dim=0)
            for A_i_weight in As:
                losses.append(F.mse_loss(A_i_weight, A_mean))

        return torch.stack(losses).mean() if losses else torch.tensor(0.0).to(device)

    def _get_param_device(self) -> torch.device:
        """获取模型参数所在的设备"""
        if self.projection_type == "diag":
            return self.diag_projections.device
        elif self.projection_type in ("lowrank", "full") and self.num_domains > 0:
            return next(self.projections_A[0].parameters()).device
        else:
            # "none" or 0 domains, fallback to buffer
            return self.A_uni_diag.device

    @torch.no_grad()
    def build_A_uni(self, strategy="avg", weights=None):
        """构建统一投影 A_uni (用于推理)"""
        if self.projection_type == "none" or self.num_domains == 0:
            self.A_uni_built = True
            return

        print(f"[LDP] Building A_uni using strategy: {strategy}")
        if strategy == "avg":
            weights = torch.ones(self.num_domains) / self.num_domains
        elif strategy == "weighted":
            assert weights is not None, "Weighted strategy needs weights"
            weights = F.softmax(weights, dim=0)
        else:  # "identity" or unknown
            self.A_uni_built = True
            return  # A_uni 保持默认值 (Identity)

        # 获取设备
        device = self._get_param_device()
        weights = weights.to(device)

        if self.projection_type == "diag":
            # As = self.diag_projections (N, D)
            # (D,) = (1, N) @ (N, D) -> squeeze
            A_uni = (weights.unsqueeze(0) @ self.diag_projections).squeeze(0)
            self.A_uni_diag.data = A_uni
        elif self.projection_type == "lowrank":
            # (D, r) = sum(w_i * L_i)
            L_uni = torch.sum(torch.stack([w * self.projections_A[i]["L"] for i, w in enumerate(weights)]), dim=0)
            # (r, D) = sum(w_i * R_i)
            R_uni = torch.sum(torch.stack([w * self.projections_A[i]["R"] for i, w in enumerate(weights)]), dim=0)
            self.A_uni_L.data = L_uni
            self.A_uni_R.data = R_uni
        elif self.projection_type == "full":
            # (D, D) = sum(w_i * A_i.weight)
            A_uni = torch.sum(torch.stack([w * self.projections_A[i].weight for i, w in enumerate(weights)]), dim=0)
            self.A_uni_full.data = A_uni

        self.A_uni_built = True
        print("[LDP] A_uni built.")


# ==========================================================
# 3. 锚点对齐 (CAA - Class-wise Anchor Alignment)
# ==========================================================
class AnchorBankCAA(nn.Module):
    """
    实现 CAA (Class-wise Anchor Alignment) 和 统计对齐 (Stats Alignment)
    - 维护 N_domain * N_class 个锚点 (EMA 更新)
    - 计算 CAA 损失 (类内/类间)
    - 计算统计对齐损失 (域间均值/方差)
    """

    def __init__(self, num_classes: int, feat_dim: int, num_domains: int,
                 ema_momentum: float = 0.9, enable_stats_alignment: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.num_domains = num_domains
        self.momentum = ema_momentum
        self.enable_stats = enable_stats_alignment

        print(f"[CAA] Init AnchorBank: {num_domains} domains, {num_classes} classes, dim={feat_dim}")

        # 1. 锚点 (Anchors)
        # (N_domain, N_class, D)
        self.register_buffer("anchors", torch.zeros(num_domains, num_classes, feat_dim))
        # (N_domain, N_class) 记录锚点是否被初始化 (EMA更新过)
        self.register_buffer("anchors_initialized", torch.zeros(num_domains, num_classes).bool())

        # 2. 统计 (Stats)
        if self.enable_stats:
            # (N_domain, D)
            self.register_buffer("domain_means", torch.zeros(num_domains, feat_dim))
            # (N_domain, D)
            self.register_buffer("domain_vars", torch.ones(num_domains, feat_dim))
            # (N_domain,)
            self.register_buffer("stats_initialized", torch.zeros(num_domains).bool())

    @torch.no_grad()
    def update(self, mu_tilde: torch.Tensor, y_true: torch.Tensor, d_true: torch.Tensor):
        """
        EMA 更新锚点和统计量
        mu_tilde: (B, D) 经过 LDP 后的特征
        y_true: (B,) or (B, T) 标签
        d_true: (B,) 域标签
        """
        # 确保 y_true 和 mu_tilde 维度匹配
        # y_true 可能是 (B, T), mu_tilde 可能是 (B, D)
        # 假设 model.py 传过来的 mu_tilde 已经是 (B, D) or (B*T, D)
        # 假设 y_true 是 (B, T), mu_tilde 是 (B, D)

        y_flat = y_true.reshape(-1)  # (B*T,)

        # 检查 mu_tilde 维度
        if mu_tilde.shape[0] == y_true.shape[0] and y_true.dim() == 2:  # mu(B,D), y(B,T)
            B, T = y_true.shape
            D = mu_tilde.shape[1]
            # mu_tilde (B, D) -> (B, 1, D) -> (B, T, D)
            mu_expanded = mu_tilde.unsqueeze(1).expand(B, T, D)
            mu_flat = mu_expanded.reshape(B * T, D)  # (B*T, D)
            # d_true (B,) -> (B, 1) -> (B, T)
            d_expanded = d_true.unsqueeze(1).expand(B, T)
            d_flat = d_expanded.reshape(B * T)  # (B*T,)
        elif mu_tilde.shape[0] == y_flat.shape[0]:  # mu(B*T, D), y(B,T) or mu(B,D), y(B,)
            mu_flat = mu_tilde
            d_flat = d_true.repeat_interleave(y_flat.shape[0] // d_true.shape[0])
        else:
            # Fallback: 无法安全匹配，可能出错
            # 假设 mu_tilde(B,D), y_true(B,)
            mu_flat = mu_tilde
            y_flat = y_true
            d_flat = d_true

        for d in range(self.num_domains):
            mask_d = (d_flat == d)
            if not mask_d.any(): continue

            mu_d = mu_flat[mask_d]  # (B_d, D)
            y_d = y_flat[mask_d]  # (B_d,)

            # 2.1 更新锚点 (Anchors)
            for c in range(self.num_classes):
                mask_c = (y_d == c)
                if not mask_c.any(): continue

                mu_c = mu_d[mask_c]  # (B_c, D)
                batch_anchor_c = mu_c.mean(dim=0)  # (D,)

                if self.anchors_initialized[d, c]:
                    # EMA 更新
                    self.anchors[d, c] = (self.momentum * self.anchors[d, c]) + \
                                         ((1.0 - self.momentum) * batch_anchor_c)
                else:
                    # 第一次初始化
                    self.anchors[d, c] = batch_anchor_c
                    self.anchors_initialized[d, c] = True

            # 2.2 更新统计量 (Stats)
            if self.enable_stats and mu_d.shape[0] > 1:
                batch_mean = mu_d.mean(dim=0)  # (D,)
                batch_var = mu_d.var(dim=0, unbiased=True)  # (D,)

                if self.stats_initialized[d]:
                    # EMA 更新
                    self.domain_means[d] = (self.momentum * self.domain_means[d]) + \
                                           ((1.0 - self.momentum) * batch_mean)
                    self.domain_vars[d] = (self.momentum * self.domain_vars[d]) + \
                                          ((1.0 - self.momentum) * batch_var)
                else:
                    # 第一次初始化
                    self.domain_means[d] = batch_mean
                    self.domain_vars[d] = batch_var
                    self.stats_initialized[d] = True

    def caa_loss(self, margin: float = 1.0) -> torch.Tensor:
        """
        计算 CAA 损失 (类内拉近，类间推开)
        """
        if self.num_domains <= 1 or not self.anchors_initialized.any():
            return torch.tensor(0.0).to(self.anchors.device)

        # 仅选择已初始化的锚点进行计算
        valid_anchors = self.anchors[self.anchors_initialized]  # (K, D) K <= N*C
        if valid_anchors.shape[0] < 2:
            return torch.tensor(0.0).to(self.anchors.device)

        # 1. 类内损失 (Intra-class)
        # (N_domain, N_class, D) -> (N_class, N_domain, D)
        A_c = self.anchors.permute(1, 0, 2)
        # (N_class, N_domain, 1)
        valid_mask_c = self.anchors_initialized.permute(1, 0).unsqueeze(-1)

        loss_intra = 0.0
        total_classes_with_anchors = 0
        for c in range(self.num_classes):
            # (K_c, D)
            anchors_for_class_c = A_c[c][valid_mask_c[c].squeeze(-1)]
            if anchors_for_class_c.shape[0] > 1:
                mean_c = anchors_for_class_c.mean(dim=0, keepdim=True)  # (1, D)
                loss_intra += F.mse_loss(anchors_for_class_c, mean_c.expand_as(anchors_for_class_c))
                total_classes_with_anchors += 1

        loss_intra = loss_intra / total_classes_with_anchors if total_classes_with_anchors > 0 else 0.0

        # 2. 类间损失 (Inter-class)
        # 计算所有域的平均锚点 (N_class, D)
        A_mean = torch.zeros(self.num_classes, self.feat_dim).to(self.anchors.device)
        valid_class_mask = torch.zeros(self.num_classes).bool().to(self.anchors.device)
        for c in range(self.num_classes):
            anchors_for_class_c = A_c[c][valid_mask_c[c].squeeze(-1)]
            if anchors_for_class_c.shape[0] > 0:
                A_mean[c] = anchors_for_class_c.mean(dim=0)
                valid_class_mask[c] = True

        # 仅使用有效的平均锚点 (K_mean, D)
        A_mean_valid = A_mean[valid_class_mask]
        if A_mean_valid.shape[0] < 2:
            return loss_intra  # 无法计算类间损失

        # (K_mean, K_mean) 距离矩阵
        dist_matrix = torch.cdist(A_mean_valid, A_mean_valid, p=2)

        loss_inter = 0.0
        count = 0
        num_valid_classes = A_mean_valid.shape[0]
        for c_i in range(num_valid_classes):
            for c_j in range(num_valid_classes):
                if c_i == c_j: continue
                # 推开不同类别的锚点，至少相距 margin
                loss_inter += F.relu(margin - dist_matrix[c_i, c_j])
                count += 1
        loss_inter = loss_inter / count if count > 0 else 0.0

        return loss_intra + loss_inter

    def stats_align_loss(self, mu_tilde: torch.Tensor, d_true: torch.Tensor) -> torch.Tensor:
        """
        计算统计对齐损失 (L_stat)
        mu_tilde: (B, D)
        d_true: (B,)
        """
        if not self.enable_stats or self.num_domains <= 1 or not self.stats_initialized.all():
            return torch.tensor(0.0).to(mu_tilde.device)

        # 全局均值/方差
        global_mean = self.domain_means.mean(dim=0)  # (D,)
        global_var = self.domain_vars.mean(dim=0)  # (D,)

        loss_mean = 0.0
        loss_var = 0.0

        # 1. 强制 mu_tilde 匹配全局统计量
        # 同样需要处理 mu_tilde 和 d_true 的维度
        y_flat_shape_approx = mu_tilde.shape[0]  # 假设 mu_tilde 是 (B,D) or (B*T, D)

        if mu_tilde.shape[0] != d_true.shape[0]:  # mu(B*T, D), d(B,)
            d_flat = d_true.repeat_interleave(mu_tilde.shape[0] // d_true.shape[0])
        else:  # mu(B,D), d(B,)
            d_flat = d_true

        mu_flat = mu_tilde  # (B*T, D) or (B, D)

        loss_mu_mean = F.mse_loss(mu_flat.mean(dim=0), global_mean)
        loss_mu_var = F.mse_loss(mu_flat.var(dim=0, unbiased=False), global_var)

        # 2. 强制各域锚点匹配全局统计量
        for d in range(self.num_domains):
            if self.stats_initialized[d]:
                loss_mean += F.mse_loss(self.domain_means[d], global_mean)
                loss_var += F.mse_loss(self.domain_vars[d], global_var)

        loss_mean /= self.num_domains
        loss_var /= self.num_domains

        return loss_mean + loss_var + loss_mu_mean + loss_mu_var