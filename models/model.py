# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =========================
# 1) 可学习域投影 LDP
# =========================
class _DiagProj(nn.Module):
    """对角投影：A = diag(a)。参数量最小，稳定好用。"""

    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(dim))  # 初始化为0 => 初始近似恒等（残差里是 f + A f）

    def forward(self, x):
        return x * self.a  # (B, D)

    def orth_reg(self):
        # 对角阵的 A^T A = diag(a^2)，与 I 的差距：sum((a^2-1)^2)；但残差里已包含恒等，正则放轻
        a2 = self.a ** 2
        return (a2 - 1.0).pow(2).mean()

    def get_matrix(self):
        # 仅用于构造 A_uni；避免大矩阵，返回向量形式
        return self.a


class _LowRankProj(nn.Module):
    """低秩投影：A = U diag(s) V^T，秩=r << D。提供正交约束稳定谱。"""

    def __init__(self, dim, rank=32):
        super().__init__()
        self.dim = dim
        self.rank = rank
        # U, V 使用正交初始化，s 初始化为0，保证初始近似恒等（残差里）
        self.U = nn.Parameter(torch.empty(dim, rank))
        self.V = nn.Parameter(torch.empty(dim, rank))
        self.s = nn.Parameter(torch.zeros(rank))
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

    def forward(self, x):
        # x: (B, D) -> (B, D)  计算 x @ (V diag(s) U^T)^T = x @ (U diag(s) V^T)
        # 为了数值稳定，采用逐步乘法
        z = x @ self.V  # (B, r)
        z = z * self.s  # (B, r)
        z = z @ self.U.t()  # (B, D)
        return z

    def orth_reg(self):
        # 近似正交：U^T U ≈ I, V^T V ≈ I；再加核范数近似（||s||_1）鼓励低秩有效性
        I_u = torch.eye(self.rank, device=self.U.device, dtype=self.U.dtype)
        I_v = torch.eye(self.rank, device=self.V.device, dtype=self.V.dtype)
        reg_orth = (self.U.t() @ self.U - I_u).pow(2).mean() + (self.V.t() @ self.V - I_v).pow(2).mean()
        reg_nuc = self.s.abs().mean()
        return reg_orth + 0.1 * reg_nuc

    def get_matrix(self):
        # 返回稀疏表达 (U, s, V) 以便合成统一矩阵；避免创建 D×D 巨阵
        return (self.U, self.s, self.V)


class DomainProjectionLDP(nn.Module):
    """
    为每个域 d 学一个轻量投影 A_d，残差式输出：f_tilde = f + A_d f
    支持 projection_type=['diag','lowrank']。
    """

    def __init__(self, dim: int, num_domains: int, projection_type: str = "diag", lowrank_rank: int = 32,
                 dropout_p: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_domains = num_domains
        self.projection_type = projection_type
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.projs = nn.ModuleList()
        for _ in range(num_domains):
            if projection_type == "diag":
                self.projs.append(_DiagProj(dim))
            elif projection_type == "lowrank":
                self.projs.append(_LowRankProj(dim, rank=lowrank_rank))
            else:
                raise ValueError(f"Unknown projection_type={projection_type}")

        # 统一投影（推理用）：训练后可调用 self.build_A_uni(strategy='avg'/'weighted')
        self._frozen_uni = None  # 存储为可调用的 lambda: x -> A_uni x

    @torch.no_grad()
    def build_A_uni(self, strategy: str = "avg", weights: torch.Tensor = None):
        """
        将多个 A_d 融合为统一 A_uni：
        - diag: 平均 a 向量
        - lowrank: 简单平均重构矩阵（近似）。若提供 weights，做加权平均
        """
        if self.projection_type == "diag":
            a_stack = torch.stack([m.get_matrix() for m in self.projs], dim=0)  # (D, num_domains) or (num_domains, D)
            if weights is None:
                a_bar = a_stack.mean(dim=0)
            else:
                w = weights / (weights.sum() + 1e-12)
                a_bar = (a_stack * w.view(-1, 1)).sum(dim=0)

            # 构造闭包
            def _apply(x):
                return x * a_bar

            self._frozen_uni = _apply

        else:  # lowrank
            # 直接显式合成 A_d 并平均：A = U diag(s) V^T，注意内存
            # 为避免 D×D 巨阵，采用分块乘：x @ A = ((x @ V) * s) @ U^T
            Us, ss, Vs, ws = [], [], [], []
            for i, m in enumerate(self.projs):
                U, s, V = m.get_matrix()
                Us.append(U.detach())
                ss.append(s.detach())
                Vs.append(V.detach())
                ws.append(1.0 if weights is None else float(weights[i].item()))
            ws = torch.tensor(ws, device=Us[0].device, dtype=Us[0].dtype)
            ws = ws / (ws.sum() + 1e-12)

            # 将多个低秩项拼接成更高秩的统一近似（拼 rank，然后按权重缩放 s）
            U_cat = torch.cat(Us, dim=1)  # (D, sum r)
            V_cat = torch.cat(Vs, dim=1)  # (D, sum r)
            s_cat = torch.cat([w * s for w, s in zip(ws, ss)])  # (sum r,)

            # 可再做一次小型SVD/QR精简秩；这里直接使用拼接近似
            def _apply(x):
                z = x @ V_cat  # (B, Rtot)
                z = z * s_cat  # (B, Rtot)
                z = z @ U_cat.t()  # (B, D)
                return z

            self._frozen_uni = _apply

    def forward(self, feats: torch.Tensor, domain_ids: torch.Tensor = None, use_uni: bool = False):
        """
        feats: (B, D)
        domain_ids: (B,) in [0, num_domains-1]
        use_uni: True 则使用统一投影（推理零目标域样本）
        return: feats_tilde, reg_loss
        """
        if use_uni:
            if self._frozen_uni is None:
                # 若未构建，默认平均
                self.build_A_uni(strategy="avg")
            Af = self._frozen_uni(feats)
            return self.dropout(feats + Af), feats.new_tensor(0.0)

        assert domain_ids is not None, "Training时需要 domain_ids 才能选择对应 A_d"
        reg = 0.0
        out = torch.empty_like(feats)
        # 按域分组应用对应投影（减少 Python 循环可后续优化）
        for d in range(self.num_domains):
            mask = (domain_ids == d)
            if mask.any():
                f_d = feats[mask]
                A_d = self.projs[d]
                out[mask] = f_d + A_d(f_d)
                reg = reg + A_d.orth_reg()
        reg = reg / max(1, self.num_domains)
        if not torch.is_tensor(reg):
            reg = feats.new_tensor(float(reg))  # 转为标量张量
        # --- 修复：将标量升维，避免 DataParallel 警告 ---
        return self.dropout(out), reg.unsqueeze(0)


# =========================
# 2) 公共锚点一致性 + 轻量统计对齐
# =========================
class AnchorBankCAA(nn.Module):
    """
    维护跨域/跨类锚点（EMA），并计算：
    - CAA: sum_d,c || mu_c^d - mu_c_bar ||^2
    - 统计对齐(可选): 域间均值/协方差对齐的轻量版本
    """

    def __init__(self, num_classes: int, feat_dim: int, num_domains: int,
                 ema_momentum: float = 0.9, enable_stats_alignment: bool = True):
        super().__init__()
        self.C = num_classes
        self.D = feat_dim
        self.M = num_domains
        self.m = ema_momentum
        self.enable_stats = enable_stats_alignment

        # 按域/类维护锚点；使用 buffer 方便保存/加载
        self.register_buffer("anchors_dc", torch.zeros(self.M, self.C, self.D))  # mu_c^d
        self.register_buffer("counts_dc", torch.zeros(self.M, self.C))  # 计数用于冷启动
        self.register_buffer("anchor_global", torch.zeros(self.C, self.D))  # \bar{mu}_c
        self.register_buffer("counts_global", torch.zeros(self.C))

        # 全局统计（可选）
        if self.enable_stats:
            self.register_buffer("global_mean", torch.zeros(self.D))
            self.register_buffer("global_cov", torch.eye(self.D))

    @torch.no_grad()
    def _update_ema(self, old, new, mom):
        return mom * old + (1.0 - mom) * new

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor, domain_ids: torch.Tensor):
        """
        feats:      (B,D) or (B,T,D)
        labels:     (B,)  or (B,T)
        domain_ids: (B,)  or (B,T)  —— 若为 (B,) 则在 T 维广播
        """
        # ---- 统一展平 ----
        if feats.dim() == 3:  # (B,T,D)
            B, T, D = feats.shape
            feats_f = feats.reshape(B * T, D)
            labels_f = labels.reshape(B * T)
            # 广播域ID：若是 (B,) -> (B*T,)
            if domain_ids.dim() == 1:
                domains_f = domain_ids.repeat_interleave(T)
            elif domain_ids.dim() == 2:
                domains_f = domain_ids.reshape(B * T)
            else:
                raise ValueError(f"domain_ids dim must be 1 or 2, got {domain_ids.dim()}")
        else:  # (B,D)
            D = feats.shape[-1]
            feats_f = feats
            labels_f = labels
            if domain_ids.dim() != 1:
                raise ValueError(f"When feats is 2D, domain_ids must be (B,), got shape {tuple(domain_ids.shape)}")
            domains_f = domain_ids

        # ---- 按域/类更新 EMA 锚点 ----
        for d in range(self.M):
            mask_d = (domains_f == d)
            if not mask_d.any():
                continue
            f_d = feats_f[mask_d]
            y_d = labels_f[mask_d]
            for c in range(self.C):
                mask_dc = (y_d == c)
                if not mask_dc.any():
                    continue
                f_dc = f_d[mask_dc]  # (K,D)
                mu_dc = f_dc.mean(dim=0)  # (D,)
                self.anchors_dc[d, c] = self._update_ema(self.anchors_dc[d, c], mu_dc, self.m)
                self.counts_dc[d, c] += float(mask_dc.sum().item())
                self.anchor_global[c] = self._update_ema(self.anchor_global[c], mu_dc, self.m)
                self.counts_global[c] += float(mask_dc.sum().item())

        # ---- 轻量全局统计 ----
        if self.enable_stats:
            mu = feats_f.mean(dim=0)
            xc = feats_f - mu
            cov = (xc.t() @ xc) / (feats_f.shape[0] + 1e-6)
            self.global_mean = self._update_ema(self.global_mean, mu, self.m)
            self.global_cov = self._update_ema(self.global_cov, cov, self.m)

    def caa_loss(self):
        """公共锚点一致性：∑_{d,c∈batch出现} || mu_c^d - mu_c_bar ||^2"""
        loss = 0.0
        valid = 0
        for d in range(self.M):
            for c in range(self.C):
                if self.counts_dc[d, c] > 0 and self.counts_global[c] > 0:
                    diff = self.anchors_dc[d, c] - self.anchor_global[c]
                    loss += (diff.pow(2).mean())
                    valid += 1
        if valid == 0:
            return torch.tensor(0.0, device=self.anchor_global.device)
        return loss / valid

    def stats_align_loss(self, feats: torch.Tensor, domain_ids: torch.Tensor):
        """
        feats:      (B,D) or (B,T,D)
        domain_ids: (B,)  or (B,T)
        """
        if not self.enable_stats:
            return feats.new_tensor(0.0)

        if feats.dim() == 3:
            B, T, D = feats.shape
            feats_f = feats.reshape(B * T, D)
            if domain_ids.dim() == 1:
                domains_f = domain_ids.repeat_interleave(T)
            elif domain_ids.dim() == 2:
                domains_f = domain_ids.reshape(B * T)
            else:
                raise ValueError(f"domain_ids dim must be 1 or 2, got {domain_ids.dim()}")
        else:
            feats_f = feats
            if domain_ids.dim() != 1:
                raise ValueError(f"When feats is 2D, domain_ids must be (B,), got shape {tuple(domain_ids.shape)}")
            domains_f = domain_ids

        loss, valid = 0.0, 0
        for d in range(self.M):
            mask = (domains_f == d)
            if not mask.any():
                continue
            f_d = feats_f[mask]
            mu_d = f_d.mean(dim=0)
            xc = f_d - mu_d
            cov_d = (xc.t() @ xc) / (f_d.shape[0] + 1e-6)
            loss += (mu_d - self.global_mean).pow(2).mean()
            loss += (cov_d - self.global_cov).pow(2).mean()
            valid += 1

        if valid == 0:
            return feats.new_tensor(0.0)
        return loss / valid


# =========================
# 3) 嵌回你的 Model
# =========================
from models.encoder import Encoder  # 你现有工程里的
from models.ae import AE  # 你现有工程里的


class Model(nn.Module):
    """
    - AE 得到 mu(512)
    - LDP: mu -> mu_tilde
    - classifier: mu_tilde -> logits
    """
    def __init__(self, params):
        super().__init__()
        self.params = params

        # —— 健壮默认值，避免外部没传参时报 AttributeError ——
        self.num_domains = int(getattr(params, "num_domains", 1))
        self.num_classes = int(getattr(params, "num_of_classes", 5))
        proj_type = getattr(params, "projection_type", "diag")
        lowrank_r = int(getattr(params, "lowrank_rank", 32))
        proj_drop = float(getattr(params, "projection_dropout", 0.0))
        enable_stats = bool(getattr(params, "enable_stats_alignment", True))
        anchor_m = float(getattr(params, "anchor_momentum", 0.9))

        self.ae = AE(params)                           # 约定 ae(x) -> recon, mu(512)
        self.classifier = nn.Linear(512, self.num_classes)

        self.ldp = DomainProjectionLDP(dim=512,
                                       num_domains=self.num_domains,
                                       projection_type=proj_type,
                                       lowrank_rank=lowrank_r,
                                       dropout_p=proj_drop)

        self.anchors = AnchorBankCAA(num_classes=self.num_classes,
                                     feat_dim=512,
                                     num_domains=self.num_domains,
                                     ema_momentum=anchor_m,
                                     enable_stats_alignment=enable_stats)
    def forward(self, x, labels=None, domain_ids=None):
        """
        修改后的 forward 方法：
        - 不再计算 loss_dict。
        - 返回计算 loss 所需的所有中间张量。
        """
        recon, mu = self.ae(x)
        mu_tilde, reg_A = self.ldp(mu, domain_ids=domain_ids, use_uni=False)
        logits = self.classifier(mu_tilde)

        # 只返回张量，不再返回 loss_dict
        return logits, recon, mu, mu_tilde, reg_A

    @torch.no_grad()
    def freeze_unified_projection(self, strategy="avg", weights=None):
        """
        训练完后调用一次：构建统一投影 A_uni （零目标域样本推理）
        - strategy: 'avg'（默认），weights（可传入各源域验证贡献）
        """
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.classifier.weight.device)
        self.ldp.build_A_uni(strategy=strategy, weights=weights)

    @torch.no_grad()
    def inference(self, x, use_uni: bool = True):
        """
        纯 DG 推理（零目标域样本）：默认使用 A_uni。
        若还未 freeze_unified_projection，会在第一次推理时自动按平均构建。
        """
        mu = self.ae.encoder(x)  # (B, 512)
        mu_tilde, _ = self.ldp(mu, domain_ids=None, use_uni=use_uni)
        return self.classifier(mu_tilde)