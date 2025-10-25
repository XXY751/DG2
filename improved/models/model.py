# improved_models/model.py
# (内部导入路径基本不变，因为它们相对于 improved_models 目录)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from types import SimpleNamespace

# --- Relative Imports within improved_models ---
try:
    from .ae import AE
    # 假设 DomainProjectionLDP 和 AnchorBankCAA 在 regulator.py 中
    # 确保 regulator.py 也在 improved_models 目录下
    from .regulator import DomainProjectionLDP, AnchorBankCAA
except ImportError as e:
    print(f"[ERROR] Relative import failed in improved_models/model.py: {e}")
    # Fallback (less robust)
    try:
        from improved.models.ae import AE
        from improved.models.regulator import DomainProjectionLDP, AnchorBankCAA
    except ImportError:
        raise ImportError("Could not import AE or regulator components within improved_models.")

# --- Absolute import for shared components ---
# 假设 transformer.py 在顶层 models 目录
try:
    from models.transformer import TransformerEncoder
except ImportError:
     print("[WARN] Could not import shared models.transformer. Using dummy class.")
     class TransformerEncoder(nn.Module):
         def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Identity()
         def forward(self, x): return self.dummy(x)

# --- Model class definition (保持不变) ---
class Model(nn.Module):
    def __init__(self, params: SimpleNamespace):
        super().__init__()
        self.params = params

        # --- 参数获取 (保持不变) ---
        self.num_domains = int(getattr(params, "num_domains", 1))
        self.num_classes = int(getattr(params, "num_of_classes", 5))
        proj_type = getattr(params, "projection_type", "diag")
        lowrank_r = int(getattr(params, "lowrank_rank", 32))
        proj_drop = float(getattr(params, "dropout", 0.0))
        enable_stats = bool(getattr(params, "enable_stats_alignment", True))
        anchor_m = float(getattr(params, "anchor_momentum", 0.9))
        feature_dim = 512

        # --- 模型组件 (保持不变) ---
        self.ae = AE(params)
        self.classifier = nn.Linear(feature_dim, self.num_classes)

        self.ldp = DomainProjectionLDP(dim=feature_dim,
                                       num_domains=self.num_domains,
                                       projection_type=proj_type,
                                       lowrank_rank=lowrank_r,
                                       dropout_p=proj_drop)

        self.anchors = AnchorBankCAA(num_classes=self.num_classes,
                                     feat_dim=feature_dim,
                                     num_domains=self.num_domains,
                                     ema_momentum=anchor_m,
                                     enable_stats_alignment=enable_stats)

    # --- forward, freeze_unified_projection, inference 方法 (保持不变) ---
    def forward(self, x, labels=None, domain_ids=None):
        recon, mu = self.ae(x) # mu shape: (B, T, D) = (bs, 20, 512)
        if mu.dim() == 3:
            mu_pooled = mu.mean(dim=1) # (B, D)
            mu_tilde, reg_A = self.ldp(mu_pooled, domain_ids=domain_ids, use_uni=False)
            logits_pooled = self.classifier(mu_tilde) # (B, C)
            logits = logits_pooled.unsqueeze(1).expand(-1, mu.size(1), -1) # -> (B, T, C)
        elif mu.dim() == 2:
            # self.logger.warning(f"Expected mu to have 3 dimensions (B, T, D), but got {mu.shape}. Applying LDP/Classifier directly.")
            mu_tilde, reg_A = self.ldp(mu, domain_ids=domain_ids, use_uni=False)
            logits = self.classifier(mu_tilde)
        else:
            raise ValueError(f"Unexpected shape for mu: {mu.shape}")
        return logits, recon, mu, mu_tilde, reg_A

    @torch.no_grad()
    def freeze_unified_projection(self, strategy="avg", weights=None):
        if hasattr(self.ldp, 'build_A_uni'):
            if weights is not None:
                device = next(self.ldp.parameters()).device
                weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
            self.ldp.build_A_uni(strategy=strategy, weights=weights)
        else:
             print("[WARN] freeze_unified_projection called, but LDP module does not have build_A_uni method.")

    @torch.no_grad()
    def inference(self, x, use_uni: bool = True):
        self.eval()
        mu = self.ae.encoder(x)
        if mu.dim() == 3:
             mu_pooled = mu.mean(dim=1)
             mu_tilde, _ = self.ldp(mu_pooled, domain_ids=None, use_uni=use_uni)
        elif mu.dim() == 2:
             mu_tilde, _ = self.ldp(mu, domain_ids=None, use_uni=use_uni)
        else:
             raise ValueError(f"Unexpected shape for mu in inference: {mu.shape}")
        logits = self.classifier(mu_tilde)
        return logits