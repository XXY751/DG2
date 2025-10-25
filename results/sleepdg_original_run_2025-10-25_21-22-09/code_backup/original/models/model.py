# original/models/model.py
import torch
import torch.nn as nn
from types import SimpleNamespace

# --- CORRECTED Relative Import ---
try:
    from .ae import AE # Assumes ae.py is in the same directory
except ImportError as e:
    print(f"[ERROR] Relative import failed in original/models/model.py: {e}")
    try:
        from original.models.ae import AE
    except ImportError:
         raise ImportError("Could not import AE component. Check file structure.")


class Model(nn.Module):
    def __init__(self, params: SimpleNamespace):
        super(Model, self).__init__()
        self.params = params
        self.ae = AE(params)
        num_classes = getattr(params, 'num_of_classes', 5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        recon, mu = self.ae(x) # Expect mu to be (B, T, D) = (bs, 20, 512)

        if mu.dim() == 3:
             pred = self.classifier(mu) # Output (B, T, C)
        elif mu.dim() == 2:
             # This case might indicate an issue upstream or requires different handling
             print(f"[WARN] Original model expected mu with 3 dims, got {mu.shape}. Applying classifier.")
             pred = self.classifier(mu) # Output (B, C)
        else:
             raise ValueError(f"Unexpected shape for mu: {mu.shape}")

        return pred, recon, mu

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            mu = self.ae.encoder(x)
            if mu.dim() == 3:
                pred = self.classifier(mu) # (B, T, C)
                # For inference, maybe return the mean prediction over time?
                # pred = pred.mean(dim=1) # -> (B, C)
                # Or keep as is, depending on evaluation needs
            elif mu.dim() == 2:
                pred = self.classifier(mu) # (B, C)
            else:
                 raise ValueError(f"Unexpected shape for mu in inference: {mu.shape}")
        return pred