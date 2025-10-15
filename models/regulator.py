# models/regulator.py

import torch
import torch.nn as nn


class ActiveDomainRegulator(nn.Module):
    """
    主动域调节器模块 (Active Domain Regulator)
    包含一个公共锚点和针对每个域的投影矩阵。
    """

    def __init__(self, feature_dim=512, num_source_domains=4):
        """
        初始化调节器。
        Args:
            feature_dim (int): 特征向量的维度。
            num_source_domains (int): 源域的数量。SleepDG中有4个源域。
        """
        super(ActiveDomainRegulator, self).__init__()
        self.num_source_domains = num_source_domains
        self.feature_dim = feature_dim

        # 1. 定义可学习的“公共锚点”
        # 这个锚点是所有域对齐的目标
        self.public_anchor = nn.Parameter(torch.randn(1, 20, feature_dim))

        # 2. 为每个源域定义一个专属的可学习“翻译”矩阵 A_i
        # 使用 nn.ModuleList 来存储这些矩阵
        self.projection_matrices = nn.ModuleList([
            self.create_projection_matrix() for _ in range(num_source_domains)
        ])

    def create_projection_matrix(self):
        """创建一个从特征空间到自身的线性投影矩阵"""
        return nn.Linear(self.feature_dim, self.feature_dim, bias=False)

    def forward(self, features, domain_ids):
        """
        前向传播，将不同域的特征投影到公共锚点空间。
        Args:
            features (Tensor): 从Encoder输出的特征，形状为 (batch_size, 20, feature_dim)。
            domain_ids (Tensor): 每个样本对应的域ID，形状为 (batch_size,)。

        Returns:
            projected_features (Tensor): 经过投影（翻译）后的特征。
            anchor_alignment_loss (Tensor): 锚点对齐损失。
        """
        batch_size = features.size(0)
        projected_features = torch.zeros_like(features)

        # 使用L2损失（MSELoss）作为锚点对齐损失
        loss_fn = nn.MSELoss()
        anchor_alignment_loss = 0.0

        # 对批次中的每个域进行处理
        for i in range(self.num_source_domains):
            # 找到当前批次中属于域 i 的样本
            domain_mask = (domain_ids == i)
            if domain_mask.any():
                domain_features = features[domain_mask]

                # 使用对应的投影矩阵 A_i 进行“翻译”
                projected = self.projection_matrices[i](domain_features)
                projected_features[domain_mask] = projected

                # 计算该域翻译后的特征与公共锚点之间的对齐损失
                # 我们将锚点扩展到与当前域样本数量相同的批次大小
                anchor_expanded = self.public_anchor.expand(projected.size(0), -1, -1)
                anchor_alignment_loss += loss_fn(projected, anchor_expanded)

        # 对所有域的损失求平均
        if self.num_source_domains > 0:
            anchor_alignment_loss /= self.num_source_domains

        return projected_features, anchor_alignment_loss