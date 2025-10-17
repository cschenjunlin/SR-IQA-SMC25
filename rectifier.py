import torch
import torch.nn as nn


class Rectifier(nn.Module):
    def __init__(self, embed_dim=768, num_outputs=2, drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        result = self.mlp(x)
        result = result.mean(dim=1)

        return result[:, 0], result[:, 1]


class Rectifier2(nn.Module):
    def __init__(self, embed_dim=768, num_outputs=2, drop=0.1):
        super().__init__()
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        bsz = x.shape[0]

        # per_patch_score = self.fc_score(x)
        # per_patch_score = per_patch_score.reshape(bsz, -1)
        # per_patch_weight = self.fc_weight(x)
        # per_patch_weight = per_patch_weight.reshape(bsz, -1)
        #
        # score = (per_patch_weight * per_patch_score).sum(dim=-1) / (per_patch_weight.sum(dim=-1) + 1e-8)
        # return score[:, 0], score[:, 1]

        per_patch_score = self.fc_score(x)
        per_patch_weight = self.fc_weight(x)
        # per_patch_score = per_patch_score.reshape(bsz, -1)
        # per_patch_weight = per_patch_weight.reshape(bsz, -1)

        score = (per_patch_weight * per_patch_score).sum(dim=1) / (per_patch_weight.sum(dim=1) + 1e-8)
        return score[:, 0], score[:, 1]
