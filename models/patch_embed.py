"""
patch_embed.py

Patch Embedding Module for VideoMAE
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Convert video frames into patch embeddings.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Input:
            x : (B, T, C, H, W)

        Output:
            (B, T, N, D)

        B = Batch
        T = Frames
        N = Number of patches
        D = Embedding dimension
        """

        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        x = self.projection(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            T,
            self.num_patches,
            -1,
        )

        return x