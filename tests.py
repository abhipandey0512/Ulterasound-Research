"""
test.py

Module Testing Script
"""

import torch

from models.patch_embed import PatchEmbedding


def main():

    # Dummy video
    video = torch.randn(
        2,      # Batch Size
        16,     # Frames
        3,      # RGB Channels
        224,    # Height
        224,    # Width
    )

    # Patch Embedding Model
    model = PatchEmbedding()

    # Forward Pass
    output = model(video)

    print("=" * 50)
    print("Patch Embedding Test")
    print("=" * 50)
    print(f"Input Shape : {video.shape}")
    print(f"Output Shape: {output.shape}")
    print("=" * 50)


if __name__ == "__main__":
    main()