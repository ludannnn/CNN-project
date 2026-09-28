"""
train_model_task2.py

training configuration that produced working weights for Task 2.

- foreground sprites are not augmented
- only background classes get augmented for dataset balance
- no colour jitter
- AdaptiveAvgPool2d(4) used

writes the get_model() function to model_snippet_task2.py which includes the string for learned weights
after training we can then copy the function into our agent implementation
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.utils.data

from grid_adventure.rendering import DEFAULT_ASSET_ROOT

#declare variables for configuration

ENTITY_CLASSES = [
    "floor", "wall", "exit", "gem", "key",
    "locked", "lava", "box", "coin",
    "boots", "shield", "ghost", "human", "opened",
]

BACKGROUND_CLASSES = {"floor", "wall", "lava"}

TILE_SIZE  = 32
EPOCHS     = 15
BATCH_SIZE = 64
LR         = 1e-3

#functions for the CNN

class TileCNN(nn.Module):
    """
    Input  : (batch, 4, 32, 32)
    Output : (batch, num_classes)

    Block 1: Conv(4->16, 3x3) -> ReLU -> MaxPool2d(2)          # 32 -> 16
    Block 2: Conv(16->32, 3x3) -> ReLU -> MaxPool2d(2)         # 16 -> 8
    Block 3: Conv(32->64, 3x3) -> ReLU -> AdaptiveAvgPool2d(4) # 8 -> 4x4
    Head   : Flatten -> Linear(1024->128) -> ReLU -> Linear(128->num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1), #output 16 feature maps
            nn.ReLU(),                                  # activation function
            nn.MaxPool2d(2),                            # 32x32 -> 16x16 (dimension)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), #outputs 32 feature maps
            nn.ReLU(),                                   #activation function
            nn.MaxPool2d(2),                             # 16x16 -> 8x8 (dimension)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), #outputs 64 feature maps
            nn.ReLU(),                                   # activation function
            nn.AdaptiveAvgPool2d(4),                     # 8x8 -> 4x4 (dimension)

            nn.Flatten(),                                # 64*4*4 = 1024 flatten so we can feed into FC layer
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),                 #map to num classes to identify what entity it is
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: #forward pass
        return self.net(x)


def load_sprites(folder: str, tile_size: int) -> list:
    #loads all PNGs from assets
    sprites = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".png"):
            continue
        img = (Image.open(os.path.join(folder, fname))
               .convert("RGBA")
               .resize((tile_size, tile_size)))
        sprites.append(img)
    return sprites

def compose_sprite(sprite: Image.Image, background: Image.Image) -> Image.Image:
    #composes a sprite onto a background
    bg = background.copy()
    bg.paste(sprite, (0, 0), sprite)
    return bg


def augment_background(img: Image.Image, rng: random.Random) -> Image.Image:
    # only background tiles get augmented, brightness contrast and random horizontal flip
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def to_array(img: Image.Image) -> np.ndarray:
    #converts an image to array
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1) #converts to the format (channels, height, width)


def build_dataset(asset_root: str, entity_classes: list, tile_size: int,
                  rng: random.Random) -> tuple:
    print("=" * 60)
    print("LOADING ASSETS")
    print("=" * 60) #for logging in the terminal

    floor_sprites = load_sprites(os.path.join(asset_root, "floor"), tile_size) #load floor sprites
    aug_multiplier = max(1, len(floor_sprites)) # counts number of floor sprites so we can apply that to other backgrounds
    print(f"  floor backgrounds loaded : {len(floor_sprites)}")
    print(f"  samples-per-sprite       : {aug_multiplier}x") # for logging in the terminal
    print()

    class_to_idx = {c: i for i, c in enumerate(entity_classes)}
    X_list, y_list = [], []

    for cls in entity_classes: #for each entity, load its sprite variants
        sprites = load_sprites(os.path.join(asset_root, cls), tile_size)

        samples_for_class = 0
        for sprite in sprites: #for all our sprites, 
            if cls in BACKGROUND_CLASSES:
                # repeated aug_multiplier times to balance the dataset
                # backgrounds are augmented for balance
                composites = [augment_background(sprite.copy(), rng)
                              for _ in range(aug_multiplier)]
            else:
                # foreground sprites are composited onto every floor background
                composites = [compose_sprite(sprite, bg) for bg in floor_sprites]

            for composite in composites:
                X_list.append(to_array(composite)) #convert the image to an array and store it in X list
                y_list.append(class_to_idx[cls]) #store the corresponding labels into y list
            samples_for_class += len(composites)

        print(f"  {cls:10s} — {samples_for_class} samples  (label {class_to_idx[cls]})")

    print()
    print(f"Total samples : {len(X_list)}  |  "
          f"Classes with data : {len(set(y_list))}/{len(entity_classes)}")
    print()

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32) # convert to tensor for training
    y_tensor = torch.tensor(y_list, dtype=torch.long) 
    return X_tensor, y_tensor, class_to_idx #returns a tuple of (images, corresponding labels, dictionary mapping)


def train(model: nn.Module, loader: torch.utils.data.DataLoader,
          epochs: int, lr: float) -> None:
    loss_fn   = nn.CrossEntropyLoss() #use cross entropy loss for our loss function
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    print("=" * 60)
    print("TRAINING")
    print("=" * 60) #for logging

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for batch_idx, (xb, yb) in enumerate(loader):
            #forward pass
            logits = model(xb)
            loss   = loss_fn(logits, yb)

            #backward pass
            optimiser.zero_grad()
            loss.backward() #get gradients
            optimiser.step() #weight update

            preds          = logits.argmax(dim=1)
            epoch_correct += (preds == yb).sum().item()
            epoch_loss    += loss.item() * len(yb)
            epoch_total   += len(yb)

            batch_acc = (preds == yb).float().mean().item() * 100
            print(f"  Epoch {epoch:2d} | Batch {batch_idx+1:3d}/{len(loader)} "
                  f"| loss={loss.item():.4f}  batch_acc={batch_acc:.1f}%")

        avg_loss = epoch_loss / epoch_total
        avg_acc  = epoch_correct / epoch_total * 100
        print(f"  {'':->54}")
        print(f"  Epoch {epoch:2d} summary  avg_loss={avg_loss:.4f}  train_acc={avg_acc:.1f}%")
        print()



def evaluate(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
             class_to_idx: dict) -> None:
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model.eval()
    with torch.no_grad():
        all_preds = model(X_tensor).argmax(dim=1)
    final_acc = (all_preds == y_tensor).float().mean().item() * 100

    print("=" * 60)
    print(f"FINAL TRAINING ACCURACY: {final_acc:.1f}%")
    print("=" * 60)
    print()
    print("Per-class accuracy:")
    for idx, cls in idx_to_class.items():
        mask = y_tensor == idx
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == idx).float().mean().item() * 100
        print(f"  {cls:10s}  {cls_acc:.1f}%")
    print()
    #print the accuracy for loggin in terminal



def export_snippet(model: nn.Module, tile_size: int, output_file: str) -> None:
    #write getmodel() to another file for easy copy and pasting
    import base64, io, zlib

    print("=" * 60)
    print("GENERATING get_model() SNIPPET")
    print("=" * 60)
    print()

    scripted   = torch.jit.script(model)
    buf        = io.BytesIO()
    torch.jit.save(scripted, buf)
    raw_bytes  = buf.getvalue()
    compressed = zlib.compress(raw_bytes, level=9)
    b64_str    = base64.b64encode(compressed).decode("ascii")

    print(f"  Raw model size    : {len(raw_bytes):,} bytes")
    print(f"  After zlib+base64 : {len(b64_str):,} chars")
    print()

    snippet = f'''\
def get_model():
    blob_b64 = "{b64_str}"
    raw = zlib.decompress(base64.b64decode(blob_b64))
    m = torch.jit.load(io.BytesIO(raw))
    m.eval()
    return m
'''

    with open(output_file, "w") as f:
        f.write("# Auto-generated by train_model_task2.py\n")
        f.write("# Paste get_model() into agent\n\n")
        f.write(snippet)

    print(f"Snippet written to: {output_file}")


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    print(f"Asset root : {DEFAULT_ASSET_ROOT}")
    print(f"Epochs     : {EPOCHS},  Batch size: {BATCH_SIZE},  LR: {LR}") #for logging
    print()

    X_tensor, y_tensor, class_to_idx = build_dataset(
        DEFAULT_ASSET_ROOT, ENTITY_CLASSES, TILE_SIZE, rng
    )

    model = TileCNN(num_classes=len(ENTITY_CLASSES))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}") #for logging
    print()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    train(model, loader, EPOCHS, LR)
    evaluate(model, X_tensor, y_tensor, class_to_idx)
    export_snippet(model, TILE_SIZE, output_file="model_snippet_task2.py")
