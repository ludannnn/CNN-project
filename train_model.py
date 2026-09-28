import os
import sys
import random
import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.utils.data

# Configuration

ENTITY_CLASSES = [
    "floor", "wall", "exit", "gem", "key",
    "locked", "lava", "box", "coin",
    "boots", "shield", "ghost", "human", "opened",
]

# Background-only classes
BACKGROUND_CLASSES = {"floor", "wall", "lava"}

TILE_SIZE     = 32    # all sprites are resized to this before training/inference
EPOCHS        = 35
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 1e-4  # L2 regularisation, penalises large weights to reduce overfitting
VAL_SPLIT     = 0.1   # amount of data held out for validation

try:
    from grid_adventure.rendering import DEFAULT_ASSET_ROOT
except ImportError:
    DEFAULT_ASSET_ROOT = os.path.join(os.path.dirname(__file__), "data", "assets")


# CNN architecture

class _TileCNN(nn.Module):
    """
    Input  : (batch, 4, 32, 32), 4-channel RGBA tile
    Output : (batch, num_classes), raw logits, one per entity type

    Architecture:
    Block 1: Conv(4->16, 3x3) -> ReLU -> MaxPool2d(2)          # 32 -> 16
    Block 2: Conv(16->32, 3x3) -> ReLU -> MaxPool2d(2)         # 16 -> 8
    Block 3: Conv(32->64, 3x3) -> ReLU -> AdaptiveAvgPool2d(4) # 8 -> 4x4
    Flatten -> Linear(1024->128) -> ReLU -> Linear(128->num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x32 -> 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16 -> 8x8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                      # 8x8 -> 4x4

            nn.Flatten(),                                 # 64*4*4 = 1024
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def load_sprites(folder: str, tile_size: int) -> list:
    #loads all PNGs from assets 
    sprites = []
    if not os.path.isdir(folder):
        return sprites
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".png"):
            continue
        try:
            img = (Image.open(os.path.join(folder, fname))
                   .convert("RGBA")
                   .resize((tile_size, tile_size)))
            sprites.append(img)
        except Exception as e:
            print(f"  [WARN] {folder}/{fname}: {e}")
    return sprites


def compose_sprite(sprite: Image.Image, background: Image.Image) -> Image.Image:
    #composes a sprite onto a background
    bg = background.copy()
    bg.paste(sprite, (0, 0), sprite)
    return bg


def augment_foreground(img: Image.Image, rng: random.Random) -> Image.Image:
    # foreground entities (gem, key, agent etc.)brightness + contrast + flip only.
    # no colour jitter
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def augment_background(img: Image.Image, rng: random.Random) -> Image.Image:
    # background tiles (floor, wall, lava)full augmentation including colour jitter.
    # these tiles need maximum variety for dataset balance
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def to_array(img: Image.Image) -> np.ndarray:
    #converts an image to array
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 4), range [0, 1]
    return arr.transpose(2, 0, 1)                    # (4, H, W)


# Dataset builder

def build_dataset(asset_root: str, entity_classes: list, tile_size: int,
                  rng: random.Random) -> tuple:
    """
    #builds training dataset
    #compose sprites onto every single possible background
    #background sprites (floors and walls) are augmented to balance the dataset

    Returns (X_tensor, y_tensor, class_to_idx).
    """
    print("=" * 60)
    print("LOADING ASSETS")
    print("=" * 60)

    floor_sprites = load_sprites(os.path.join(asset_root, "floor"), tile_size) #load floor sprites
    aug_multiplier = max(1, len(floor_sprites)) # counts number of floor sprites so we can apply that to other backgrounds
    print(f"  floor backgrounds loaded : {len(floor_sprites)}")
    print(f"  samples-per-sprite       : {aug_multiplier}x")
    print()

    class_to_idx = {c: i for i, c in enumerate(entity_classes)}
    X_list, y_list = [], []

    for cls in entity_classes: # for each entity, load its sprite variants
        sprites = load_sprites(os.path.join(asset_root, cls), tile_size)
        if not sprites:
            print(f"  [SKIP] {cls:10s} — no sprites found")
            continue

        samples_for_class = 0
        for sprite in sprites: #for all our sprites, 
            if cls in BACKGROUND_CLASSES or not floor_sprites:
                # background tile: full augmentation (including colour jitter)
                # repeated aug_multiplier times to balance the dataset
                composites = [augment_background(sprite.copy(), rng)
                              for _ in range(aug_multiplier)]
            else:
                # foreground entity: composite onto every floor background,
                # then apply foreground augmentation (no colour jitter)
                composites = [augment_foreground(compose_sprite(sprite, bg), rng)
                              for bg in floor_sprites]

            for composite in composites:
                X_list.append(to_array(composite)) #convert the image to an array and store it in X list
                y_list.append(class_to_idx[cls]) #store the corresponding labels into y list
            samples_for_class += len(composites)

        print(f"  {cls:10s} — {samples_for_class} samples  (label {class_to_idx[cls]})")

    print()
    print(f"Total samples : {len(X_list)}  |  "
          f"Classes with data : {len(set(y_list))}/{len(entity_classes)}")
    print()

    if not X_list:
        print("ERROR: No training data found. Check DEFAULT_ASSET_ROOT.")
        sys.exit(1)

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32) # convert to tensor for training
    y_tensor = torch.tensor(y_list, dtype=torch.long) 
    return X_tensor, y_tensor, class_to_idx #returns a tuple of (images, corresponding labels, dictionary mapping)


# Training

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          epochs: int, lr: float, weight_decay: float) -> None:
    """Train the model in-place, printing per-batch and per-epoch stats.

    Evaluates on the val set after every epoch and keeps the weights that
    achieved the best val accuracy (best checkpoint).
    """
    loss_fn   = nn.CrossEntropyLoss()
    # weight_decay adds L2 penalty to the loss, discouraging large weights
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # cosine annealing gradually reduces LR to near 0 by the final epoch,
    # letting the model make large updates early and fine-tune at the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_val_acc   = 0.0
    best_weights   = None   # deepcopy of state_dict at the best val epoch

    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # training pass
        model.train()
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            # Forward pass
            logits = model(xb)
            loss   = loss_fn(logits, yb)

            # Backward pass
            optimiser.zero_grad()   # clear gradients from previous step
            loss.backward()         # compute gradients via backprop
            optimiser.step()        # update weights with Adam

            # Accumulate stats
            preds          = logits.argmax(dim=1)
            epoch_correct += (preds == yb).sum().item()
            epoch_loss    += loss.item() * len(yb)
            epoch_total   += len(yb)

            batch_acc = (preds == yb).float().mean().item() * 100
            print(f"  Epoch {epoch:2d} | Batch {batch_idx+1:3d}/{len(train_loader)} "
                  f"| loss={loss.item():.4f}  batch_acc={batch_acc:.1f}%")

        scheduler.step()  # update LR at end of each epoch

        train_acc = epoch_correct / epoch_total * 100

        # validation pass
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds        = model(xb).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total   += len(yb)
        val_acc = val_correct / val_total * 100

        # checkpoint if best val accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import copy
            best_weights = copy.deepcopy(model.state_dict())
            checkpoint_marker = "  *** best checkpoint ***"
        else:
            checkpoint_marker = ""

        avg_loss = epoch_loss / epoch_total
        print(f"  {'':->54}")
        print(f"  Epoch {epoch:2d}  loss={avg_loss:.4f}  "
              f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}{checkpoint_marker}")
        print()

    # restore the best weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"Restored best checkpoint  (val_acc={best_val_acc:.1f}%)")


# Evaluation

def evaluate(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
             class_to_idx: dict, label: str = "Val") -> None:
    """Print overall and per-class accuracy on the given set."""
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model.eval()
    with torch.no_grad():
        all_preds = model(X_tensor).argmax(dim=1)
    final_acc = (all_preds == y_tensor).float().mean().item() * 100

    print("=" * 60)
    print(f"FINAL {label.upper()} ACCURACY: {final_acc:.1f}%")
    print("=" * 60)
    print()
    print(f"Per-class accuracy ({label}):")
    for idx, cls in idx_to_class.items():
        mask = y_tensor == idx
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == idx).float().mean().item() * 100
        print(f"  {cls:10s}  {cls_acc:.1f}%")
    print()


# Snippet export

def export_snippet(model: nn.Module, tile_size: int, output_file: str) -> None:
    """TorchScript the model, compress with zlib, encode as base64, and write
    a self-contained get_model() function to output_file."""
    import base64, io, zlib

    print("=" * 60)
    print("GENERATING get_model() SNIPPET")
    print("=" * 60)
    print()

    # TorchScript the model so get_model() has no dependency on _TileCNN
    scripted = torch.jit.script(model)

    # save the scripted model to an in-memory buffer
    buf = io.BytesIO()
    torch.jit.save(scripted, buf)
    raw_bytes = buf.getvalue()

    # compress with zlib then encode as base64 string
    compressed = zlib.compress(raw_bytes, level=9)
    b64_str = base64.b64encode(compressed).decode("ascii")

    print(f"  Raw model size       : {len(raw_bytes):,} bytes")
    print(f"  After zlib+base64    : {len(b64_str):,} chars")
    print()

    # write the snippet
    snippet = f'''\
def get_model():
    blob_b64 = "{b64_str}"
    raw = zlib.decompress(base64.b64decode(blob_b64))
    m = torch.jit.load(io.BytesIO(raw), map_location="cpu")
    m.eval()
    return m
'''

    with open(output_file, "w") as f:
        f.write("# Auto-generated by train_model.py — do not edit by hand.\n")
        f.write("# Paste get_model() into agent_with_image_parser.py at module level.\n\n")
        f.write(snippet)

    print(f"Snippet written to: {output_file}")
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. Open {output_file} and copy the get_model() function into
   agent_with_image_parser.py at module level (outside the Agent class).

2. In Agent.__init__, replace:
       self._cnn, self._idx_to_class = self._build_and_train_cnn()
   with:
       self._cnn = get_model()
       self._idx_to_class = {{i: c for i, c in enumerate(self.ENTITY_CLASSES)}}

3. You can delete the _build_and_train_cnn method — it's no longer needed.
""")

if __name__ == "__main__":
    random.seed(42)
    _rng = random.Random(42)

    print(f"Asset root   : {DEFAULT_ASSET_ROOT}")
    print(f"Classes      : {ENTITY_CLASSES}")
    print(f"Tile size    : {TILE_SIZE}x{TILE_SIZE} (RGBA = 4 channels)")
    print(f"Epochs       : {EPOCHS},  Batch size: {BATCH_SIZE},  LR: {LR}")
    print(f"Weight decay : {WEIGHT_DECAY},  Val split: {VAL_SPLIT}")
    print()

    X_tensor, y_tensor, class_to_idx = build_dataset(
        DEFAULT_ASSET_ROOT, ENTITY_CLASSES, TILE_SIZE, _rng
    )

    # 90/10 train/val split, val set is held out and never trained on
    full_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    n_val   = int(VAL_SPLIT * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train samples : {n_train}  |  Val samples : {n_val}")
    print()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = _TileCNN(num_classes=len(ENTITY_CLASSES))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    train(model, train_loader, val_loader, EPOCHS, LR, WEIGHT_DECAY)

    # evaluate best checkpoint on the held-out val set
    val_X, val_y = zip(*[(x, y) for x, y in val_ds])
    evaluate(model, torch.stack(val_X), torch.tensor(val_y), class_to_idx, label="Val")

    export_snippet(model, TILE_SIZE, output_file="model_snippet.py")
