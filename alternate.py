import os
import math
import glob
import random
import shutil
import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO

# ==================================================
# SETTINGS
# ==================================================

ROOT = r"D:\data\COMPUTER VISON\M6_Road_Damage_Detection"
DATA = os.path.join(ROOT, "datasets", "RDD2022", "India")

OUT = os.path.join(ROOT, "SIMPLE_DIFFUSION_PROJECT")
os.makedirs(OUT, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "CPU")

IMG_SIZE = 416
CROP_SIZE = 64
MIN_CROP_PIXELS = 24        # skip tiny potholes (they become blurry when upscaled)

TRAIN_LIMIT = 300
VAL_LIMIT = 60

YOLO_EPOCHS = 50            # was 5, too few for a meaningful comparison
DIFF_EPOCHS = 200           # was 3, diffusion needs thousands of steps
TIMESTEPS = 300

N_GENERATE = 200            # candidates to generate
N_SELECT = 50               # how many to keep after scoring

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==================================================
# STEP 1: PREPARE POTHOLE-ONLY DATASET
# ==================================================

print("\nSTEP 1: Preparing pothole dataset")

with open(os.path.join(DATA, "data.yaml"), "r") as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, dict):
    pothole_id = next(
        int(k) for k, v in names.items()
        if str(v).upper() == "D40"
    )
else:
    pothole_id = next(
        i for i, v in enumerate(names)
        if str(v).upper() == "D40"
    )

print("Pothole class ID:", pothole_id)

WORK = os.path.join(OUT, "dataset")

# Start clean so re-running doesn't pile up old files.
if os.path.exists(WORK):
    shutil.rmtree(WORK)

TRAIN_IMG = os.path.join(WORK, "train", "images")
TRAIN_LBL = os.path.join(WORK, "train", "labels")
VAL_IMG = os.path.join(WORK, "valid", "images")
VAL_LBL = os.path.join(WORK, "valid", "labels")

for folder in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    os.makedirs(folder, exist_ok=True)


def prepare_split(split, limit, image_out, label_out):

    source_images = os.path.join(DATA, split, "images")
    source_labels = os.path.join(DATA, split, "labels")

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(source_images, ext)))

    random.shuffle(images)
    count = 0

    for image_path in images:

        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(source_labels, stem + ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Keep only D40 pothole annotations, remapped to class 0.
        potholes = []

        for line in lines:
            values = line.strip().split()

            if len(values) >= 5 and int(float(values[0])) == pothole_id:
                potholes.append("0 " + " ".join(values[1:5]) + "\n")

        if not potholes:
            continue

        shutil.copy2(
            image_path,
            os.path.join(image_out, os.path.basename(image_path))
        )

        with open(os.path.join(label_out, stem + ".txt"), "w") as f:
            f.writelines(potholes)

        count += 1

        if count >= limit:
            break

    return count


train_count = prepare_split("train", TRAIN_LIMIT, TRAIN_IMG, TRAIN_LBL)
val_count = prepare_split("valid", VAL_LIMIT, VAL_IMG, VAL_LBL)

if train_count == 0 or val_count == 0:
    raise RuntimeError("No pothole images found. Check your D40 labels.")

yaml_path = os.path.join(WORK, "data.yaml")

with open(yaml_path, "w") as f:
    yaml.dump({
        "path": WORK,
        "train": "train/images",
        "val": "valid/images",
        "names": {0: "Pothole"}
    }, f)

print("Training images:", train_count)
print("Validation images:", val_count)


# ==================================================
# STEP 2: TRAIN BASELINE YOLO
# ==================================================

print("\nSTEP 2: Training baseline YOLO11n")

baseline = YOLO("yolo11n.pt")

baseline.train(
    data=yaml_path,
    epochs=YOLO_EPOCHS,
    imgsz=IMG_SIZE,
    batch=8,
    device=DEVICE,
    workers=0,
    project=OUT,
    name="baseline",
    exist_ok=True,
    plots=False,
    seed=SEED
)

baseline_path = os.path.join(OUT, "baseline", "weights", "best.pt")
baseline = YOLO(baseline_path)


# ==================================================
# STEP 3: EXTRACT REAL POTHOLE CROPS
# ==================================================

print("\nSTEP 3: Extracting pothole crops")

crops = []

image_files = glob.glob(os.path.join(TRAIN_IMG, "*"))

for image_path in image_files:

    image = cv2.imread(image_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    stem = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(TRAIN_LBL, stem + ".txt")

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        for line in f:

            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w, int((xc + bw / 2) * w))
            y2 = min(h, int((yc + bh / 2) * h))

            if (x2 - x1) < MIN_CROP_PIXELS or (y2 - y1) < MIN_CROP_PIXELS:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size:
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop)
                # Horizontal flip doubles the data for free.
                crops.append(np.ascontiguousarray(crop[:, ::-1]))

if len(crops) < 10:
    raise RuntimeError("Not enough pothole crops to train diffusion.")

crops = np.array(crops, dtype=np.float32) / 127.5 - 1.0

print("Real pothole crops (incl. flips):", len(crops))

crop_tensor = torch.tensor(crops).permute(0, 3, 1, 2)

loader = DataLoader(
    TensorDataset(crop_tensor),
    batch_size=16,
    shuffle=True,
    drop_last=len(crop_tensor) >= 32
)


# ==================================================
# STEP 4: SMALL U-NET DIFFUSION MODEL
# ==================================================

print("\nSTEP 4: Training diffusion model")

betas = torch.linspace(1e-4, 0.04, TIMESTEPS).to(TORCH_DEVICE)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
print("alpha_bar[-1] =", alpha_bar[-1].item(), "(should be close to 0)")


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.t = nn.Linear(tdim, cout)

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.t(te)[:, :, None, None]
        return F.silu(self.n2(self.c2(h)))


class TinyDiffusion(nn.Module):
    def __init__(self, base=64, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(
            nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )
        self.d1 = Block(3, base, tdim)
        self.d2 = Block(base, base * 2, tdim)
        self.mid = Block(base * 2, base * 2, tdim)
        self.u2 = Block(base * 4, base, tdim)
        self.u1 = Block(base * 2, base, tdim)
        self.out = nn.Conv2d(base, 3, 1)

    def embed(self, t):
        half = self.tdim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        a = t.float()[:, None] * freqs[None]
        return self.temb(torch.cat([a.sin(), a.cos()], dim=1))

    def forward(self, x, t):
        te = self.embed(t)
        h1 = self.d1(x, te)                                  # 64x64
        h2 = self.d2(F.avg_pool2d(h1, 2), te)                # 32x32
        m = self.mid(F.avg_pool2d(h2, 2), te)                # 16x16
        u = self.u2(
            torch.cat([F.interpolate(m, scale_factor=2), h2], 1), te
        )
        u = self.u1(
            torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te
        )
        return self.out(u)


diffusion = TinyDiffusion().to(TORCH_DEVICE)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

for epoch in range(DIFF_EPOCHS):

    diffusion.train()
    total_loss = 0

    for (real,) in loader:

        real = real.to(TORCH_DEVICE)
        t = torch.randint(0, TIMESTEPS, (real.size(0),), device=TORCH_DEVICE)

        noise = torch.randn_like(real)
        a = alpha_bar[t].view(-1, 1, 1, 1)
        noisy = torch.sqrt(a) * real + torch.sqrt(1 - a) * noise

        predicted = diffusion(noisy, t)
        loss = F.mse_loss(predicted, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Diffusion epoch {epoch+1}/{DIFF_EPOCHS} | "
            f"Loss: {total_loss / len(loader):.4f}"
        )


# ==================================================
# STEP 5: GENERATE SYNTHETIC IMAGES
# ==================================================

print("\nSTEP 5: Generating synthetic potholes")

SYNTH = os.path.join(OUT, "synthetic")
if os.path.exists(SYNTH):
    shutil.rmtree(SYNTH)
os.makedirs(SYNTH, exist_ok=True)


@torch.no_grad()
def generate_batch(n):

    diffusion.eval()

    x = torch.randn(n, 3, CROP_SIZE, CROP_SIZE, device=TORCH_DEVICE)

    for t in reversed(range(TIMESTEPS)):

        time = torch.full((n,), t, device=TORCH_DEVICE, dtype=torch.long)
        noise = diffusion(x, time)

        x = (
            x - (betas[t] / torch.sqrt(1 - alpha_bar[t])) * noise
        ) / torch.sqrt(alphas[t])

        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)

    x = x.clamp(-1, 1)
    x = ((x + 1) * 127.5).byte()

    return x.permute(0, 2, 3, 1).cpu().numpy()      # N x H x W x 3 (RGB)


generated = []
BATCH_GEN = 50
idx = 0

while idx < N_GENERATE:
    n = min(BATCH_GEN, N_GENERATE - idx)
    batch = generate_batch(n)

    for img in batch:
        idx += 1
        path = os.path.join(SYNTH, f"pothole_{idx}.jpg")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        generated.append(path)

print("Generated images:", len(generated))

# Sanity check: real crops (top rows) vs generated (bottom rows).
# LOOK AT THIS IMAGE. If generated crops are just noise, fix the
# generator before trusting any YOLO result.
real_show = ((crops[:16] + 1) * 127.5).astype(np.uint8)
gen_show = np.array([
    cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in generated[:16]
])


def make_grid(imgs, cols=8):
    rows = []
    for r in range(0, len(imgs), cols):
        row = list(imgs[r:r + cols])
        while len(row) < cols:
            row.append(np.zeros_like(imgs[0]))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


grid = np.concatenate([make_grid(real_show), make_grid(gen_show)], axis=0)
grid_path = os.path.join(OUT, "real_vs_generated.png")
cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("Saved real-vs-generated grid:", grid_path)


# ==================================================
# STEP 6: USEFULNESS SELECTION
# ==================================================

print("\nSTEP 6: Selecting useful synthetic images")

scores = []

for path in generated:

    result = baseline.predict(
        path,
        imgsz=IMG_SIZE,
        conf=0.01,
        verbose=False
    )[0]

    # p(1-p) is highest when the detector is unsure (confidence near 0.5),
    # i.e. the sample is informative but still looks pothole-like.
    if len(result.boxes):
        confidence = float(result.boxes.conf.max())
        score = confidence * (1 - confidence)
    else:
        confidence = 0.0
        score = 0.0

    scores.append((path, score, confidence))

scores.sort(key=lambda x: x[1], reverse=True)
selected = scores[:N_SELECT]

print(f"\nTop {min(10, len(selected))} selected synthetic samples:")

for path, score, confidence in selected[:10]:
    print(os.path.basename(path), "Score:", round(score, 4),
          "Conf:", round(confidence, 4))


# ==================================================
# STEP 7: CREATE AUGMENTED TRAINING DATASET
# ==================================================

print("\nSTEP 7: Creating augmented dataset")

AUG_TRAIN_IMG = os.path.join(WORK, "augmented", "train", "images")
AUG_TRAIN_LBL = os.path.join(WORK, "augmented", "train", "labels")

os.makedirs(AUG_TRAIN_IMG, exist_ok=True)
os.makedirs(AUG_TRAIN_LBL, exist_ok=True)

# Copy original training images and annotations.
for file in glob.glob(os.path.join(TRAIN_IMG, "*")):
    shutil.copy2(file, AUG_TRAIN_IMG)

for file in glob.glob(os.path.join(TRAIN_LBL, "*")):
    shutil.copy2(file, AUG_TRAIN_LBL)

backgrounds = glob.glob(os.path.join(TRAIN_IMG, "*"))


def boxes_overlap(a, b):
    """a, b = (x1, y1, x2, y2) in pixels."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


made = 0

for path, score, confidence in selected:

    background_path = random.choice(backgrounds)

    image = cv2.imread(background_path)
    patch = cv2.imread(path)

    if image is None or patch is None:
        continue

    h, w = image.shape[:2]

    # Existing labels of the background image must be kept.
    bg_stem = os.path.splitext(os.path.basename(background_path))[0]
    with open(os.path.join(TRAIN_LBL, bg_stem + ".txt"), "r") as f:
        old_labels = f.read()

    if old_labels and not old_labels.endswith("\n"):
        old_labels += "\n"

    existing = []
    for line in old_labels.strip().splitlines():
        _, xc, yc, bw, bh = map(float, line.split())
        existing.append((
            (xc - bw / 2) * w, (yc - bh / 2) * h,
            (xc + bw / 2) * w, (yc + bh / 2) * h
        ))

    # Pick a patch size (even number) and a spot that doesn't cover real potholes.
    max_size = min(128, h // 3, w // 3)
    min_size = min(48, max_size)
    placed = False

    for _ in range(30):
        size = random.randint(min_size, max_size)
        size -= size % 2
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        new_box = (x, y, x + size, y + size)

        if not any(boxes_overlap(new_box, b) for b in existing):
            placed = True
            break

    if not placed:
        continue

    patch = cv2.resize(patch, (size, size))

    # Blend with Poisson (seamless) cloning so there are no hard square edges.
    mask = np.zeros(patch.shape[:2], np.uint8)
    mask[3:-3, 3:-3] = 255

    try:
        image = cv2.seamlessClone(
            patch, image, mask,
            (x + size // 2, y + size // 2),
            cv2.NORMAL_CLONE
        )
    except cv2.error:
        continue

    made += 1
    name = f"synthetic_{made}"

    cv2.imwrite(os.path.join(AUG_TRAIN_IMG, name + ".jpg"), image)

    xc = (x + size / 2) / w
    yc = (y + size / 2) / h
    bw = size / w
    bh = size / h

    with open(os.path.join(AUG_TRAIN_LBL, name + ".txt"), "w") as f:
        f.write(old_labels)
        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

print("Synthetic composites created:", made)

aug_yaml = os.path.join(WORK, "augmented.yaml")

with open(aug_yaml, "w") as f:
    yaml.dump({
        "path": WORK,
        "train": "augmented/train/images",
        "val": "valid/images",
        "names": {0: "Pothole"}
    }, f)


# ==================================================
# STEP 8: TRAIN AUGMENTED YOLO
# ==================================================

print("\nSTEP 8: Training augmented YOLO")

aug_model = YOLO("yolo11n.pt")

aug_model.train(
    data=aug_yaml,
    epochs=YOLO_EPOCHS,
    imgsz=IMG_SIZE,
    batch=8,
    device=DEVICE,
    workers=0,
    project=OUT,
    name="augmented",
    exist_ok=True,
    plots=False,
    seed=SEED
)

aug_path = os.path.join(OUT, "augmented", "weights", "best.pt")
aug_model = YOLO(aug_path)


# ==================================================
# STEP 9: PERFORMANCE COMPARISON
# ==================================================

print("\nSTEP 9: Evaluating both models")

# Both are evaluated on the same untouched real validation set.
base_metrics = baseline.val(
    data=yaml_path, imgsz=IMG_SIZE, device=DEVICE, workers=0, plots=False
)

aug_metrics = aug_model.val(
    data=yaml_path, imgsz=IMG_SIZE, device=DEVICE, workers=0, plots=False
)

metric_names = ["Precision", "Recall", "mAP50", "mAP50-95"]

base_values = [
    base_metrics.box.mp, base_metrics.box.mr,
    base_metrics.box.map50, base_metrics.box.map
]

aug_values = [
    aug_metrics.box.mp, aug_metrics.box.mr,
    aug_metrics.box.map50, aug_metrics.box.map
]

print("\n========== FINAL COMPARISON ==========")

for name, b, a in zip(metric_names, base_values, aug_values):
    print(f"{name:12} Baseline: {b:.4f} | Augmented: {a:.4f}")

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(x - width / 2, np.array(base_values) * 100, width, label="Baseline")
plt.bar(x + width / 2, np.array(aug_values) * 100, width, label="Augmented")

plt.xticks(x, metric_names)
plt.ylabel("Percentage (%)")
plt.title("YOLO11n: Baseline vs Diffusion Augmentation")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(OUT, "final_comparison.png"), dpi=200)
plt.show()

print("\nPROJECT COMPLETED!")
print("All results saved in:", OUT)