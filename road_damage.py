"""
Diffusion-based data augmentation for pothole detection (YOLO11n, RDD2022 India)

Pipeline
  1. Build a pothole-only (D40) dataset
  2. Train baseline YOLO
  3. Extract real pothole crops
  4. Train a small U-Net diffusion model (DDPM) on the crops
  5. Generate synthetic pothole patches
  6. Rank them with the baseline detector (usefulness score)
  7. Blend them into real road images (Poisson blending) with correct labels
  8. Train YOLO on real + synthetic data
  9. Compare models and save all visualisations to <OUT>/visuals

Tip: set QUICK_TEST = True first to check that everything runs in a few minutes.
"""

import os
import math
import glob
import time
import random
import shutil
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO

# ==================================================
# SETTINGS
# ==================================================

QUICK_TEST = False          # True = tiny run just to verify the code works
RUN_ABLATION = False        # True = also train a 3rd model (random synthetic pick)
REUSE_DIFFUSION = True      # reuse saved diffusion weights if they exist
SHOW_PLOTS = False          # True = pop up every figure (blocks until closed)

ROOT = r"D:\data\COMPUTER VISON\M6_Road_Damage_Detection"
DATA = os.path.join(ROOT, "datasets", "RDD2022", "India")

IMG_SIZE = 416
CROP_SIZE = 64
MIN_CROP_PIXELS = 24

TRAIN_LIMIT = 300
VAL_LIMIT = 60
YOLO_EPOCHS = 50
DIFF_EPOCHS = 200
TIMESTEPS = 300
N_GENERATE = 200
N_SELECT = 50
SEED = 42

if QUICK_TEST:
    TRAIN_LIMIT, VAL_LIMIT = 30, 10
    YOLO_EPOCHS, DIFF_EPOCHS = 1, 3
    TIMESTEPS = 50
    N_GENERATE, N_SELECT = 20, 10

OUT = os.path.join(
    ROOT, "SIMPLE_DIFFUSION_PROJECT_QUICK" if QUICK_TEST
    else "SIMPLE_DIFFUSION_PROJECT"
)
os.makedirs(OUT, exist_ok=True)

VIS = os.path.join(OUT, "visuals")
os.makedirs(VIS, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "CPU")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

timings = {}


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *args):
        timings[self.label] = time.time() - self.t


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(VIS, name), dpi=200, bbox_inches="tight")
    print("  saved visual:", name)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tensor_to_uint8(x):
    """N x 3 x H x W in [-1, 1]  ->  N x H x W x 3 uint8 (RGB)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()


# ==================================================
# STEP 1: PREPARE POTHOLE-ONLY DATASET
# ==================================================

print("\nSTEP 1: Preparing pothole dataset")

with open(os.path.join(DATA, "data.yaml"), "r") as f:
    config = yaml.safe_load(f)

names = config["names"]

if isinstance(names, dict):
    pothole_id = next(
        int(k) for k, v in names.items() if str(v).upper() == "D40"
    )
else:
    pothole_id = next(
        i for i, v in enumerate(names) if str(v).upper() == "D40"
    )

print("Pothole class ID:", pothole_id)

WORK = os.path.join(OUT, "dataset")

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

        potholes = []
        for line in lines:
            values = line.strip().split()
            if len(values) >= 5 and int(float(values[0])) == pothole_id:
                potholes.append("0 " + " ".join(values[1:5]) + "\n")

        if not potholes:
            continue

        shutil.copy2(
            image_path, os.path.join(image_out, os.path.basename(image_path))
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
# YOLO HELPERS
# ==================================================

def train_yolo(data_yaml, name):
    model = YOLO("yolo11n.pt")
    model.train(
        data=data_yaml,
        epochs=YOLO_EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,
        device=DEVICE,
        workers=0,
        project=OUT,
        name=name,
        exist_ok=True,
        plots=False,
        seed=SEED
    )
    save_dir = str(model.trainer.save_dir)
    best = YOLO(os.path.join(save_dir, "weights", "best.pt"))
    return best, save_dir


def evaluate(model):
    m = model.val(
        data=yaml_path, imgsz=IMG_SIZE, device=DEVICE,
        workers=0, plots=False, verbose=False
    )
    return [m.box.mp, m.box.mr, m.box.map50, m.box.map]


# ==================================================
# STEP 2: TRAIN BASELINE YOLO
# ==================================================

print("\nSTEP 2: Training baseline YOLO11n")

with Timer("Baseline YOLO training"):
    baseline, baseline_dir = train_yolo(yaml_path, "baseline")


# ==================================================
# STEP 3: EXTRACT REAL POTHOLE CROPS
# ==================================================

print("\nSTEP 3: Extracting pothole crops")

crops = []

for image_path in glob.glob(os.path.join(TRAIN_IMG, "*")):

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
                crops.append(np.ascontiguousarray(crop[:, ::-1]))   # h-flip

if len(crops) < 4:
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
# STEP 4: DIFFUSION MODEL (small U-Net, DDPM)
# ==================================================

print("\nSTEP 4: Diffusion model")

betas = torch.linspace(1e-4, 0.04, TIMESTEPS).to(TORCH_DEVICE)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
print("alpha_bar[-1] =", alpha_bar[-1].item(), "(close to 0 is good)")


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
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), te)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te)
        return self.out(u)


diffusion = TinyDiffusion().to(TORCH_DEVICE)
n_params = sum(p.numel() for p in diffusion.parameters())
print(f"Diffusion model parameters: {n_params:,}")

DIFF_WEIGHTS = os.path.join(OUT, "diffusion.pt")
loss_history = []

if REUSE_DIFFUSION and os.path.exists(DIFF_WEIGHTS):
    ckpt = torch.load(DIFF_WEIGHTS, map_location=TORCH_DEVICE)
    diffusion.load_state_dict(ckpt["model"])
    loss_history = ckpt["loss"]
    print("Loaded saved diffusion weights (delete diffusion.pt to retrain).")
else:
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

    with Timer("Diffusion training"):
        for epoch in range(DIFF_EPOCHS):

            diffusion.train()
            total_loss = 0

            for (real,) in loader:
                real = real.to(TORCH_DEVICE)
                t = torch.randint(
                    0, TIMESTEPS, (real.size(0),), device=TORCH_DEVICE
                )

                noise = torch.randn_like(real)
                a = alpha_bar[t].view(-1, 1, 1, 1)
                noisy = torch.sqrt(a) * real + torch.sqrt(1 - a) * noise

                loss = F.mse_loss(diffusion(noisy, t), noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            loss_history.append(total_loss / len(loader))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Diffusion epoch {epoch+1}/{DIFF_EPOCHS} | "
                      f"Loss: {loss_history[-1]:.4f}")

    torch.save({"model": diffusion.state_dict(), "loss": loss_history},
               DIFF_WEIGHTS)


# ==================================================
# STEP 5: GENERATE SYNTHETIC IMAGES
# ==================================================

print("\nSTEP 5: Generating synthetic potholes")

SYNTH = os.path.join(OUT, "synthetic")
if os.path.exists(SYNTH):
    shutil.rmtree(SYNTH)
os.makedirs(SYNTH, exist_ok=True)


@torch.no_grad()
def generate_batch(n, snap_steps=None):
    """Reverse diffusion. If snap_steps is given, also return snapshots."""
    diffusion.eval()
    x = torch.randn(n, 3, CROP_SIZE, CROP_SIZE, device=TORCH_DEVICE)
    snaps = []

    for t in reversed(range(TIMESTEPS)):
        time_t = torch.full((n,), t, device=TORCH_DEVICE, dtype=torch.long)
        noise = diffusion(x, time_t)

        x = (x - (betas[t] / torch.sqrt(1 - alpha_bar[t])) * noise) \
            / torch.sqrt(alphas[t])

        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)

        if snap_steps is not None and t in snap_steps:
            snaps.append((t, x.clone()))

    imgs = tensor_to_uint8(x)
    return (imgs, snaps) if snap_steps is not None else imgs


generated = []
idx = 0

with Timer("Generating synthetic potholes"):
    while idx < N_GENERATE:
        n = min(50, N_GENERATE - idx)
        for img in generate_batch(n):
            idx += 1
            path = os.path.join(SYNTH, f"pothole_{idx}.jpg")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            generated.append(path)

print("Generated images:", len(generated))


# ==================================================
# STEP 6: USEFULNESS SELECTION
# ==================================================

print("\nSTEP 6: Selecting useful synthetic images")

scores = []

for path in generated:
    result = baseline.predict(
        path, imgsz=IMG_SIZE, conf=0.01, verbose=False
    )[0]

    # p(1-p) peaks at confidence 0.5: samples the detector finds
    # "pothole-like but uncertain" are the most informative to train on.
    if len(result.boxes):
        confidence = float(result.boxes.conf.max())
        score = confidence * (1 - confidence)
    else:
        confidence, score = 0.0, 0.0

    scores.append((path, score, confidence))

scores.sort(key=lambda x: x[1], reverse=True)
selected = scores[:N_SELECT]

print(f"Top {min(10, len(selected))} selected samples:")
for path, score, confidence in selected[:10]:
    print(" ", os.path.basename(path), "score", round(score, 4),
          "conf", round(confidence, 4))


# ==================================================
# STEP 7: BUILD AUGMENTED DATASET
# ==================================================

def boxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def build_augmented(chosen, tag, seed_offset=0):
    """Blend synthetic patches into real images; keep all old labels."""
    rng = random.Random(SEED + seed_offset)

    img_dir = os.path.join(WORK, tag, "train", "images")
    lbl_dir = os.path.join(WORK, tag, "train", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for file in glob.glob(os.path.join(TRAIN_IMG, "*")):
        shutil.copy2(file, img_dir)
    for file in glob.glob(os.path.join(TRAIN_LBL, "*")):
        shutil.copy2(file, lbl_dir)

    backgrounds = glob.glob(os.path.join(TRAIN_IMG, "*"))
    composites = []
    made = 0

    for path, _, _ in chosen:

        background_path = rng.choice(backgrounds)
        image = cv2.imread(background_path)
        patch = cv2.imread(path)

        if image is None or patch is None:
            continue

        h, w = image.shape[:2]

        stem = os.path.splitext(os.path.basename(background_path))[0]
        with open(os.path.join(TRAIN_LBL, stem + ".txt"), "r") as f:
            old_labels = f.read()
        if old_labels and not old_labels.endswith("\n"):
            old_labels += "\n"

        existing = []
        for line in old_labels.strip().splitlines():
            _, xc, yc, bw, bh = map(float, line.split())
            existing.append(((xc - bw / 2) * w, (yc - bh / 2) * h,
                             (xc + bw / 2) * w, (yc + bh / 2) * h))

        max_size = min(128, h // 3, w // 3)
        min_size = min(48, max_size)
        placed = False

        for _ in range(30):
            size = rng.randint(min_size, max_size)
            size -= size % 2
            x = rng.randint(0, w - size)
            y = rng.randint(0, h - size)
            if not any(boxes_overlap((x, y, x + size, y + size), b)
                       for b in existing):
                placed = True
                break

        if not placed:
            continue

        patch = cv2.resize(patch, (size, size))
        mask = np.zeros(patch.shape[:2], np.uint8)
        mask[3:-3, 3:-3] = 255

        try:
            image = cv2.seamlessClone(
                patch, image, mask, (x + size // 2, y + size // 2),
                cv2.NORMAL_CLONE
            )
        except cv2.error:
            continue

        made += 1
        name = f"synthetic_{made}"
        img_out = os.path.join(img_dir, name + ".jpg")
        lbl_out = os.path.join(lbl_dir, name + ".txt")

        cv2.imwrite(img_out, image)

        with open(lbl_out, "w") as f:
            f.write(old_labels)
            f.write(f"0 {(x + size / 2) / w:.6f} {(y + size / 2) / h:.6f} "
                    f"{size / w:.6f} {size / h:.6f}\n")

        composites.append((img_out, lbl_out))

    yaml_file = os.path.join(WORK, tag + ".yaml")
    with open(yaml_file, "w") as f:
        yaml.dump({
            "path": WORK,
            "train": f"{tag}/train/images",
            "val": "valid/images",
            "names": {0: "Pothole"}
        }, f)

    print(f"[{tag}] synthetic composites created: {made}")
    return yaml_file, composites


print("\nSTEP 7: Creating augmented dataset")
aug_yaml, composites = build_augmented(selected, "augmented")

if RUN_ABLATION:
    random_pick = random.sample(scores, min(N_SELECT, len(scores)))
    rand_yaml, _ = build_augmented(random_pick, "ablation_random", 1)


# ==================================================
# STEP 8: TRAIN AUGMENTED YOLO
# ==================================================

print("\nSTEP 8: Training augmented YOLO")

with Timer("Augmented YOLO training"):
    aug_model, aug_dir = train_yolo(aug_yaml, "augmented")

models = {"Baseline": baseline, "Diffusion + selection": aug_model}
run_dirs = {"Baseline": baseline_dir, "Diffusion + selection": aug_dir}

if RUN_ABLATION:
    with Timer("Ablation YOLO training"):
        rand_model, rand_dir = train_yolo(rand_yaml, "ablation_random")
    models["Diffusion (random pick)"] = rand_model
    run_dirs["Diffusion (random pick)"] = rand_dir


# ==================================================
# STEP 9: EVALUATION
# ==================================================

print("\nSTEP 9: Evaluating models on the real validation set")

metric_names = ["Precision", "Recall", "mAP50", "mAP50-95"]
results = {name: evaluate(m) for name, m in models.items()}

print("\n========== FINAL COMPARISON ==========")
header = f"{'Metric':12}" + "".join(f"{n:>26}" for n in results)
print(header)
for i, mname in enumerate(metric_names):
    print(f"{mname:12}" + "".join(f"{results[n][i]:>26.4f}" for n in results))

df = pd.DataFrame(results, index=metric_names)
base_col = df["Baseline"]
for c in list(df.columns):
    if c != "Baseline":
        df[f"{c} (delta vs baseline)"] = df[c] - base_col
df.to_csv(os.path.join(OUT, "metrics.csv"))


# ==================================================
# STEP 10: VISUALISATIONS
# ==================================================

print("\nSTEP 10: Creating visualisations in", VIS)


def vis_forward_process():
    """Noise schedule + a real pothole being destroyed by noise."""
    x0 = crop_tensor[:1].to(TORCH_DEVICE)
    noise = torch.randn_like(x0)
    steps = [0, TIMESTEPS // 8, TIMESTEPS // 4, TIMESTEPS // 2,
             3 * TIMESTEPS // 4, TIMESTEPS - 1]

    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, len(steps), height_ratios=[1.2, 1])

    ax = fig.add_subplot(gs[0, :])
    ab = alpha_bar.cpu().numpy()
    ax.plot(ab, label="signal kept (alpha_bar)", linewidth=2)
    ax.plot(1 - ab, label="noise level (1 - alpha_bar)", linewidth=2)
    ax.set_xlabel("timestep t")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Noise schedule")

    for i, t in enumerate(steps):
        a = alpha_bar[t]
        xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise
        axi = fig.add_subplot(gs[1, i])
        axi.imshow(tensor_to_uint8(xt)[0])
        axi.set_title(f"t = {t}")
        axi.axis("off")

    fig.suptitle("Forward diffusion: a real pothole is gradually turned into noise",
                 fontsize=14)
    save_fig("1_forward_diffusion.png")


def vis_loss_curve():
    if not loss_history:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Noise-prediction MSE (log scale)")
    plt.title("Diffusion model training loss")
    plt.grid(alpha=0.3)
    save_fig("2_diffusion_loss.png")


def vis_reverse_process():
    """Pure noise -> pothole patch, step by step."""
    wanted = sorted({TIMESTEPS - 1, int(TIMESTEPS * 0.75), TIMESTEPS // 2,
                     TIMESTEPS // 4, TIMESTEPS // 8, TIMESTEPS // 20, 0},
                    reverse=True)
    _, snaps = generate_batch(4, snap_steps=wanted)

    fig, axes = plt.subplots(4, len(snaps), figsize=(2 * len(snaps), 8.5))
    for c, (t, x) in enumerate(snaps):
        imgs = tensor_to_uint8(x)
        for r in range(4):
            axes[r, c].imshow(imgs[r])
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"t = {t}")
    fig.suptitle("Reverse diffusion: noise is denoised into a pothole patch",
                 fontsize=14)
    save_fig("3_reverse_diffusion.png")


def make_grid(imgs, cols=8):
    rows = []
    for r in range(0, len(imgs), cols):
        row = list(imgs[r:r + cols])
        while len(row) < cols:
            row.append(np.zeros_like(imgs[0]))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def vis_real_vs_generated():
    k = min(16, len(crops))
    pick = np.random.choice(len(crops), k, replace=False)
    real_show = ((crops[pick] + 1) * 127.5).astype(np.uint8)
    gen_show = np.array([bgr2rgb(cv2.imread(p)) for p in generated[:16]])

    fig, axes = plt.subplots(2, 1, figsize=(14, 4.2))
    axes[0].imshow(make_grid(real_show))
    axes[0].set_title("Real pothole crops")
    axes[1].imshow(make_grid(gen_show))
    axes[1].set_title("Diffusion-generated pothole patches")
    for a in axes:
        a.axis("off")
    save_fig("4_real_vs_generated.png")


def vis_selected(k=16):
    k = min(k, len(selected))
    cols = math.ceil(k / 2)
    fig, axes = plt.subplots(2, cols, figsize=(2 * cols, 5.4))
    axes = np.array(axes).reshape(-1)
    for a in axes:
        a.axis("off")
    for a, (p, s, c) in zip(axes, selected[:k]):
        a.imshow(bgr2rgb(cv2.imread(p)))
        a.set_title(f"conf {c:.2f}\nscore {s:.3f}", fontsize=8)
    fig.suptitle("Synthetic patches selected by the usefulness score "
                 "(highest first)", fontsize=13)
    save_fig("5_selected_samples.png")


def vis_composites(k=8):
    items = composites[:k]
    if not items:
        return
    cols = 4
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for a in axes:
        a.axis("off")

    for a, (ip, lp) in zip(axes, items):
        img = cv2.imread(ip)
        h, w = img.shape[:2]
        with open(lp) as f:
            lines = f.read().strip().splitlines()
        for j, line in enumerate(lines):
            _, xc, yc, bw, bh = map(float, line.split())
            p1 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
            p2 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
            color = (0, 0, 255) if j == len(lines) - 1 else (0, 200, 0)
            cv2.rectangle(img, p1, p2, color, 2)
        a.imshow(bgr2rgb(img))

    fig.suptitle("Augmented training images  (red = synthetic pothole, "
                 "green = original real labels)", fontsize=13)
    save_fig("6_augmented_examples.png")


def vis_detections(n=5):
    val_images = sorted(glob.glob(os.path.join(VAL_IMG, "*")))[:n]
    cols = 1 + len(models)
    fig, axes = plt.subplots(len(val_images), cols,
                             figsize=(4.6 * cols, 4.6 * len(val_images)))
    axes = np.array(axes).reshape(len(val_images), cols)

    for r, ip in enumerate(val_images):
        img0 = cv2.imread(ip)
        h, w = img0.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]

        gt = img0.copy()
        with open(os.path.join(VAL_LBL, stem + ".txt")) as f:
            for line in f:
                _, xc, yc, bw, bh = map(float, line.split())
                cv2.rectangle(
                    gt,
                    (int((xc - bw / 2) * w), int((yc - bh / 2) * h)),
                    (int((xc + bw / 2) * w), int((yc + bh / 2) * h)),
                    (0, 200, 0), 3
                )
        axes[r, 0].imshow(bgr2rgb(gt))
        axes[r, 0].set_title("Ground truth")
        axes[r, 0].axis("off")

        for c, (name, model) in enumerate(models.items(), start=1):
            vis = img0.copy()
            res = model.predict(ip, imgsz=IMG_SIZE, conf=0.25, verbose=False)[0]
            for box, cf in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(vis, f"{cf:.2f}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            axes[r, c].imshow(bgr2rgb(vis))
            axes[r, c].set_title(f"{name} ({len(res.boxes)} detections)")
            axes[r, c].axis("off")

    fig.suptitle("Detections on unseen validation images", fontsize=14)
    save_fig("7_detection_comparison.png")


def vis_metrics():
    n = len(results)
    x = np.arange(len(metric_names))
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 5.5))
    top = 0
    for i, (name, vals) in enumerate(results.items()):
        bars = ax.bar(x + (i - (n - 1) / 2) * width,
                      np.array(vals) * 100, width, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{v * 100:.1f}", ha="center", fontsize=9)
        top = max(top, max(vals) * 100)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, min(105, top + 12))
    ax.set_title("YOLO11n: baseline vs diffusion augmentation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig("8_metrics_comparison.png")


def vis_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    drew = False
    for name, d in run_dirs.items():
        csv_path = os.path.join(d, "results.csv")
        if not os.path.exists(csv_path):
            continue
        d_f = pd.read_csv(csv_path)
        d_f.columns = d_f.columns.str.strip()
        try:
            axes[0].plot(d_f["epoch"], d_f["metrics/mAP50(B)"], label=name)
            axes[1].plot(d_f["epoch"], d_f["metrics/mAP50-95(B)"], label=name)
            drew = True
        except KeyError:
            pass
    if not drew:
        plt.close()
        return
    axes[0].set_title("Validation mAP50 per epoch")
    axes[1].set_title("Validation mAP50-95 per epoch")
    for a in axes:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
        a.legend()
    save_fig("9_yolo_training_curves.png")


for fn in (vis_forward_process, vis_loss_curve, vis_reverse_process,
           vis_real_vs_generated, vis_selected, vis_composites,
           vis_detections, vis_metrics, vis_training_curves):
    try:
        fn()
    except Exception as e:                      # a plot must never kill the run
        print(f"  (skipped {fn.__name__}: {e})")


# ==================================================
# SUMMARY
# ==================================================

summary_lines = [
    "POTHOLE DETECTION WITH DIFFUSION AUGMENTATION - SUMMARY",
    f"Train images: {train_count} | Val images: {val_count}",
    f"Real crops (with flips): {len(crops)}",
    f"Diffusion: U-Net, {n_params:,} params, {TIMESTEPS} steps, "
    f"{DIFF_EPOCHS} epochs",
    f"Generated: {len(generated)} | Selected: {len(selected)} | "
    f"Composites: {len(composites)}",
    f"YOLO epochs: {YOLO_EPOCHS}",
    "",
    df.round(4).to_string(),
    "",
    "Timings (s):",
] + [f"  {k}: {v:.1f}" for k, v in timings.items()]

with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("\n" + "\n".join(summary_lines))
print("\nPROJECT COMPLETED!")
print("Results:", OUT)
print("Visuals:", VIS)

if hasattr(os, "startfile"):            # Windows: open the visuals folder
    try:
        os.startfile(VIS)
    except Exception:
        pass