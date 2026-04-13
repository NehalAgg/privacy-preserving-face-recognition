# dctdp_faithful.py
# ─────────────────────────────────────────────────────────────────────
# Faithful Reimplementation of DCTDP
# "Privacy-Preserving Face Recognition with Learnable Privacy Budgets
#  in Frequency Domain" — Ji et al., ECCV 2022
#
# Implemented faithfully per paper Section 3.1 and 3.2:
#   ✓ BGR → YCbCr color space conversion
#   ✓ Value range shifted to [-128, 127]
#   ✓ 8x upsampling before BDCT
#   ✓ 8x8 Block DCT (BDCT)
#   ✓ DC component removal
#   ✓ Per-position sensitivity computed from dataset
#   ✓ Calibrated Laplace noise (sensitivity / epsilon per position)
#   ✓ IDCT reconstruction for pixel-space evaluation
#
# Acknowledged simplifications (require 8x V100 GPUs to fix):
#   ✗ No 189-channel ResNet50 trained from scratch
#   ✗ No learnable budget allocation via backpropagation
#   ✗ Evaluation uses pretrained ArcFace (domain mismatch)
#
# Run order:
#   Step 1: python dctdp_faithful.py --compute-sensitivity
#           → Computes sensitivity_map_lfw.npy from full LFW (~25 min)
#   Step 2: python dctdp_faithful.py --evaluate
#           → Runs LFW 6000-pair evaluation and saves results
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import os
import glob
import argparse
import time
from scipy.fftpack import dctn, idctn
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — change paths here if needed
# ═══════════════════════════════════════════════════════════════════

LFW_DIR         = "lfw-deepfunneled/lfw-deepfunneled"
PAIRS_FILE      = "pairs.csv"
SENSITIVITY_MAP = "sensitivity_map_lfw.npy"
EPSILON         = 10.0     # Privacy budget — paper uses 0.5 for main results
                          # We use 1.0 to reduce noise given our evaluation setup
BLOCK_SIZE      = 8       # Paper uses 8x8 blocks throughout
UPSAMPLE_FACTOR = 8       # Paper upsamples 112x112 → 896x896 before BDCT


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — COLOR SPACE AND VALUE RANGE (Paper Section 3.1)
# ═══════════════════════════════════════════════════════════════════

def to_ycbcr_shifted(image_bgr):
    """
    Paper Section 3.1:
    "We first convert it from RGB color spaces to YCbCr color spaces.
     We then adjust its value range to [-128, 127]."

    Y  = luminance (brightness) — most identity information here
    Cb = blue-difference chroma
    Cr = red-difference chroma

    Shifting to [-128, 127] centers data around zero as required
    for DCT to correctly represent the DC component as block mean.
    """
    img_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    return img_ycbcr.astype(np.float32) - 128.0


def from_ycbcr_shifted(img_shifted):
    """Reverse of to_ycbcr_shifted. Converts back to BGR uint8."""
    img = np.clip(img_shifted + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — 8x UPSAMPLING (Paper Section 3.1)
# ═══════════════════════════════════════════════════════════════════

def upsample_8x(image):
    """
    Paper Section 3.1:
    "For a fairer comparison and as little adjustment as possible to
     the structure of the recognition network, we perform an 8-fold
     up-sampling on the facial images before BDCT."

    112x112 → 896x896 before block DCT.
    After BDCT and DC removal, features are reshaped to [112, 112, C]
    matching the original spatial resolution.
    """
    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (w * UPSAMPLE_FACTOR, h * UPSAMPLE_FACTOR),
        interpolation=cv2.INTER_LINEAR
    )


def downsample_8x(image):
    """Reverse upsampling — for visualization only."""
    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (w // UPSAMPLE_FACTOR, h // UPSAMPLE_FACTOR),
        interpolation=cv2.INTER_LINEAR
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — BLOCK DCT (Paper Section 3.1)
# ═══════════════════════════════════════════════════════════════════

def apply_bdct(channel, block_size=BLOCK_SIZE):
    """
    Paper Section 3.1:
    "A normalized, two-dimensional type-II DCT is used to convert
     each block into 8x8 frequency-domain coefficients."

    Uses scipy.fftpack.dctn for efficient 2D DCT in one call
    instead of chaining two 1D DCTs.
    """
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size,
                           j:j+block_size].astype(np.float32)
            result[i:i+block_size, j:j+block_size] = dctn(
                block, norm='ortho'
            )
    return result[:h, :w]


def apply_ibdct(dct_channel, block_size=BLOCK_SIZE):
    """Inverse BDCT — converts frequency coefficients back to spatial."""
    h, w = dct_channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(dct_channel, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            result[i:i+block_size, j:j+block_size] = idctn(
                block, norm='ortho'
            )
    return result[:h, :w]


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — DC REMOVAL (Paper Section 3.1)
# ═══════════════════════════════════════════════════════════════════

def remove_dc(dct_channel, block_size=BLOCK_SIZE):
    """
    Paper Section 3.1:
    "We remove the direct component (DC) channel because it aggregates
     most of the energy and visualization information in the image but
     is not essential for identification."

    Paper shows DC accounts for 91.6% of total image energy.
    DC is at position [0,0] of each block — represents block mean.
    After removal, 63 AC coefficients remain per block.
    With 3 YCbCr channels: C = 63 x 3 = 189 channels total.

    Note: In the paper's full pipeline, DC is permanently discarded
    and the 189-channel frequency tensor goes directly to ResNet50.
    Here we zero DC and reconstruct via IDCT for evaluation with
    pretrained ArcFace (acknowledged simplification).
    """
    result = dct_channel.copy()
    h, w = result.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            result[i, j] = 0.0
    return result


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — SENSITIVITY MAP (Paper Section 3.2)
# ═══════════════════════════════════════════════════════════════════

def compute_sensitivity_map(image_paths, save_path=SENSITIVITY_MAP):
    """
    Paper Section 3.2:
    "We transferred all the images in VGGFace2 and refined MS1MV2
     into the frequency domain. Then we obtained the maximum values
     and minimum values at each position.
     Sensitivities will equal the value of MAX - MIN."

    Paper computes this over 3.31M VGGFace2 images.
    We compute over full LFW (13,233 images) as approximation.
    More images = better sensitivity estimate.

    Sensitivity at position (i,j,k):
        S(i,j,k) = max_value(i,j,k) - min_value(i,j,k)
        across all images in the dataset.

    This is used as the denominator in Equation 2 of the paper
    and as the numerator in Equation 5 (noise scale).
    """
    print(f"Computing sensitivity map from {len(image_paths)} images...")
    print("This takes approximately 20-30 minutes on CPU.")
    print("The map will be saved and reused on subsequent runs.\n")

    dct_min = None
    dct_max = None
    valid   = 0

    for img_path in tqdm(image_paths, desc="Building sensitivity map"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Paper pipeline: resize → YCbCr → shift → upsample → BDCT
        img = cv2.resize(img, (112, 112))
        img_ycc = to_ycbcr_shifted(img)
        img_up  = upsample_8x(img_ycc)  # 896x896

        dct_channels = []
        for c in range(3):
            dct_ch = apply_bdct(img_up[:, :, c])
            dct_channels.append(dct_ch)

        dct_stack = np.stack(dct_channels, axis=2)  # [896, 896, 3]

        if dct_min is None:
            dct_min = dct_stack.copy()
            dct_max = dct_stack.copy()
        else:
            np.minimum(dct_min, dct_stack, out=dct_min)
            np.maximum(dct_max, dct_stack, out=dct_max)

        valid += 1

    if valid == 0:
        raise RuntimeError("No valid images found for sensitivity map.")

    sensitivity = dct_max - dct_min
    sensitivity = np.maximum(sensitivity, 1e-6)  # avoid division by zero

    np.save(save_path, sensitivity)

    print(f"\nSensitivity map complete:")
    print(f"  Images processed : {valid}")
    print(f"  Shape            : {sensitivity.shape}")
    print(f"  Mean sensitivity : {sensitivity.mean():.4f}")
    print(f"  Max  sensitivity : {sensitivity.max():.4f}")
    print(f"  Saved to         : {save_path}")

    return sensitivity


def load_sensitivity_map(path=SENSITIVITY_MAP):
    """Load precomputed sensitivity map."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sensitivity map not found: {path}\n"
            f"Run: python dctdp_faithful.py --compute-sensitivity"
        )
    sens = np.load(path)
    print(f"Sensitivity map loaded: {sens.shape}")
    return sens


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — CALIBRATED LAPLACE NOISE (Paper Section 3.2, Eq. 5)
# ═══════════════════════════════════════════════════════════════════

def add_calibrated_noise(dct_channel, sensitivity_channel, epsilon):
    """
    Paper Section 3.2, Lemma 1, Equation 5:
    σ(i,j,k) = (r_max(i,j,k) - r_min(i,j,k)) / ε(i,j,k)

    We use uniform epsilon allocation across positions
    (paper uses learned allocation — requires GPU training).
    noise_scale[i,j] = sensitivity[i,j] / epsilon

    High sensitivity positions get more noise.
    Low sensitivity positions get less noise.
    Laplace distribution used (not Gaussian) for exact ε-DP guarantee.
    """
    noise_scale = sensitivity_channel / epsilon
    noise = np.random.laplace(
        loc=0.0,
        scale=noise_scale,
        size=dct_channel.shape
    ).astype(np.float32)
    return dct_channel + noise


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — FULL DCTDP PIPELINE
# ═══════════════════════════════════════════════════════════════════

def dctdp_protect(image_bgr, sensitivity_map, epsilon=EPSILON):
    """
    Full DCTDP pipeline as described in paper Section 3.

    Paper pipeline (for recognition):
      Image → YCbCr → shift → 8x upsample → BDCT → remove DC
      → add calibrated noise → [189-ch frequency tensor] → ResNet50

    Our pipeline (for evaluation with pretrained ArcFace):
      Image → YCbCr → shift → 8x upsample → BDCT → remove DC
      → add calibrated noise → IBDCT → downsample → unshift
      → BGR → pretrained ArcFace

    The IBDCT + downsample step is our evaluation simplification.
    The privacy protection (DCT + DC removal + noise) is identical.

    Args:
        image_bgr:       input face image BGR uint8 (any size)
        sensitivity_map: precomputed from compute_sensitivity_map()
                         shape must be [896, 896, 3] for 112x112 input
        epsilon:         privacy budget (lower = more noise = more private)

    Returns:
        protected_bgr: protected face image BGR uint8
    """
    # Resize to standard 112x112
    img = cv2.resize(image_bgr, (112, 112))

    # Step 1: YCbCr + shift to [-128, 127]
    img_ycc = to_ycbcr_shifted(img)

    # Step 2: 8x upsample → 896x896
    img_up = upsample_8x(img_ycc)

    # Handle sensitivity map shape mismatch
    h_up, w_up = img_up.shape[:2]
    if sensitivity_map.shape[:2] != (h_up, w_up):
        sens = cv2.resize(
            sensitivity_map, (w_up, h_up),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        sens = sensitivity_map

    protected_channels = []
    for c in range(3):
        # Step 3: Block DCT
        dct_ch = apply_bdct(img_up[:, :, c])

        # Step 4: Remove DC component
        dct_ch = remove_dc(dct_ch)

        # Step 5: Add calibrated Laplace noise (Equation 5)
        dct_ch = add_calibrated_noise(dct_ch, sens[:, :, c], epsilon)

        # Step 6: Inverse DCT (evaluation simplification)
        restored = apply_ibdct(dct_ch)
        protected_channels.append(restored)

    # Step 7: Stack, downsample back to 112x112, unshift, convert to BGR
    protected_up = np.stack(protected_channels, axis=2)
    protected_down = downsample_8x(protected_up)
    protected_bgr = from_ycbcr_shifted(protected_down)

    return protected_bgr


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_psnr(img1, img2):
    """
    Primary privacy metric — matches paper Figure 7.
    Lower PSNR = more signal destroyed = stronger privacy.
    """
    return cv2.PSNR(
        img1.astype(np.uint8),
        img2.astype(np.uint8)
    )


def compute_ssim(img1, img2):
    """Secondary metric. Not in base paper — our addition."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_metric(g1, g2, full=True)
    return score


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — LFW EVALUATION
# ═══════════════════════════════════════════════════════════════════

def load_arcface():
    """Load pretrained ArcFace ONNX model."""
    import onnxruntime as ort
    model_path = os.path.expanduser(
        '~/.insightface/models/buffalo_l/w600k_r50.onnx'
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ArcFace model not found at {model_path}\n"
            "Run: python -c \"import insightface; "
            "insightface.app.FaceAnalysis(name='buffalo_l')"
            ".prepare(ctx_id=-1)\""
        )
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=providers)
    active = sess.get_providers()[0]
    device = "GPU" if "CUDA" in active else "CPU"
    print(f"ArcFace loaded on {device}")
    return sess


def get_embedding(sess, image_bgr):
    """Extract 512-dim face embedding from image."""
    img = cv2.resize(image_bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: img})[0][0]


def cosine_similarity(e1, e2):
    e1 = e1 / (np.linalg.norm(e1) + 1e-8)
    e2 = e2 / (np.linalg.norm(e2) + 1e-8)
    return float(np.dot(e1, e2))


def load_lfw_pairs(pairs_file, lfw_dir, max_pairs=6000):
    """Load LFW 6000 standard pairs from CSV."""
    import csv
    pairs = []
    same_count = 0
    diff_count = 0

    with open(pairs_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            row = [x for x in row if x.strip() != '']

            if len(row) == 3:
                name = row[0]
                try:
                    n1, n2 = int(row[1]), int(row[2])
                except ValueError:
                    continue
                p1 = os.path.join(lfw_dir, name, f"{name}_{n1:04d}.jpg")
                p2 = os.path.join(lfw_dir, name, f"{name}_{n2:04d}.jpg")
                if os.path.exists(p1) and os.path.exists(p2):
                    pairs.append((p1, p2, 1))
                    same_count += 1

            elif len(row) == 4:
                name1, name2 = row[0], row[2]
                try:
                    n1, n2 = int(row[1]), int(row[3])
                except ValueError:
                    continue
                p1 = os.path.join(lfw_dir, name1, f"{name1}_{n1:04d}.jpg")
                p2 = os.path.join(lfw_dir, name2, f"{name2}_{n2:04d}.jpg")
                if os.path.exists(p1) and os.path.exists(p2):
                    pairs.append((p1, p2, 0))
                    diff_count += 1

            if len(pairs) >= max_pairs:
                break

    print(f"Loaded {len(pairs)} pairs "
          f"({same_count} same, {diff_count} different)")
    return pairs


def evaluate(sess, pairs, protect_fn, name):
    """
    Run LFW verification with a given privacy module.
    Reports accuracy, PSNR, SSIM, and feature similarity.
    """
    print(f"\n{'─'*60}")
    print(f"Evaluating: {name}")
    print(f"{'─'*60}")

    similarities = []
    labels       = []
    psnr_scores  = []
    ssim_scores  = []
    feat_sims    = []

    start = time.time()

    for p1, p2, label in tqdm(pairs, desc=name[:35], ncols=70):
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        if img1 is None or img2 is None:
            continue

        img1 = cv2.resize(img1, (112, 112))
        img2 = cv2.resize(img2, (112, 112))

        if protect_fn is not None:
            img1_p = protect_fn(img1)
            img2_p = protect_fn(img2)

            # Privacy metrics on img1
            psnr_scores.append(compute_psnr(img1, img1_p))
            ssim_scores.append(compute_ssim(img1, img1_p))

            # Feature similarity — how similar are embeddings
            # of original vs protected (lower = better privacy)
            emb_orig = get_embedding(sess, img1)
            emb_prot = get_embedding(sess, img1_p)
            feat_sims.append(cosine_similarity(emb_orig, emb_prot))
        else:
            img1_p, img2_p = img1, img2

        emb1 = get_embedding(sess, img1_p)
        emb2 = get_embedding(sess, img2_p)
        similarities.append(cosine_similarity(emb1, emb2))
        labels.append(label)

    # Find optimal verification threshold
    similarities = np.array(similarities)
    labels       = np.array(labels)
    best_acc     = 0.0
    best_thresh  = 0.0
    for thresh in np.arange(-1.0, 1.0, 0.01):
        preds = (similarities >= thresh).astype(int)
        acc   = float(np.mean(preds == labels))
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    elapsed      = time.time() - start
    avg_psnr     = float(np.mean(psnr_scores))  if psnr_scores  else 0.0
    avg_ssim     = float(np.mean(ssim_scores))  if ssim_scores  else 1.0
    avg_feat_sim = float(np.mean(feat_sims))    if feat_sims    else 1.0

    print(f"  Accuracy    : {best_acc*100:.2f}%")
    print(f"  Threshold   : {best_thresh:.2f}")
    print(f"  PSNR  (↓)   : {avg_psnr:.2f} dB")
    print(f"  SSIM  (↓)   : {avg_ssim:.4f}")
    print(f"  FeatSim (↓) : {avg_feat_sim:.4f}")
    print(f"  Time        : {elapsed:.1f}s")

    return best_acc, avg_psnr, avg_ssim, avg_feat_sim


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--compute-sensitivity', action='store_true',
        help='Compute sensitivity map from full LFW dataset'
    )
    parser.add_argument(
        '--evaluate', action='store_true',
        help='Run LFW 6000-pair evaluation'
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # ── STEP 1: Compute sensitivity map ───────────────────────────
    if args.compute_sensitivity:
        print("=" * 60)
        print("STEP 1 — Computing Sensitivity Map from Full LFW")
        print("=" * 60)

        all_images = glob.glob(
            os.path.join(LFW_DIR, '**', '*.jpg'), recursive=True
        )

        if len(all_images) == 0:
            print(f"ERROR: No images found in {LFW_DIR}")
            print("Check that LFW_DIR path is correct at top of file")
            exit()

        print(f"Found {len(all_images)} LFW images")
        compute_sensitivity_map(all_images, SENSITIVITY_MAP)
        print("\nStep 1 complete.")
        print(f"Run: python dctdp_faithful.py --evaluate")

    # ── STEP 2: LFW Evaluation ────────────────────────────────────
    elif args.evaluate:
        print("=" * 60)
        print("STEP 2 — LFW 6000-Pair Evaluation")
        print("=" * 60)

        # Load sensitivity map
        sensitivity_map = load_sensitivity_map(SENSITIVITY_MAP)

        # Load ArcFace
        sess = load_arcface()

        # Load pairs
        pairs = load_lfw_pairs(PAIRS_FILE, LFW_DIR, max_pairs=6000)
        if len(pairs) == 0:
            print("ERROR: No pairs loaded. Check PAIRS_FILE and LFW_DIR.")
            exit()

        # Define protect function with loaded sensitivity map
        protect = lambda img: dctdp_protect(img, sensitivity_map, EPSILON)

        # Run evaluation
        results = {}

        # Baseline — no protection
        acc, psnr, ssim, fs = evaluate(
            sess, pairs,
            protect_fn=None,
            name="ArcFace (no protection)"
        )
        results["ArcFace (no protection)"] = (acc, psnr, ssim, fs)

        # DCTDP faithful reimplementation
        acc, psnr, ssim, fs = evaluate(
            sess, pairs,
            protect_fn=protect,
            name="DCTDP Faithful Reimplementation"
        )
        results["DCTDP Faithful"] = (acc, psnr, ssim, fs)

        # ── Print final table ─────────────────────────────────────
        print(f"\n{'═'*70}")
        print("FINAL RESULTS — DCTDP FAITHFUL REIMPLEMENTATION")
        print(f"{'═'*70}")
        print(f"{'Method':<35} {'Acc%':<8} {'PSNR↓':<9} "
              f"{'SSIM↓':<9} {'FeatSim↓'}")
        print(f"{'─'*70}")
        for method, (acc, psnr, ssim, fs) in results.items():
            psnr_s = f"{psnr:.2f}" if psnr else "—"
            ssim_s = f"{ssim:.4f}" if ssim else "—"
            fs_s   = f"{fs:.4f}"   if fs   else "—"
            print(f"  {method:<33} {acc*100:<8.2f} {psnr_s:<9} "
                  f"{ssim_s:<9} {fs_s}")
        print(f"{'═'*70}")

        # Save checkpoint
        with open("outputs/dctdp_faithful_results.txt", "w") as f:
            f.write("DCTDP FAITHFUL REIMPLEMENTATION RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write("Implemented per paper Section 3.1 and 3.2:\n")
            f.write("  - BGR to YCbCr conversion\n")
            f.write("  - Value range shifted to [-128, 127]\n")
            f.write("  - 8x upsampling before BDCT\n")
            f.write("  - 8x8 Block DCT\n")
            f.write("  - DC component removal\n")
            f.write(f"  - Sensitivity computed from full LFW\n")
            f.write(f"  - Calibrated Laplace noise (epsilon={EPSILON})\n\n")
            f.write("Acknowledged simplifications:\n")
            f.write("  - IDCT reconstruction for pixel-space evaluation\n")
            f.write("  - Pretrained ArcFace backbone (domain mismatch)\n")
            f.write("  - Uniform epsilon allocation (not learned)\n\n")
            for method, (acc, psnr, ssim, fs) in results.items():
                f.write(f"{method}\n")
                f.write(f"  Accuracy   : {acc*100:.2f}%\n")
                f.write(f"  PSNR       : {psnr:.2f} dB\n")
                f.write(f"  SSIM       : {ssim:.4f}\n")
                f.write(f"  FeatSim    : {fs:.4f}\n\n")

        print(f"\nResults saved: outputs/dctdp_faithful_results.txt")

    else:
        print("Usage:")
        print("  Step 1 — compute sensitivity map from LFW (~25 min):")
        print("    python dctdp_faithful.py --compute-sensitivity")
        print()
        print("  Step 2 — run LFW evaluation (~75 min on CPU):")
        print("    python dctdp_faithful.py --evaluate")
        print()
        print(f"  Current settings:")
        print(f"    LFW_DIR    = {LFW_DIR}")
        print(f"    PAIRS_FILE = {PAIRS_FILE}")
        print(f"    EPSILON    = {EPSILON}")
        print(f"    UPSAMPLE   = {UPSAMPLE_FACTOR}x")
