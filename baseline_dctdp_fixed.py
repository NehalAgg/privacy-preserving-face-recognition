# baseline_dctdp_fixed.py
# ─────────────────────────────────────────────────────────────────────
# DCTDP Baseline — FIXED VERSION
# Fixes applied vs original baseline_dctdp.py:
#
#   FIX 1: BGR → YCbCr color space conversion (paper Section 3.1)
#   FIX 2: Value range shifted to [-128, 127] before DCT
#   FIX 3: Per-position sensitivity map computed from actual images
#   FIX 4: Calibrated Laplace noise using sensitivity (not uniform)
#   FIX 5: PSNR used as primary metric (matches paper Figure 7)
#          SSIM kept as secondary metric only
#
# Still simplified vs paper (acknowledged in report):
#   - No 8x upsampling (requires 8x more compute, not feasible on CPU)
#   - No learnable budget allocation (requires GPU training from scratch)
#   - Pixel-space output via IDCT (paper keeps features in freq domain)
#
# Usage:
#   python baseline_dctdp_fixed.py
#   → Computes sensitivity map from test_images/ first run
#   → Saves sensitivity_map.npy for reuse
#   → Runs protection and saves results to outputs/
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim_metric


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — COLOR SPACE CONVERSION (FIX 1 + 2)
# ═══════════════════════════════════════════════════════════════════

def bgr_to_ycbcr_shifted(image_bgr):
    """
    FIX 1+2: Convert BGR → YCbCr and shift to [-128, 127].

    Why YCbCr?
        Y  = luminance (brightness) — carries most face structure
        Cb = blue-difference chroma
        Cr = red-difference chroma
        Separating luminance from color is standard in JPEG and
        in the base paper. DCT on YCbCr channels produces cleaner
        frequency separation than DCT on raw RGB channels.

    Why shift to [-128, 127]?
        DCT assumes zero-mean input. Shifting by -128 centers the
        values around zero, which is required for the DC component
        to correctly represent the block mean.
        The base paper explicitly states this requirement.
    """
    # OpenCV's YCrCb is equivalent to YCbCr for our purposes
    img_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    img_shifted = img_ycbcr.astype(np.float32) - 128.0
    return img_shifted


def ycbcr_shifted_to_bgr(img_shifted):
    """
    Reverse of bgr_to_ycbcr_shifted.
    Shift back → convert YCbCr → BGR.
    """
    img_shifted_back = img_shifted + 128.0
    img_clipped = np.clip(img_shifted_back, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_clipped, cv2.COLOR_YCrCb2BGR)
    return img_bgr


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — BLOCK DCT (same as before, works on shifted YCbCr now)
# ═══════════════════════════════════════════════════════════════════

def apply_block_dct(image_channel, block_size=8):
    """
    Apply 2D DCT to every (block_size x block_size) patch.
    Input channel should already be in shifted [-128,127] range.
    """
    h, w = image_channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(image_channel, ((0, pad_h), (0, pad_w)),
                    mode='reflect')
    ph, pw = padded.shape
    dct_result = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size,
                           j:j+block_size].astype(np.float32)
            dct_block = dct(dct(block, axis=0, norm='ortho'),
                            axis=1, norm='ortho')
            dct_result[i:i+block_size, j:j+block_size] = dct_block

    return dct_result[:h, :w]


def apply_block_idct(dct_channel, block_size=8):
    """Inverse of apply_block_dct."""
    h, w = dct_channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(dct_channel, ((0, pad_h), (0, pad_w)),
                    mode='reflect')
    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            idct_block = idct(idct(block, axis=0, norm='ortho'),
                              axis=1, norm='ortho')
            result[i:i+block_size, j:j+block_size] = idct_block

    return result[:h, :w]


def remove_dc_component(dct_channel, block_size=8):
    """
    Zero out DC coefficient (position [0,0]) in every block.
    This removes the mean brightness of each block — the most
    visually identifiable low-frequency component.
    After this operation, if you do IDCT the image will look dark
    because the mean has been removed.
    """
    result = dct_channel.copy()
    h, w = result.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            result[i, j] = 0.0
    return result


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — SENSITIVITY MAP (FIX 3)
# ═══════════════════════════════════════════════════════════════════

def compute_sensitivity_map(image_folder, block_size=8,
                             save_path='sensitivity_map.npy'):
    """
    FIX 3: Compute per-position DCT sensitivity from a set of images.

    What sensitivity means here:
        For each position (i,j) in the DCT block grid, sensitivity
        = max coefficient value - min coefficient value across all
        images in the dataset.

        High sensitivity at position (i,j) means that coefficient
        varies a lot across different faces — it carries identity
        information. Low sensitivity means it barely changes.

    How the base paper does it:
        They compute this over VGGFace2 (3.31M images).
        We compute it over whatever images are available locally.
        The more images you provide, the better the estimate.

    Why this matters for noise:
        noise_scale[i,j] = sensitivity[i,j] / epsilon
        Positions with high sensitivity get proportionally more noise.
        This is more accurate than our original uniform noise.

    Args:
        image_folder: path to folder containing face images
        block_size:   DCT block size (8, matching paper)
        save_path:    where to save the computed map for reuse

    Returns:
        sensitivity_map: np.array of shape [H, W, 3] — one map per
                         YCbCr channel
    """
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No images found in {image_folder}. "
              f"Using uniform sensitivity.")
        return None

    print(f"Computing sensitivity map from "
          f"{len(image_files)} images...")

    # We'll track min and max at each DCT position across all images
    # Shape will be [H, W, 3] — one per channel
    dct_min = None
    dct_max = None
    valid_count = 0

    for fname in image_files:
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (112, 112))
        # Apply FIX 1+2
        img_ycbcr = bgr_to_ycbcr_shifted(img)

        # DCT each channel
        dct_channels = []
        for c in range(3):
            dct_ch = apply_block_dct(img_ycbcr[:, :, c], block_size)
            dct_channels.append(dct_ch)

        dct_stack = np.stack(dct_channels, axis=2)  # [H, W, 3]

        if dct_min is None:
            dct_min = dct_stack.copy()
            dct_max = dct_stack.copy()
        else:
            dct_min = np.minimum(dct_min, dct_stack)
            dct_max = np.maximum(dct_max, dct_stack)

        valid_count += 1

    if valid_count == 0:
        print("No valid images processed.")
        return None

    # Sensitivity = range of values at each position
    sensitivity_map = dct_max - dct_min

    # Avoid zero sensitivity (would cause division by zero in noise)
    sensitivity_map = np.maximum(sensitivity_map, 1e-6)

    np.save(save_path, sensitivity_map)
    print(f"Sensitivity map computed from {valid_count} images.")
    print(f"Shape: {sensitivity_map.shape}")
    print(f"Mean sensitivity: {sensitivity_map.mean():.4f}")
    print(f"Max  sensitivity: {sensitivity_map.max():.4f}")
    print(f"Saved to: {save_path}")

    return sensitivity_map


def load_or_compute_sensitivity(image_folder, block_size=8,
                                 save_path='sensitivity_map.npy'):
    """
    Load sensitivity map if already computed, else compute it.
    Saves time on repeated runs.
    """
    if os.path.exists(save_path):
        sensitivity_map = np.load(save_path)
        print(f"Loaded sensitivity map from {save_path} "
              f"(shape: {sensitivity_map.shape})")
        return sensitivity_map
    else:
        return compute_sensitivity_map(image_folder, block_size,
                                        save_path)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — CALIBRATED DP NOISE (FIX 4)
# ═══════════════════════════════════════════════════════════════════

def add_calibrated_dp_noise(dct_channel, sensitivity_channel,
                             epsilon=1.0):
    """
    FIX 4: Add per-position calibrated Laplace noise.

    Original code used:
        noise_scale = 1.0 / epsilon   (same everywhere)

    Fixed version uses:
        noise_scale[i,j] = sensitivity[i,j] / epsilon

    This means:
        - High sensitivity positions (carry identity info) →
          large noise scale → more noise added →
          identity info scrambled more aggressively

        - Low sensitivity positions (stable across faces) →
          small noise scale → less noise →
          those positions preserved more accurately

    This is an approximation of the paper's LEARNED budget
    allocation. Instead of learning it via backprop (which needs
    GPU training), we use the sensitivity map as a proxy for
    importance. This is our 'sensitivity-guided budget allocation'
    contribution — a training-free approximation.

    Args:
        dct_channel:        [H, W] DCT coefficients for one channel
        sensitivity_channel:[H, W] sensitivity values for same channel
        epsilon:            privacy budget (lower = more privacy)
    """
    noise_scale = sensitivity_channel / epsilon
    noise = np.random.laplace(
        loc=0.0,
        scale=noise_scale,
        size=dct_channel.shape
    ).astype(np.float32)
    return dct_channel + noise


def add_uniform_dp_noise(dct_channel, epsilon=1.0):
    """
    Original uniform noise — kept for ablation comparison.
    Shows what happens without sensitivity calibration.
    """
    noise = np.random.laplace(
        loc=0.0,
        scale=1.0 / epsilon,
        size=dct_channel.shape
    ).astype(np.float32)
    return dct_channel + noise


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def dctdp_protect_original(image_bgr, epsilon=1.0, block_size=8):
    """
    ORIGINAL method — kept for ablation comparison.
    Uses BGR color space and uniform noise.
    This is what was in baseline_dctdp.py before fixes.
    """
    img = image_bgr.astype(np.float32)
    protected_channels = []
    for c in range(3):
        ch = apply_block_dct(img[:, :, c], block_size)
        ch = remove_dc_component(ch, block_size)
        ch = add_uniform_dp_noise(ch, epsilon)
        ch = apply_block_idct(ch, block_size)
        protected_channels.append(ch)
    protected = np.stack(protected_channels, axis=2)
    return np.clip(protected, 0, 255).astype(np.uint8)


def dctdp_protect_ycbcr(image_bgr, epsilon=1.0, block_size=8):
    """
    FIX 1+2 only: YCbCr + shifted range, but still uniform noise.
    Used in ablation to isolate the effect of color space fix.
    """
    img_ycbcr = bgr_to_ycbcr_shifted(image_bgr)
    protected_channels = []

    for c in range(3):
        ch = apply_block_dct(img_ycbcr[:, :, c], block_size)
        ch = remove_dc_component(ch, block_size)
        ch = add_uniform_dp_noise(ch, epsilon)
        ch = apply_block_idct(ch, block_size)
        protected_channels.append(ch)

    # Stack and convert back to BGR
    protected_ycbcr = np.stack(protected_channels, axis=2)
    return ycbcr_shifted_to_bgr(protected_ycbcr)


def dctdp_protect_full(image_bgr, sensitivity_map,
                        epsilon=1.0, block_size=8):
    """
    FULL FIXED pipeline: YCbCr + sensitivity-calibrated noise.
    This is the closest we can get to the base paper without
    GPU training (fixes 1, 2, 3, 4 applied).

    Args:
        image_bgr:       input face image (BGR uint8)
        sensitivity_map: [H, W, 3] sensitivity map from
                         load_or_compute_sensitivity()
                         Pass None to fall back to uniform noise.
        epsilon:         privacy budget
    """
    img_ycbcr = bgr_to_ycbcr_shifted(image_bgr)

    # Handle case where sensitivity map shape doesn't match
    h, w = image_bgr.shape[:2]
    if sensitivity_map is not None:
        if sensitivity_map.shape[:2] != (h, w):
            # Resize sensitivity map to match image
            sens_resized = cv2.resize(
                sensitivity_map, (w, h),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            sens_resized = sensitivity_map
    else:
        sens_resized = None

    protected_channels = []
    for c in range(3):
        ch = apply_block_dct(img_ycbcr[:, :, c], block_size)
        ch = remove_dc_component(ch, block_size)

        # Use calibrated noise if sensitivity available
        if sens_resized is not None:
            ch = add_calibrated_dp_noise(
                ch, sens_resized[:, :, c], epsilon
            )
        else:
            ch = add_uniform_dp_noise(ch, epsilon)

        ch = apply_block_idct(ch, block_size)
        protected_channels.append(ch)

    protected_ycbcr = np.stack(protected_channels, axis=2)
    return ycbcr_shifted_to_bgr(protected_ycbcr)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — METRICS (FIX 5 — PSNR PRIMARY, SSIM SECONDARY)
# ═══════════════════════════════════════════════════════════════════

def compute_psnr(img1, img2):
    """
    FIX 5: PSNR is the base paper's actual metric (Figure 7).
    Lower PSNR = more signal destroyed = better privacy.
    The paper reports PSNR between original and attacker-recovered
    image. We report it between original and protected image
    as a proxy (we don't implement the full UNet attacker here).
    """
    return cv2.PSNR(img1.astype(np.uint8), img2.astype(np.uint8))


def compute_ssim(img1, img2):
    """
    SSIM kept as secondary metric.
    NOT used in base paper — this is our own addition.
    Clearly label it as such in the report.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_metric(gray1, gray2, full=True)
    return score


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — ABLATION COMPARISON VISUALISATION
# ═══════════════════════════════════════════════════════════════════

def visualise_ablation(original, results_dict, save_path):
    """
    Side-by-side comparison of all DCTDP variants.
    results_dict = {method_name: protected_image}

    Shows visually how each fix changes the output.
    """
    n = len(results_dict) + 1  # +1 for original
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for i, (name, protected) in enumerate(results_dict.items()):
        psnr = compute_psnr(original, protected)
        ssim = compute_ssim(original, protected)
        axes[i+1].imshow(cv2.cvtColor(protected, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(
            f'{name}\nPSNR={psnr:.1f}dB\nSSIM={ssim:.3f}',
            fontsize=9
        )
        axes[i+1].axis('off')

    plt.suptitle(
        'DCTDP Ablation: Effect of Each Fix',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ablation figure saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    TEST_FOLDER = "test_images"
    EPSILON     = 1.0

    image_files = sorted([
        f for f in os.listdir(TEST_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print("ERROR: No images in test_images/")
        exit()

    print(f"Found {len(image_files)} images.\n")

    # ── Step 1: Compute or load sensitivity map ───────────────────
    print("=" * 60)
    print("STEP 1: Sensitivity Map")
    print("=" * 60)
    sensitivity_map = load_or_compute_sensitivity(
        TEST_FOLDER, save_path='sensitivity_map.npy'
    )

    # ── Step 2: Run all variants and compare ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Running all DCTDP variants")
    print("=" * 60)

    header = (f"\n{'Image':<20} {'Method':<30} "
              f"{'PSNR↓':<12} {'SSIM↓'}")
    print(header)
    print("-" * 75)

    all_results = {
        'Original (v1)': [],
        'YCbCr only (v2)': [],
        'Full fixed (v3)': []
    }

    for fname in image_files:
        img_path = os.path.join(TEST_FOLDER, fname)
        original = cv2.imread(img_path)
        if original is None:
            continue
        original = cv2.resize(original, (112, 112))

        # Version 1: Original (no fixes)
        v1 = dctdp_protect_original(original, EPSILON)

        # Version 2: YCbCr fix only
        v2 = dctdp_protect_ycbcr(original, EPSILON)

        # Version 3: Full fix (YCbCr + calibrated sensitivity)
        v3 = dctdp_protect_full(original, sensitivity_map, EPSILON)

        # Metrics for each
        for version_name, protected in [
            ('Original (v1)', v1),
            ('YCbCr only (v2)', v2),
            ('Full fixed (v3)', v3)
        ]:
            psnr = compute_psnr(original, protected)
            ssim = compute_ssim(original, protected)
            all_results[version_name].append((psnr, ssim))
            print(f"  {fname:<18} {version_name:<30} "
                  f"{psnr:<12.2f} {ssim:.4f}")

        # Save ablation figure for first image only
        if fname == image_files[0]:
            visualise_ablation(
                original,
                {
                    'v1: Original\n(BGR+uniform)': v1,
                    'v2: +YCbCr\n(uniform noise)': v2,
                    'v3: Full Fix\n(YCbCr+sensitivity)': v3
                },
                'outputs/dctdp_ablation.png'
            )

    # ── Step 3: Summary table ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Average across all images")
    print("=" * 60)
    print(f"{'Method':<30} {'Avg PSNR↓':<15} {'Avg SSIM↓'}")
    print("-" * 55)

    for method, scores in all_results.items():
        psnrs = [s[0] for s in scores]
        ssims = [s[1] for s in scores]
        print(f"  {method:<28} {np.mean(psnrs):<15.2f} "
              f"{np.mean(ssims):.4f}")

    print("\nNote: Lower PSNR and lower SSIM = better privacy")
    print("Note: v3 uses sensitivity map — if computed from only")
    print("      5 images the map is rough. More images = better.")
    print("\nOutputs saved to outputs/")
    print("sensitivity_map.npy saved for reuse in evaluate_lfw.py")
