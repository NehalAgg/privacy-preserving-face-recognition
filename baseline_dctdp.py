# baseline_dctdp.py
# ─────────────────────────────────────────────────────────────────────
# DCTDP Baseline Privacy Module
# Reimplementation of: "Privacy-Preserving Face Recognition with
# Learnable Privacy Budgets in Frequency Domain" (ECCV 2022)
#
# What this does (in plain English):
#   1. Take a face image
#   2. Cut it into 8x8 pixel blocks
#   3. Apply DCT to each block (converts pixels → frequencies)
#   4. DELETE the DC component (the "average brightness" of each block)
#      → This removes the most visually identifiable information
#   5. Add Gaussian noise to the remaining frequency coefficients
#      → This is the "Differential Privacy" part
#   6. Reconstruct back to an image
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import os


# ── CORE FUNCTIONS ───────────────────────────────────────────────────

def apply_block_dct(image_channel, block_size=8):
    """
    Apply 2D DCT to every (block_size x block_size) patch of a single
    grayscale channel. Returns an array of the same shape containing
    DCT coefficients instead of pixel values.
    """
    h, w = image_channel.shape
    # Pad image so it divides evenly into blocks
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(image_channel, ((0, pad_h), (0, pad_w)), mode='reflect')

    ph, pw = padded.shape
    dct_result = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size, j:j+block_size].astype(np.float32)
            # 2D DCT = apply 1D DCT along rows, then columns
            dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            dct_result[i:i+block_size, j:j+block_size] = dct_block

    # Crop back to original size
    return dct_result[:h, :w]


def apply_block_idct(dct_channel, block_size=8):
    """
    Inverse of apply_block_dct. Converts DCT coefficients back to pixels.
    """
    h, w = dct_channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(dct_channel, ((0, pad_h), (0, pad_w)), mode='reflect')

    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            idct_block = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            result[i:i+block_size, j:j+block_size] = idct_block

    return result[:h, :w]


def remove_dc_component(dct_channel, block_size=8):
    """
    Set the DC coefficient (position [0,0] in each block) to zero.
    This is the KEY operation from the base paper — it removes the
    "average brightness" of each block, which carries most of the
    visually identifiable low-frequency information.
    """
    result = dct_channel.copy()
    h, w = result.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # DC component is the top-left coefficient of each block
            result[i, j] = 0.0
    return result


def add_dp_noise(dct_channel, epsilon=1.0, sensitivity=1.0):
    """
    Add Laplace noise (Differential Privacy mechanism) to DCT coefficients.

    epsilon: Privacy budget. LOWER = more noise = more privacy but less accuracy.
             The base paper uses epsilon=1 as default.
    sensitivity: How much a single person's data can change the output.

    This is the "learnable budget" part — in the base paper, a neural
    network learns how much noise to add to each frequency channel.
    Here we use a fixed epsilon for simplicity (still valid baseline).
    """
    noise_scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=noise_scale, size=dct_channel.shape)
    return dct_channel + noise.astype(np.float32)


def dctdp_protect(image_bgr, epsilon=1.0, block_size=8):
    """
    Full DCTDP pipeline on a BGR image (as loaded by OpenCV).
    Returns the protected image as uint8 BGR.

    Steps:
        1. Convert to float, split channels
        2. Block DCT each channel
        3. Remove DC component
        4. Add DP noise
        5. Inverse DCT back to spatial domain
        6. Clip to [0, 255] and return
    """
    img = image_bgr.astype(np.float32)

    protected_channels = []
    for c in range(3):  # B, G, R
        channel = img[:, :, c]

        # Step 2: Block DCT
        dct_ch = apply_block_dct(channel, block_size)

        # Step 3: Remove DC (most visually informative component)
        dct_ch = remove_dc_component(dct_ch, block_size)

        # Step 4: Add Laplace noise (Differential Privacy)
        dct_ch = add_dp_noise(dct_ch, epsilon=epsilon)

        # Step 5: Inverse DCT → back to pixel domain
        restored = apply_block_idct(dct_ch, block_size)

        protected_channels.append(restored)

    protected = np.stack(protected_channels, axis=2)

    # Clip to valid pixel range
    protected = np.clip(protected, 0, 255).astype(np.uint8)
    return protected


# ── METRICS ──────────────────────────────────────────────────────────

def compute_ssim(img1, img2):
    """
    SSIM: Structural Similarity Index.
    Range: 0 (completely different) to 1 (identical).
    For privacy: LOWER is BETTER — means the protected image looks
    nothing like the original.
    Target from our project: SSIM < 0.05
    """
    from skimage.metrics import structural_similarity as ssim
    # Convert BGR to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


def compute_psnr(img1, img2):
    """
    PSNR: Peak Signal-to-Noise Ratio (in dB).
    For privacy: LOWER is BETTER — means more information destroyed.
    """
    return cv2.PSNR(img1, img2)


# ── VISUALISATION ─────────────────────────────────────────────────────

def visualise_result(original, protected, save_path, epsilon):
    """
    Side-by-side plot: original vs protected face.
    Also shows the difference map (what was removed).
    """
    diff = cv2.absdiff(original, protected)
    # Amplify diff for visibility
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)

    ssim_score = compute_ssim(original, protected)
    psnr_score = compute_psnr(original, protected)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'DCTDP Baseline (ε={epsilon})  |  SSIM={ssim_score:.4f}  |  PSNR={psnr_score:.2f}dB',
                 fontsize=13, fontweight='bold')

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Face', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(protected, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Protected (DCTDP)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Difference ×5', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return ssim_score, psnr_score


# ── MAIN: Run on all test images ──────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    test_folder = "test_images"
    image_files = [f for f in os.listdir(test_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("ERROR: No images found in test_images/")
        print("Add some face photos named face_01.jpg, face_02.jpg etc.")
        exit()

    print(f"Found {len(image_files)} images. Running DCTDP baseline...\n")
    print(f"{'Image':<20} {'SSIM (↓ better)':<20} {'PSNR (↓ better)'}")
    print("-" * 55)

    all_ssim = []
    all_psnr = []

    for fname in sorted(image_files):
        img_path = os.path.join(test_folder, fname)
        original = cv2.imread(img_path)

        if original is None:
            print(f"  Skipping {fname} — couldn't read file")
            continue

        # Resize to 112x112 (standard face recognition input size)
        original = cv2.resize(original, (112, 112))

        # Apply DCTDP protection
        protected = dctdp_protect(original, epsilon=1.0)

        # Compute metrics
        ssim_score = compute_ssim(original, protected)
        psnr_score = compute_psnr(original, protected)
        all_ssim.append(ssim_score)
        all_psnr.append(psnr_score)

        # Save side-by-side visualisation
        out_path = os.path.join("outputs", f"dctdp_{fname.split('.')[0]}.png")
        visualise_result(original, protected, out_path, epsilon=1.0)

        print(f"  {fname:<18} {ssim_score:<20.4f} {psnr_score:.2f} dB")

    if all_ssim:
        print("-" * 55)
        print(f"  {'AVERAGE':<18} {np.mean(all_ssim):<20.4f} {np.mean(all_psnr):.2f} dB")
        print(f"\nAll output images saved to: outputs/")
        print(f"\n{'='*55}")
        print(f"BASELINE SUMMARY")
        print(f"  Average SSIM : {np.mean(all_ssim):.4f}  (target for YOUR method: < 0.05)")
        print(f"  Average PSNR : {np.mean(all_psnr):.2f} dB (lower = more private)")
        print(f"{'='*55}")
        print(f"\nStep 1 complete! These are your baseline numbers.")
        print(f"Your novel DWT method will aim to beat (lower) the SSIM score.")