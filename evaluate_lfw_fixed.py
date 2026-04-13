# evaluate_lfw_fixed.py
# ─────────────────────────────────────────────────────────────────────
# Full LFW Evaluation — FIXED + EXTENDED VERSION
#
# Changes vs original evaluate_lfw.py:
#
#   FIX 1: PSNR as primary privacy metric (matches base paper Fig 7)
#           SSIM kept as secondary
#
#   FIX 2: Feature Similarity metric added (matches base paper Fig 7)
#           Feed attacker-recovered image through ArcFace,
#           compare embedding to original → lower = better privacy
#
#   FIX 3: All new methods added to evaluation:
#           - DCTDP original (v1)
#           - DCTDP + YCbCr (v2)
#           - DCTDP full fixed (v3)
#           - DWT 1-level (original)
#           - DWT 2-level (new)
#           - DWT + DP hybrid (new)
#           - DWT db2 wavelet (new)
#
#   FIX 4: GPU support via CUDAExecutionProvider
#           Falls back to CPU automatically if GPU unavailable
#
#   FIX 5: Sensitivity map loaded if available
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import os
import time
from tqdm import tqdm
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim_metric
import pywt
import onnxruntime as ort


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def load_arcface(model_path=None, use_gpu=True):
    """
    Load ArcFace ONNX model.
    Tries GPU first, falls back to CPU.
    Works with both original and fine-tuned models.
    """
    if model_path is None:
        model_path = model_path = 'w600k_r50_dwt_finetuned.onnx'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: python -c \"import insightface; "
            f"insightface.app.FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)\""
        )

    providers = []
    if use_gpu:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    sess = ort.InferenceSession(model_path, providers=providers)

    # Report which provider is actually being used
    active = sess.get_providers()[0]
    device = "GPU" if "CUDA" in active else "CPU"
    print(f"ArcFace loaded on {device} ({active})")
    print(f"Model: {os.path.basename(model_path)}")
    return sess


def get_embedding(sess, image_bgr):
    """
    Extract 512-dim face embedding from cropped face image.
    Input: BGR uint8 image (any size — resized internally to 112x112)
    Output: 512-dim float32 numpy array
    """
    img = cv2.resize(image_bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: img})[0][0]


def cosine_similarity(e1, e2):
    """Cosine similarity between two embeddings. Range: -1 to 1."""
    e1 = e1 / (np.linalg.norm(e1) + 1e-8)
    e2 = e2 / (np.linalg.norm(e2) + 1e-8)
    return float(np.dot(e1, e2))


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — ALL PRIVACY MODULES
# (self-contained copies so this file runs standalone)
# ═══════════════════════════════════════════════════════════════════

# ── Block DCT helpers ─────────────────────────────────────────────

def _block_dct(channel, block_size=8):
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)
    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            b = padded[i:i+block_size, j:j+block_size].astype(np.float32)
            result[i:i+block_size, j:j+block_size] = dct(
                dct(b, axis=0, norm='ortho'), axis=1, norm='ortho')
    return result[:h, :w]


def _block_idct(channel, block_size=8):
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)
    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            b = padded[i:i+block_size, j:j+block_size]
            result[i:i+block_size, j:j+block_size] = idct(
                idct(b, axis=0, norm='ortho'), axis=1, norm='ortho')
    return result[:h, :w]


def _remove_dc(channel, block_size=8):
    r = channel.copy()
    for i in range(0, r.shape[0], block_size):
        for j in range(0, r.shape[1], block_size):
            r[i, j] = 0.0
    return r


# ── DCTDP v1: Original ───────────────────────────────────────────

def protect_dctdp_v1(image_bgr, epsilon=1.0):
    """Original DCTDP — BGR + uniform noise."""
    img = image_bgr.astype(np.float32)
    channels = []
    for c in range(3):
        ch = _block_dct(img[:, :, c])
        ch = _remove_dc(ch)
        ch += np.random.laplace(0, 1.0/epsilon, ch.shape).astype(np.float32)
        channels.append(_block_idct(ch))
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


# ── DCTDP v2: + YCbCr ────────────────────────────────────────────

def protect_dctdp_v2(image_bgr, epsilon=1.0):
    """DCTDP with YCbCr conversion fix."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    img -= 128.0
    channels = []
    for c in range(3):
        ch = _block_dct(img[:, :, c])
        ch = _remove_dc(ch)
        ch += np.random.laplace(0, 1.0/epsilon, ch.shape).astype(np.float32)
        channels.append(_block_idct(ch))
    result = np.clip(np.stack(channels, axis=2) + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)


# ── DCTDP v3: + Sensitivity ──────────────────────────────────────

def protect_dctdp_v3(image_bgr, sensitivity_map, epsilon=1.0):
    """DCTDP full fix: YCbCr + sensitivity-calibrated noise."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    img -= 128.0
    h, w = image_bgr.shape[:2]

    if sensitivity_map is not None and sensitivity_map.shape[:2] != (h, w):
        sensitivity_map = cv2.resize(sensitivity_map, (w, h))

    channels = []
    for c in range(3):
        ch = _block_dct(img[:, :, c])
        ch = _remove_dc(ch)
        if sensitivity_map is not None:
            noise_scale = sensitivity_map[:, :, c] / epsilon
            ch += np.random.laplace(0, noise_scale, ch.shape).astype(np.float32)
        else:
            ch += np.random.laplace(0, 1.0/epsilon, ch.shape).astype(np.float32)
        channels.append(_block_idct(ch))

    result = np.clip(np.stack(channels, axis=2) + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)


# ── DWT helpers ───────────────────────────────────────────────────

def _permute(subband, key, band_id):
    flat = subband.flatten()
    rng = np.random.default_rng(seed=key * 1000 + band_id)
    idx = rng.permutation(len(flat))
    return flat[idx].reshape(subband.shape)


# ── DWT v1: 1-level (original) ───────────────────────────────────

def protect_dwt_1l(image_bgr, secret_key=42, wavelet='haar'):
    """Original 1-level DWT permutation."""
    img = image_bgr.astype(np.float32)
    channels = []
    for c in range(3):
        LL, (LH, HL, HH) = pywt.dwt2(img[:, :, c], wavelet)
        coeffs = (
            _permute(LL, secret_key, 3),
            (_permute(LH, secret_key, 0),
             _permute(HL, secret_key, 1),
             _permute(HH, secret_key, 2))
        )
        channels.append(pywt.idwt2(coeffs, wavelet))
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


# ── DWT v2: 2-level (new) ────────────────────────────────────────

def protect_dwt_2l(image_bgr, secret_key=42, wavelet='haar'):
    """2-level DWT — recurses into LL sub-band."""
    img = image_bgr.astype(np.float32)
    channels = []
    for c in range(3):
        LL1, (LH1, HL1, HH1) = pywt.dwt2(img[:, :, c], wavelet)
        LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, wavelet)

        LL1_r = pywt.idwt2((
            _permute(LL2,  secret_key, 10),
            (_permute(LH2, secret_key, 11),
             _permute(HL2, secret_key, 12),
             _permute(HH2, secret_key, 13))
        ), wavelet)

        full = pywt.idwt2((
            LL1_r,
            (_permute(LH1, secret_key, 0),
             _permute(HL1, secret_key, 1),
             _permute(HH1, secret_key, 2))
        ), wavelet)
        channels.append(full)
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


# ── DWT v3: Hybrid + DP (new) ────────────────────────────────────

def protect_dwt_dp(image_bgr, secret_key=42,
                   epsilon=1.0, wavelet='haar'):
    """Hybrid: DWT permutation + Laplace DP noise on detail bands."""
    img = image_bgr.astype(np.float32)
    channels = []
    noise_scale = 1.0 / epsilon
    for c in range(3):
        LL, (LH, HL, HH) = pywt.dwt2(img[:, :, c], wavelet)
        LL_p  = _permute(LL, secret_key, 3)
        LH_p  = _permute(LH, secret_key, 0) + np.random.laplace(
            0, noise_scale, LH.shape).astype(np.float32)
        HL_p  = _permute(HL, secret_key, 1) + np.random.laplace(
            0, noise_scale, HL.shape).astype(np.float32)
        HH_p  = _permute(HH, secret_key, 2) + np.random.laplace(
            0, noise_scale, HH.shape).astype(np.float32)
        channels.append(pywt.idwt2((LL_p, (LH_p, HL_p, HH_p)), wavelet))
    return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)


# ─── DWT v4: db2 wavelet (new) ───────────────────────────────────

def protect_dwt_db2(image_bgr, secret_key=42):
    """1-level DWT with Daubechies-2 wavelet instead of Haar."""
    return protect_dwt_1l(image_bgr, secret_key, wavelet='db2')


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — METRICS (FIX 1 + 2)
# ═══════════════════════════════════════════════════════════════════

def compute_psnr(img1, img2):
    """
    FIX 1: PSNR — primary privacy metric matching base paper Figure 7.
    Lower = more signal destroyed = better privacy.
    """
    return cv2.PSNR(img1.astype(np.uint8), img2.astype(np.uint8))


def compute_ssim(img1, img2):
    """Secondary metric. Not in base paper — our addition."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_metric(gray1, gray2, full=True)
    return score


def compute_feature_similarity(sess, img_original, img_recovered):
    """
    FIX 2: Feature Similarity — second privacy metric in base paper Fig 7.

    Process:
        1. Get ArcFace embedding of original face
        2. Get ArcFace embedding of attacker's recovered/protected face
        3. Compute cosine similarity between embeddings

    Lower feature similarity = attacker's recovered face has a more
    different embedding from original = harder to use for recognition
    = better privacy at the feature level.

    This is stronger than PSNR because a face can look different
    visually but still fool a face recognizer. Feature similarity
    tests exactly this.
    """
    emb_orig = get_embedding(sess, img_original)
    emb_recv = get_embedding(sess, img_recovered)
    return cosine_similarity(emb_orig, emb_recv)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — LFW DATASET LOADER
# ═══════════════════════════════════════════════════════════════════

def load_lfw_pairs(pairs_file, lfw_dir, max_pairs=6000):
    """Load LFW standard 6000 pairs from CSV file."""
    import csv
    pairs = []
    same_count = 0
    diff_count = 0

    with open(pairs_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            # Strip empty trailing fields from trailing commas
            row = [x for x in row if x.strip() != '']

            if len(row) == 3:
                # Same person pair: name, n1, n2
                name = row[0]
                try:
                    n1, n2 = int(row[1]), int(row[2])
                except ValueError:
                    continue
                img1 = os.path.join(
                    lfw_dir, name, f"{name}_{n1:04d}.jpg")
                img2 = os.path.join(
                    lfw_dir, name, f"{name}_{n2:04d}.jpg")
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, 1))
                    same_count += 1

            elif len(row) == 4:
                # Different person pair: name1, n1, name2, n2
                name1, name2 = row[0], row[2]
                try:
                    n1, n2 = int(row[1]), int(row[3])
                except ValueError:
                    continue
                img1 = os.path.join(
                    lfw_dir, name1, f"{name1}_{n1:04d}.jpg")
                img2 = os.path.join(
                    lfw_dir, name2, f"{name2}_{n2:04d}.jpg")
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, 0))
                    diff_count += 1

            if len(pairs) >= max_pairs:
                break

    print(f"Loaded {len(pairs)} pairs "
          f"({same_count} same, {diff_count} different)")
    return pairs

# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def evaluate_method(sess, pairs, protect_fn,
                    method_name, max_pairs=6000):
    """
    Evaluate one privacy method on LFW pairs.

    For each pair:
      1. Load both images
      2. Apply protect_fn to both
      3. Get ArcFace embeddings
      4. Cosine similarity → predict same/different
      5. Track PSNR, SSIM, Feature Similarity

    Returns:
        accuracy:           best LFW verification accuracy
        avg_psnr:           average PSNR (primary privacy metric)
        avg_ssim:           average SSIM (secondary)
        avg_feat_sim:       average feature similarity (privacy)
    """
    print(f"\n{'─'*60}")
    print(f"Evaluating: {method_name}")
    print(f"{'─'*60}")

    similarities = []
    labels       = []
    psnr_scores  = []
    ssim_scores  = []
    feat_sims    = []
    skipped      = 0

    start_time = time.time()

    for img1_path, img2_path, label in tqdm(
        pairs[:max_pairs], desc=method_name[:30], ncols=70
    ):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            skipped += 1
            continue

        img1 = cv2.resize(img1, (112, 112))
        img2 = cv2.resize(img2, (112, 112))

        # Apply privacy protection
        if protect_fn is not None:
            img1_p = protect_fn(img1)
            img2_p = protect_fn(img2)
        else:
            img1_p, img2_p = img1, img2

        # Privacy metrics (on img1 only for efficiency)
        if protect_fn is not None:
            psnr_scores.append(compute_psnr(img1, img1_p))
            ssim_scores.append(compute_ssim(img1, img1_p))
            feat_sims.append(
                compute_feature_similarity(sess, img1, img1_p)
            )

        # Recognition: get embeddings and compare
        emb1 = get_embedding(sess, img1_p)
        emb2 = get_embedding(sess, img2_p)
        similarities.append(cosine_similarity(emb1, emb2))
        labels.append(label)

    # Find optimal threshold
    similarities = np.array(similarities)
    labels       = np.array(labels)
    best_acc     = 0.0
    best_thresh  = 0.0
    for thresh in np.arange(-1.0, 1.0, 0.01):
        preds = (similarities >= thresh).astype(int)
        acc   = float(np.mean(preds == labels))
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    elapsed = time.time() - start_time
    avg_psnr     = float(np.mean(psnr_scores))  if psnr_scores  else 0.0
    avg_ssim     = float(np.mean(ssim_scores))  if ssim_scores  else 1.0
    avg_feat_sim = float(np.mean(feat_sims))    if feat_sims    else 1.0

    print(f"  Accuracy       : {best_acc*100:.2f}%")
    print(f"  Threshold      : {best_thresh:.2f}")
    print(f"  Avg PSNR  (↓)  : {avg_psnr:.2f} dB")
    print(f"  Avg SSIM  (↓)  : {avg_ssim:.4f}")
    print(f"  Feat Sim  (↓)  : {avg_feat_sim:.4f}")
    print(f"  Skipped        : {skipped}")
    print(f"  Time           : {elapsed:.1f}s")

    return best_acc, avg_psnr, avg_ssim, avg_feat_sim


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    LFW_DIR = "lfw-deepfunneled/lfw-deepfunneled"
    PAIRS_FILE = "pairs.csv"
    SECRET_KEY = 42

    # ── Validate paths ────────────────────────────────────────────
    for path, name in [(LFW_DIR, "LFW dir"), (PAIRS_FILE, "pairs.csv")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at '{path}'")
            exit()

    # ── Load model ────────────────────────────────────────────────
    sess = load_arcface(use_gpu=True)

    # ── Load sensitivity map if available ────────────────────────
    sensitivity_map = None
    if os.path.exists('sensitivity_map.npy'):
        sensitivity_map = np.load('sensitivity_map.npy')
        print(f"Sensitivity map loaded: {sensitivity_map.shape}")
    else:
        print("No sensitivity map found — DCTDP v3 will use uniform noise")
        print("Run baseline_dctdp_fixed.py first to generate it\n")

    # ── Load pairs ────────────────────────────────────────────────
    pairs = load_lfw_pairs(PAIRS_FILE, LFW_DIR, max_pairs=6000)
    if len(pairs) == 0:
        print("ERROR: No pairs loaded.")
        exit()

    # ── Define all methods ────────────────────────────────────────
    methods = {
        # No protection — upper bound
        #'ArcFace (no protection)': None,

        # DCTDP versions
        #'DCTDP v1 (original BGR)':
        #    lambda img: protect_dctdp_v1(img, epsilon=1.0),
        #'DCTDP v2 (+YCbCr)':
        #    lambda img: protect_dctdp_v2(img, epsilon=1.0),
        #'DCTDP v3 (+YCbCr +Sensitivity)':
        #    lambda img: protect_dctdp_v3(img, sensitivity_map, epsilon=1.0),

        # DWT versions
        'DWT 1-level Haar (original)':
            lambda img: protect_dwt_1l(img, SECRET_KEY, 'haar'),
        'DWT 2-level Haar (new)':
            lambda img: protect_dwt_2l(img, SECRET_KEY, 'haar'),
        'DWT 1L + DP hybrid (new)':
            lambda img: protect_dwt_dp(img, SECRET_KEY, 1.0, 'haar'),
        'DWT 1-level db2 (new)':
            lambda img: protect_dwt_db2(img, SECRET_KEY),
    }

    # ── Run evaluation ────────────────────────────────────────────
    results = {}
    for method_name, protect_fn in methods.items():
        acc, psnr, ssim, feat_sim = evaluate_method(
            sess, pairs, protect_fn, method_name
        )
        results[method_name] = {
            'accuracy': acc,
            'psnr': psnr,
            'ssim': ssim,
            'feat_sim': feat_sim,
            'revocable': 'Yes' if 'DWT' in method_name else 'No'
        }

        # Save after every method — safety checkpoint
        with open("outputs/lfw_results_checkpoint.txt", "w") as f:
            f.write("CHECKPOINT RESULTS\n")
            f.write("=" * 80 + "\n\n")
            for m, r in results.items():
                f.write(
                    f"{m}\n"
                    f"  Accuracy : {r['accuracy']*100:.2f}%\n"
                    f"  PSNR     : {r['psnr']:.2f} dB\n"
                    f"  SSIM     : {r['ssim']:.4f}\n"
                    f"  FeatSim  : {r['feat_sim']:.4f}\n"
                    f"  Revocable: {r['revocable']}\n\n"
                )
        print(f"  Checkpoint saved → outputs/lfw_results_checkpoint.txt")
    
    # ── Print final table ─────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"FINAL ABLATION TABLE — LFW 6000 PAIRS")
    print(f"{'═'*80}")
    print(f"{'Method':<35} {'Acc%':<8} {'PSNR↓':<9} "
          f"{'SSIM↓':<9} {'FeatSim↓':<11} {'Revoc'}")
    print(f"{'─'*80}")

    for method, r in results.items():
        acc_str  = f"{r['accuracy']*100:.2f}"
        psnr_str = f"{r['psnr']:.2f}" if r['psnr'] else "—"
        ssim_str = f"{r['ssim']:.4f}" if r['ssim'] else "—"
        feat_str = f"{r['feat_sim']:.4f}" if r['feat_sim'] else "—"
        print(f"  {method:<33} {acc_str:<8} {psnr_str:<9} "
              f"{ssim_str:<9} {feat_str:<11} {r['revocable']}")

    print(f"{'═'*80}")
    print("\nNote: PSNR and FeatSim are PRIMARY metrics (match base paper)")
    print("Note: SSIM is secondary (our addition)")
    print("Note: Lower PSNR, SSIM, FeatSim = stronger privacy")

    # ── Save results ──────────────────────────────────────────────
    with open("outputs/lfw_results_full.txt", "w") as f:
        f.write("FULL ABLATION RESULTS — LFW 6000 PAIRS\n")
        f.write("=" * 80 + "\n\n")
        for method, r in results.items():
            f.write(
                f"{method}\n"
                f"  Accuracy : {r['accuracy']*100:.2f}%\n"
                f"  PSNR     : {r['psnr']:.2f} dB\n"
                f"  SSIM     : {r['ssim']:.4f}\n"
                f"  FeatSim  : {r['feat_sim']:.4f}\n"
                f"  Revocable: {r['revocable']}\n\n"
            )

    print("\nResults saved: outputs/lfw_results_full.txt")
