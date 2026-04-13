# evaluate_lfw.py
# ─────────────────────────────────────────────────────────────────────
# Full LFW Evaluation
# Runs all three methods on the standard 6000-pair LFW benchmark:
#   1. No protection (ArcFace baseline)
#   2. DCTDP baseline (your reimplementation)
#   3. DWT + Key Permutation (your novel method)
#
# Produces the final results table for your report.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import os
import time
from tqdm import tqdm
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim_metric
import pywt
import insightface
from insightface.app import FaceAnalysis


# ── LOAD ARCFACE MODEL ────────────────────────────────────────────────

def load_arcface():
    """Load only the recognition model directly, skip face detector."""
    import onnxruntime as ort
    import os
    # Path where insightface already downloaded the model
    model_path = os.path.expanduser(
        "~/.insightface/models/buffalo_l/w600k_r50.onnx"
    )
    print(f"Loading recognition model from: {model_path}")
    sess = ort.InferenceSession(model_path,
                                providers=['CPUExecutionProvider'])
    print("ArcFace loaded.\n")
    return sess


def get_embedding(sess, image_bgr):
    """
    Extract embedding directly from cropped face image.
    Bypasses face detector — works perfectly since LFW images
    are already aligned and cropped to face.
    """
    # Resize to exactly 112x112 (model input size)
    img = cv2.resize(image_bgr, (112, 112))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize to [-1, 1]
    img = (img.astype(np.float32) - 127.5) / 127.5
    # Reshape to (1, 3, 112, 112)
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    # Run inference
    input_name = sess.get_inputs()[0].name
    embedding = sess.run(None, {input_name: img})[0][0]
    return embedding


def cosine_similarity(e1, e2):
    """Cosine similarity between two embeddings. Range: -1 to 1."""
    e1 = e1 / (np.linalg.norm(e1) + 1e-8)
    e2 = e2 / (np.linalg.norm(e2) + 1e-8)
    return float(np.dot(e1, e2))


# ── DCTDP PRIVACY MODULE (from baseline_dctdp.py) ────────────────────

def apply_block_dct(channel, block_size=8):
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)
    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size, j:j+block_size].astype(np.float32)
            result[i:i+block_size, j:j+block_size] = dct(
                dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    return result[:h, :w]

def apply_block_idct(channel, block_size=8):
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape
    result = np.zeros_like(padded, dtype=np.float32)
    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            result[i:i+block_size, j:j+block_size] = idct(
                idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    return result[:h, :w]

def remove_dc(channel, block_size=8):
    result = channel.copy()
    for i in range(0, result.shape[0], block_size):
        for j in range(0, result.shape[1], block_size):
            result[i, j] = 0.0
    return result

def dctdp_protect(image_bgr, epsilon=1.0):
    img = image_bgr.astype(np.float32)
    channels = []
    for c in range(3):
        ch = apply_block_dct(img[:, :, c])
        ch = remove_dc(ch)
        ch += np.random.laplace(0, 1.0/epsilon, ch.shape).astype(np.float32)
        ch = apply_block_idct(ch)
        channels.append(ch)
    out = np.stack(channels, axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── DWT PRIVACY MODULE (from dwt_permutation.py) ─────────────────────

def permute_subband(subband, secret_key, band_id=0):
    flat = subband.flatten()
    rng = np.random.default_rng(seed=secret_key * 1000 + band_id)
    indices = rng.permutation(len(flat))
    return flat[indices].reshape(subband.shape)

def dwt_protect(image_bgr, secret_key=42, wavelet='haar'):
    img = image_bgr.astype(np.float32)
    channels = []
    for c in range(3):
        LL, (LH, HL, HH) = pywt.dwt2(img[:, :, c], wavelet)
        coeffs = (
            permute_subband(LL, secret_key, 3),
            (permute_subband(LH, secret_key, 0),
             permute_subband(HL, secret_key, 1),
             permute_subband(HH, secret_key, 2))
        )
        channels.append(pywt.idwt2(coeffs, wavelet))
    out = np.stack(channels, axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── LFW DATASET LOADER ────────────────────────────────────────────────

def load_lfw_pairs(pairs_file, lfw_dir, max_pairs=6000):
    import csv
    pairs = []
    same_count = 0
    diff_count = 0

    with open(pairs_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row
        print(f"CSV columns: {header}")

        for row in reader:
            if len(row) >= 3 and (len(row) == 3 or row[3] == ''):
                # Same person pair
                name = row[0]
                try:
                    n1 = int(row[1])
                    n2 = int(row[2])
                except:
                    continue
                img1 = os.path.join(lfw_dir, name, f"{name}_{n1:04d}.jpg")
                img2 = os.path.join(lfw_dir, name, f"{name}_{n2:04d}.jpg")
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, 1))
                    same_count += 1

            elif len(row) == 4 and row[3] != '':
                # Different people pair
                name1 = row[0]
                name2 = row[2]
                try:
                    n1 = int(row[1])
                    n2 = int(row[3])
                except:
                    continue
                img1 = os.path.join(lfw_dir, name1, f"{name1}_{n1:04d}.jpg")
                img2 = os.path.join(lfw_dir, name2, f"{name2}_{n2:04d}.jpg")
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, 0))
                    diff_count += 1

            elif len(row) == 4:
                # Different people: name1, imagenum1, name2, imagenum2
                name1 = row[0]
                name2 = row[2]
                try:
                    n1 = int(row[1])
                    n2 = int(row[3])
                except:
                    continue
                img1 = os.path.join(lfw_dir, name1, f"{name1}_{n1:04d}.jpg")
                img2 = os.path.join(lfw_dir, name2, f"{name2}_{n2:04d}.jpg")
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, 0))
                    diff_count += 1

            if len(pairs) >= max_pairs:
                break

    print(f"Loaded {len(pairs)} pairs "
          f"({same_count} same-person, {diff_count} different)")
    return pairs


# ── EVALUATE ONE METHOD ───────────────────────────────────────────────

def evaluate_method(app, pairs, protect_fn, method_name, max_pairs=6000):
    """
    Run face verification on LFW pairs using a given privacy method.

    For each pair:
      1. Load both images
      2. Apply protection (or none)
      3. Extract ArcFace embedding
      4. Compute cosine similarity
      5. Threshold similarity to predict same/different

    Returns accuracy, optimal threshold, and average SSIM.
    """
    print(f"\nEvaluating: {method_name}")
    print("-" * 50)

    similarities = []
    labels       = []
    ssim_scores  = []
    skipped      = 0

    pairs_to_run = pairs[:max_pairs]

    for img1_path, img2_path, label in tqdm(pairs_to_run, desc=method_name):
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            skipped += 1
            continue

        # Resize to standard face recognition size
        img1 = cv2.resize(img1, (112, 112))
        img2 = cv2.resize(img2, (112, 112))

        # Apply privacy protection
        if protect_fn is not None:
            img1_p = protect_fn(img1)
            img2_p = protect_fn(img2)
        else:
            img1_p = img1
            img2_p = img2

        # Compute SSIM for privacy measurement (only on img1)
        if protect_fn is not None:
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(img1_p, cv2.COLOR_BGR2GRAY)
            score, _ = ssim_metric(g1, g2, full=True)
            ssim_scores.append(score)

        # Get embeddings
        emb1 = get_embedding(app, img1_p)
        emb2 = get_embedding(app, img2_p)
        if emb1 is None or emb2 is None:
            skipped += 1
            continue

        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)
        labels.append(label)

    similarities = np.array(similarities)
    labels       = np.array(labels)

    # Find optimal threshold by sweeping
    best_acc   = 0
    best_thresh = 0
    for thresh in np.arange(-1.0, 1.0, 0.01):
        preds = (similarities >= thresh).astype(int)
        acc   = np.mean(preds == labels)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh

    avg_ssim = float(np.mean(ssim_scores)) if ssim_scores else 1.0

    print(f"  Pairs evaluated : {len(similarities)}")
    print(f"  Skipped         : {skipped}")
    print(f"  Best accuracy   : {best_acc*100:.2f}%")
    print(f"  Best threshold  : {best_thresh:.2f}")
    print(f"  Avg SSIM        : {avg_ssim:.4f}")

    return best_acc, best_thresh, avg_ssim


# ── MAIN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Paths — adjust if your folder is named differently ───────────
    LFW_DIR    = "lfw-deepfunneled"
    PAIRS_FILE = "pairs.csv"
    SECRET_KEY = 42

    # ── Check files exist ─────────────────────────────────────────────
    if not os.path.exists(LFW_DIR):
        print(f"ERROR: '{LFW_DIR}' folder not found.")
        print("Download lfw-funneled.tgz from:")
        print("  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz")
        print("Extract it into cv_project/ and run again.")
        exit()

    if not os.path.exists(PAIRS_FILE):
        print(f"ERROR: '{PAIRS_FILE}' not found.")
        print("Download pairs.txt from:")
        print("  http://vis-www.cs.umass.edu/lfw/pairs.txt")
        exit()

    # ── Load model and pairs ──────────────────────────────────────────
    app   = load_arcface()
    pairs = load_lfw_pairs(PAIRS_FILE, LFW_DIR, max_pairs=6000)

    if len(pairs) == 0:
        print("ERROR: No pairs loaded. Check LFW_DIR path.")
        exit()

    # ── Run all three methods ─────────────────────────────────────────
    results = {}

    # Method 1: No protection (upper bound on accuracy)
    acc, thresh, avg_ssim = evaluate_method(
        app, pairs,
        protect_fn=None,
        method_name="ArcFace (No Protection)"
    )
    results["ArcFace (No Protection)"] = (acc, avg_ssim)

    # Method 2: DCTDP baseline
    acc, thresh, avg_ssim = evaluate_method(
        app, pairs,
        protect_fn=dctdp_protect,
        method_name="DCTDP Baseline"
    )
    results["DCTDP Baseline"] = (acc, avg_ssim)

    # Method 3: Your novel DWT + key permutation
    acc, thresh, avg_ssim = evaluate_method(
        app, pairs,
        protect_fn=lambda img: dwt_protect(img, secret_key=SECRET_KEY),
        method_name="DWT + Key Permutation (Ours)"
    )
    results["DWT + Key Permutation (Ours)"] = (acc, avg_ssim)

    # ── Print final results table ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"FINAL RESULTS TABLE — LFW 6000 PAIRS")
    print(f"{'='*65}")
    print(f"{'Method':<35} {'LFW Acc %':<15} {'Avg SSIM':<12} {'Revocable'}")
    print(f"{'-'*65}")
    for method, (acc, ssim) in results.items():
        revocable = "Yes" if "Ours" in method else "No"
        print(f"  {method:<33} {acc*100:<15.2f} {ssim:<12.4f} {revocable}")
    print(f"{'='*65}")

    # Save results to text file
    with open("outputs/lfw_results.txt", "w") as f:
        f.write("FINAL LFW RESULTS\n")
        f.write("="*65 + "\n")
        for method, (acc, ssim) in results.items():
            f.write(f"{method}: Acc={acc*100:.2f}%  SSIM={ssim:.4f}\n")

    print(f"\nResults saved to outputs/lfw_results.txt")
    print(f"\nStep 4 complete! These are your final report numbers.")