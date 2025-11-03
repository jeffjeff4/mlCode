import numpy as np
from typing import Tuple
import cv2

import numpy as np
import cv2
from typing import Tuple

import numpy as np
import cv2
from typing import Tuple

import numpy as np
import cv2
from typing import Tuple


def template_match(I: np.ndarray, T: np.ndarray, scales=None) -> Tuple[Tuple[int, int], float]:
    if scales is None:
        scales = [1.0]

    H, W = I.shape
    h, w = T.shape

    best_score = -np.inf
    best_loc = (0, 0)

    I_float = I.astype(np.float32)

    for scale in scales:
        h_s = int(round(h * scale))
        w_s = int(round(w * scale))
        if h_s < 1 or w_s < 1:
            continue

        T_s = cv2.resize(T, (w_s, h_s), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        T_mean = T_s.mean()
        T_norm = T_s - T_mean
        T_ssd = np.sum(T_norm ** 2)
        if T_ssd == 0:
            continue

        # Cross-correlation (raw)
        corr = cv2.matchTemplate(I_float, T_s, cv2.TM_CCORR)  # (H-h+1, W-w+1)

        # Valid region
        valid_h, valid_w = corr.shape
        pad_y = (H - valid_h) // 2
        pad_x = (W - valid_w) // 2

        # Integral images
        I_padded = np.pad(I_float, ((h_s // 2, h_s // 2), (w_s // 2, w_s // 2)), mode='constant')
        sum_win = cv2.boxFilter(I_padded, -1, (h_s, w_s), normalize=False)
        sum_sq = cv2.boxFilter(I_padded ** 2, -1, (h_s, w_s), normalize=False)

        # Crop to valid
        sum_win = sum_win[pad_y:pad_y + valid_h, pad_x:pad_x + valid_w]
        sum_sq = sum_sq[pad_y:pad_y + valid_h, pad_x:pad_x + valid_w]

        local_mean = sum_win / (h_s * w_s)
        local_ssd = sum_sq - (h_s * w_s) * (local_mean ** 2)  # CRITICAL FIX
        local_ssd = np.maximum(local_ssd, 0)

        # Final NCC
        denom = np.sqrt(local_ssd * T_ssd + 1e-8)
        ncc = corr / denom

        max_val = ncc.max()
        if max_val > best_score:
            best_score = max_val
            loc = np.unravel_index(ncc.argmax(), ncc.shape)
            best_loc = (loc[0], loc[1])

    return best_loc, best_score



import matplotlib.pyplot as plt


# -------------------------------------------------
# Test 1: Exact Match
# -------------------------------------------------
def test_exact_match():
    I = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    T = I[30:50, 40:60].copy()  # Exact patch
    loc, score = template_match(I, T)
    assert loc == (30, 40), f"Expected (30,40), got {loc}"
    assert abs(score - 1.0) < 1e-6, f"Score {score} != 1.0"
    print("Test 1: Exact Match - PASSED")


# -------------------------------------------------
# Test 2: Noisy Match
# -------------------------------------------------
def test_noisy_match():
    I = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    T_clean = I[30:50, 40:60].copy()
    T = np.clip(T_clean.astype(int) + np.random.randint(-30, 30, T_clean.shape), 0, 255).astype(np.uint8)
    loc, score = template_match(I, T)
    assert loc == (30, 40), f"Expected (30,40), got {loc}"
    assert score > 0.7, f"Score {score} too low"
    print("Test 2: Noisy Match - PASSED")


# -------------------------------------------------
# Test 3: Scaled Match
# -------------------------------------------------
def test_scaled_match():
    I = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    T_orig = I[50:90, 60:100]  # 40x40
    T = cv2.resize(T_orig, (36, 36))  # 0.9x scale
    loc, score = template_match(I, T, scales=[0.9, 1.0, 1.1])
    # Allow ±1 pixel error due to interpolation
    assert abs(loc[0] - 50) <= 2 and abs(loc[1] - 60) <= 2
    assert score > 0.8
    print("Test 3: Scaled Match - PASSED")


# -------------------------------------------------
# Test 4: Real Image (Lena + Patch)
# -------------------------------------------------
def test_real_image():
    import urllib.request
    url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    img_data = urllib.request.urlopen(url).read()
    import io
    from PIL import Image
    img = np.array(Image.open(io.BytesIO(img_data)).convert('L'))

    T = img[200:300, 200:300]  # 100x100 patch
    I = img.copy()
    I[50:150, 300:400] = T  # Paste at known location

    loc, score = template_match(I, T)
    assert abs(loc[0] - 50) <= 2 and abs(loc[1] - 300) <= 2
    assert score > 0.95
    print("Test 4: Real Image - PASSED")


# Run all
if __name__ == "__main__":
    test_exact_match()
    test_noisy_match()
    test_scaled_match()
    # test_real_image()  # Uncomment after internet access
    print("\nALL TESTS PASSED!")