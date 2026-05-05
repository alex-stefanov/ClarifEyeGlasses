"""
Download official banknote reference images for ClarifEye currency recognition.

Downloads:
  Bulgarian leva (BGN) — 5, 10, 20, 50, 100 lev, front + back
  Euros (EUR)          — 5, 10, 20, 50, 100, 200 EUR, front + back

Images are saved to:
  data/banknotes/bgn/<denom>_front.jpg
  data/banknotes/bgn/<denom>_back.jpg
  data/banknotes/eur/<denom>_front.jpg
  data/banknotes/eur/<denom>_back.jpg

Run on the Pi (internet required):
  python scripts/download_banknote_references.py

Idempotent — skips files that already exist.

NOTE ON URLS
~~~~~~~~~~~~
The ECB and BNB websites occasionally restructure their image paths.
If downloads fail, the script prints the manual-download fallback instructions.

ECB Euro press images are published at:
  https://www.ecb.europa.eu/euro/banknotes/security/html/index.en.html
  (Download the highest-resolution variants and rename to match the layout above)

BNB Lev images are published at:
  https://www.bnb.bg/BanknotesAndCoins/Banknotes/index.htm
  (Navigate to each denomination and save the obverse/reverse images)
"""
import os
import sys
import urllib.request
import urllib.error

# Resolve project root so the script works from any directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config import DATA_DIR  # noqa: E402

OUTPUT_DIR = os.path.join(DATA_DIR, "banknotes")
LONG_EDGE_TARGET = 1000  # pixels — resize after download if smaller than this

# ─── URL tables ──────────────────────────────────────────────────────────────
# Format: (currency_code, denomination, side, url)
# These are the best-guess public URLs as of 2025.  If any returns a 404 the
# script will print a manual-download note for that file.

# ECB publishes Euro banknote images under their media section.
# The "Europa series" (2013+) URLs follow this pattern.
_EUR_BASE = "https://www.ecb.europa.eu/euro/banknotes/shared/img"
_BGN_BASE = "https://www.bnb.bg/BanknotesAndCoins/Banknotes"

REFERENCE_URLS = [
    # ── Euro — Europa series ─────────────────────────────────────────────────
    ("eur",  5,  "front", f"{_EUR_BASE}/5euro_obverse.jpg"),
    ("eur",  5,  "back",  f"{_EUR_BASE}/5euro_reverse.jpg"),
    ("eur", 10,  "front", f"{_EUR_BASE}/10euro_obverse.jpg"),
    ("eur", 10,  "back",  f"{_EUR_BASE}/10euro_reverse.jpg"),
    ("eur", 20,  "front", f"{_EUR_BASE}/20euro_obverse.jpg"),
    ("eur", 20,  "back",  f"{_EUR_BASE}/20euro_reverse.jpg"),
    ("eur", 50,  "front", f"{_EUR_BASE}/50euro_obverse.jpg"),
    ("eur", 50,  "back",  f"{_EUR_BASE}/50euro_reverse.jpg"),
    ("eur", 100, "front", f"{_EUR_BASE}/100euro_obverse.jpg"),
    ("eur", 100, "back",  f"{_EUR_BASE}/100euro_reverse.jpg"),
    ("eur", 200, "front", f"{_EUR_BASE}/200euro_obverse.jpg"),
    ("eur", 200, "back",  f"{_EUR_BASE}/200euro_reverse.jpg"),
    # ── Bulgarian leva ───────────────────────────────────────────────────────
    # BNB uses varying URL schemes; these follow the most common 2024 pattern.
    ("bgn",   5, "front", f"{_BGN_BASE}/5lv_face.jpg"),
    ("bgn",   5, "back",  f"{_BGN_BASE}/5lv_back.jpg"),
    ("bgn",  10, "front", f"{_BGN_BASE}/10lv_face.jpg"),
    ("bgn",  10, "back",  f"{_BGN_BASE}/10lv_back.jpg"),
    ("bgn",  20, "front", f"{_BGN_BASE}/20lv_face.jpg"),
    ("bgn",  20, "back",  f"{_BGN_BASE}/20lv_back.jpg"),
    ("bgn",  50, "front", f"{_BGN_BASE}/50lv_face.jpg"),
    ("bgn",  50, "back",  f"{_BGN_BASE}/50lv_back.jpg"),
    ("bgn", 100, "front", f"{_BGN_BASE}/100lv_face.jpg"),
    ("bgn", 100, "back",  f"{_BGN_BASE}/100lv_back.jpg"),
]


def _download(url: str, dest: str) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    headers = {"User-Agent": "ClarifEye/1.0 (banknote reference downloader)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(dest, "wb") as fh:
            fh.write(data)
        return True
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code}: {url}")
        return False
    except urllib.error.URLError as exc:
        print(f"  Network error ({exc.reason}): {url}")
        return False
    except Exception as exc:
        print(f"  Error ({exc}): {url}")
        return False


def main() -> None:
    failed = []

    for currency_code, denom, side, url in REFERENCE_URLS:
        out_dir = os.path.join(OUTPUT_DIR, currency_code)
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, f"{denom}_{side}.jpg")

        if os.path.exists(dest):
            print(f"[SKIP] {dest} already exists.")
            continue

        print(f"[DOWN] {currency_code.upper()} {denom} {side} …", end=" ", flush=True)
        ok = _download(url, dest)
        if ok:
            size_kb = os.path.getsize(dest) // 1024
            print(f"OK ({size_kb} KB)")
        else:
            failed.append((currency_code, denom, side, url))

    print()
    if not failed:
        print("All reference images downloaded successfully.")
        print(f"Images saved to: {OUTPUT_DIR}")
        return

    print(f"{len(failed)} download(s) failed.")
    print()
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("For failed images, download manually and place at the paths below.")
    print()
    print("Euro images — ECB press kit:")
    print("  https://www.ecb.europa.eu/euro/banknotes/security/html/index.en.html")
    print()
    print("Bulgarian leva images — BNB:")
    print("  https://www.bnb.bg/BanknotesAndCoins/Banknotes/index.htm")
    print()
    for currency_code, denom, side, url in failed:
        out_dir = os.path.join(OUTPUT_DIR, currency_code)
        dest = os.path.join(out_dir, f"{denom}_{side}.jpg")
        print(f"  Save as: {dest}")
        print(f"  Tried:   {url}")
        print()
    print("Minimum resolution: 800 px on the long edge (higher is better).")


if __name__ == "__main__":
    main()
