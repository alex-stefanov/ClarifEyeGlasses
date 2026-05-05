#!/usr/bin/env python3
"""
Install argostranslate language packages required by ClarifEye.

Required pairs: en → bg  and  bg → en

Idempotent — already-installed packages are skipped.

Usage (run once on the Pi after `pip install argostranslate`):
    python scripts/install_argos_packages.py
"""
from argostranslate import package

REQUIRED_PAIRS = [("en", "bg"), ("bg", "en")]


def main() -> None:
    print("Updating argostranslate package index …")
    package.update_package_index()

    available = package.get_available_packages()
    installed = {(p.from_code, p.to_code) for p in package.get_installed_packages()}

    for from_code, to_code in REQUIRED_PAIRS:
        if (from_code, to_code) in installed:
            print(f"  [ok] {from_code} → {to_code} already installed")
            continue

        pkg = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if pkg is None:
            print(f"  [!!] {from_code} → {to_code} not found in package index — check your internet connection")
            continue

        print(f"  [..] Downloading and installing {from_code} → {to_code} …")
        package.install_from_path(pkg.download())
        print(f"  [ok] {from_code} → {to_code} installed")

    print("Done.")


if __name__ == "__main__":
    main()
