"""Explicit isolated research setup; no training and no official-test access."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs/video_research_v1"
REVISION = "d6d9a351f32334ae9a3a250872c05836e0dd0f2c"
ASSETS = {
    "FFraw.tar": (
        "https://drive.usercontent.google.com/download?id=12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV&export=download&confirm=t",
        "e477fc1789f8c1d2147cffe86a47eb37a24adedae10b106b7d47c1eeb73777e2",
    ),
    "FFc23.tar": (
        "https://drive.usercontent.google.com/download?id=1X0-NYT8KPursLZZdxduRQju6E52hauV0&export=download&confirm=t",
        "2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267",
    ),
    "retinaface_download.zip": (
        "https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",
        "f168fae3ce6d6d3af69dd712049bfe857d3bc1422f715ac8ddfdf552ed7070ef",
    ),
}
UPSTREAM = {
    "LICENSE": "9b78089abdb0bd3689d6c0fff04f4554d1a174ba13b7919ce74b8b07bb7cce3f",
    "README.md": "87228b403d1a27cb4873b40730078c4737a2ca2b7aaae99d5d71206a70241c48",
    "src/inference/preprocess.py": "cd9d03e4e85fa39fb83e8bf6fe0756ed3a1a92c32fe252918225490cb018341e",
    "src/model.py": "bfd0d4b454df6bd1c3aecf1135fff5b2fbc5663640ca2bd7df2f6d65811af01b",
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha(path) != expected:
            raise ValueError(f"Existing asset hash mismatch; preserved: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".download")
    with (
        urllib.request.urlopen(url, timeout=90) as response,
        temporary.open("xb") as handle,
    ):
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if sha(temporary) != expected:
        raise ValueError(f"Downloaded hash mismatch; not activated: {temporary}")
    temporary.rename(path)


def main():
    parser = argparse.ArgumentParser(
        description="Research-only SBI setup. Requires existing CUDA .venv. Never trains."
    )
    parser.add_argument("--download-and-install", action="store_true")
    parser.add_argument("--audit-train-val", action="store_true")
    args = parser.parse_args()
    if not args.download_and_install and not args.audit_train_val:
        parser.error("Select an explicit setup/audit action.")
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.download_and_install:
        for name, checksum in UPSTREAM.items():
            download(
                f"https://raw.githubusercontent.com/mapooon/SelfBlendedImages/{REVISION}/{name}",
                ROOT / "upstream" / name.replace("/", "_"),
                checksum,
            )
        for name, (url, checksum) in ASSETS.items():
            download(url, ROOT / name, checksum)
        extracted = ROOT / "retinaface.pth"
        if not extracted.exists():
            with zipfile.ZipFile(ROOT / "retinaface_download.zip") as archive:
                data = archive.read("retinaface_resnet50_2020-07-20.pth")
            with extracted.open("xb") as handle:
                handle.write(data)
        if (
            sha(extracted)
            != "b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2"
        ):
            raise ValueError("RetinaFace extracted checkpoint checksum mismatch.")
        environment = ROOT / "env"
        if not environment.exists():
            venv.EnvBuilder(with_pip=True).create(environment)
            (environment / "Lib/site-packages/project_runtime.pth").write_text(
                str(PROJECT / ".venv/Lib/site-packages") + "\n", encoding="utf-8"
            )
        python = environment / "Scripts/python.exe"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-r",
                str(PROJECT / "requirements-video-forensic.txt"),
            ],
            check=True,
        )
        print(
            "Isolated environment ready. Upstream license: internal noncommercial research only; do not redistribute weights/code."
        )
    if args.audit_train_val:
        sys.path.insert(0, str(PROJECT))
        from src.video_cache import audit_feature_cache
        from src.video_manifest import load_manifest
        from src.video_model import VideoClassifier
        from src.video_pipeline import FacePipeline, FeatureCache, VideoSettings

        manifest = load_manifest(
            PROJECT / "outputs/video_identity_v1/manifest.json", verify_inventory=False
        )
        pipeline = FacePipeline(VideoSettings())
        model = VideoClassifier(pretrained=True)
        cache = FeatureCache(
            PROJECT / "outputs/video_v1/cache",
            pipeline.signature,
            model.backbone_fingerprint(),
        )
        audit = audit_feature_cache(manifest, cache)
        path = ROOT / "roster_audit.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                existing["cache_fingerprint"] != audit["cache_fingerprint"]
                or existing["manifest_fingerprint"] != audit["manifest_fingerprint"]
            ):
                raise ValueError("Existing audit differs; preserved.")
        else:
            path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print({k: v["accepted"] for k, v in audit["splits"].items()})


if __name__ == "__main__":
    main()
