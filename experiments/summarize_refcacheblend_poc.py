#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FFMPEG = (
    "/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/h3-nrt/lib/python3.11/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def decode(path: Path, width: int = 320, height: int = 180, fps: int = 6, frames: int = 24) -> np.ndarray:
    raw = subprocess.check_output(
        [
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale={width}:{height}:flags=bicubic",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    frame_bytes = width * height * 3
    count = len(raw) // frame_bytes
    if count == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8).reshape(
        count, height, width, 3
    ).astype(np.float32)


def metrics(reference: Path, candidate: Path) -> dict[str, float]:
    x = decode(reference)
    y = decode(candidate)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    mse = float(np.mean((x - y) ** 2))
    psnr = float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    xf = x.reshape(n, -1)
    yf = y.reshape(n, -1)
    mux = xf.mean(axis=1)
    muy = yf.mean(axis=1)
    vx = xf.var(axis=1)
    vy = yf.var(axis=1)
    cov = ((xf - mux[:, None]) * (yf - muy[:, None])).mean(axis=1)
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
        (mux * mux + muy * muy + c1) * (vx + vy + c2)
    )
    return {"mse": mse, "psnr": psnr, "global_ssim": float(np.mean(ssim))}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_refcacheblend_poc.py <job_id>")
    job_id = sys.argv[1]
    log = ROOT / "logs" / f"refcacheblend_poc_{job_id}.out"
    if not log.exists():
        raise SystemExit(f"missing log: {log}")
    results: dict[str, Path] = {}
    elapsed: dict[str, int] = {}
    for line in log.read_text(errors="ignore").splitlines():
        match = re.match(r"^RESULT_(.+)=(.+prompt0000\.mp4)$", line)
        if match:
            results[match.group(1)] = Path(match.group(2))
            continue
        match = re.match(r"^ELAPSED_(.+)=(\d+)$", line)
        if match:
            elapsed[match.group(1)] = int(match.group(2))

    out = ROOT / "summaries" / f"refcacheblend_poc_{job_id}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    teacher = results.get("teacher_abc")
    for tag, path in sorted(results.items()):
        row: dict[str, object] = {
            "variant": tag,
            "path": str(path),
            "elapsed_sec": elapsed.get(tag, ""),
        }
        if teacher is not None and path.exists() and tag != "teacher_abc":
            row.update(metrics(teacher, path))
        rows.append(row)
        if path.exists():
            shutil.copy2(path, out / f"{tag}.mp4")

    fields = sorted({key for row in rows for key in row})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(out)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
