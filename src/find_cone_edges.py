"""
find_cone_edges.py
------------------------

Automatic Python port of the MATLAB edge-tracing routines
(`find_cone_edge_points.m`, `find_nearest_edge_points.m`,
`trace_edge.m`, `trace_edge_stopstart.m`, `get_new_indices.m`)
written by Dr. Stuart Laurence of UMD.

The original MATLAB code relied on the user clicking start/stop points on
every frame.  This version runs fully automatically, which makes it
suitable for batch-processing hundreds of images.

How it works
============
For each image:

1.  The dark cone is segmented from the lighter background using Otsu
    thresholding, and the largest connected component (the cone itself)
    is kept.  Any holes inside the silhouette are filled.

2.  A Canny edge detector is run on the original intensity image with the
    same threshold/sigma defaults as the MATLAB script
    (``threshold = [0.03, 0.1]``, ``smooth_sigma = 1.0``).

3.  Canny edges are restricted to a thin band around the silhouette
    boundary, so stray edges elsewhere in the image cannot contaminate the
    result.

4.  For every image column that contains cone pixels, the top-most and
    bottom-most edge pixels are taken as the upper-surface and
    lower-surface points, respectively.  This is the fully automatic
    equivalent of the manual "start point / stop point" tracing in the
    MATLAB version, and it naturally ignores the vertical base of the
    cone.

5.  Results are written to ``<edge_dir>/frame<N>_upper.txt`` and
    ``<edge_dir>/frame<N>_lower.txt`` in exactly the same tab-separated
    ``x\\ty`` format produced by the MATLAB script, so any downstream
    code that reads those files should continue to work unchanged.

Usage
=====
From the command line::

    python src/find_cone_edges.py \\
        --im-dir  ./data/mfg_m4_cone/images \\
        --edge-dir .data/mfg_m4_cone/edges \\
        --pattern 'cone_{:03d}.tif' \\
        --frames 0-192 \\
        --save-previews-dir ./data/mfg_m4_cone/previews

Or, as a library::

    from find_cone_edges import process_image, process_directory
    process_directory('images/', 'edges/', pattern='cone_{:03d}.tif',
                      preview_dir='previews/')

Dependencies: numpy, scipy, scikit-image, Pillow.  Matplotlib is only
needed if you want to save preview plots (``--save-previews``).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.feature import canny
from skimage.filters import threshold_otsu


# ---------------------------------------------------------------------------
# Core edge-extraction
# ---------------------------------------------------------------------------

@dataclass
class EdgeResult:
    """Container for the detected upper and lower edge points of one frame."""
    x_upper: np.ndarray
    y_upper: np.ndarray
    x_lower: np.ndarray
    y_lower: np.ndarray

    @property
    def n_upper(self) -> int:
        return len(self.x_upper)

    @property
    def n_lower(self) -> int:
        return len(self.x_lower)


def extract_cone_edges(
    img: np.ndarray,
    canny_sigma: float = 1.0,
    canny_low: float = 0.03,
    canny_high: float = 0.1,
    boundary_band_width: int = 2,
    min_cone_area: int = 500,
) -> EdgeResult:
    """Extract the upper and lower edges of a dark cone on a light background.

    Parameters
    ----------
    img : 2-D ndarray
        Grayscale image, typically ``uint8``.  A 2-D slice of a colour
        image will also work but colour images should be converted first.
    canny_sigma : float
        Standard deviation of the Gaussian smoothing kernel used before
        Canny edge detection (matches MATLAB ``smooth_sigma``).
    canny_low, canny_high : float
        Low/high hysteresis thresholds for Canny, as *fractions* of the
        maximum gradient magnitude (matches the MATLAB ``threshold``
        array ``[0.03, 0.1]``).
    boundary_band_width : int
        How many pixels of dilation are applied to the silhouette
        boundary when masking Canny edges.  The default (2) keeps edges
        within ~2 px of the silhouette.
    min_cone_area : int
        Safety check: raises ``RuntimeError`` if the largest connected
        component has fewer pixels than this.  Helps catch frames where
        the cone is missing or thresholding has failed badly.

    Returns
    -------
    EdgeResult
        Arrays of ``(x, y)`` pixel coordinates for the upper and lower
        surfaces, with the same conventions as the original MATLAB
        output (``x`` = column index, ``y`` = row index, 1-based is
        *not* used -- indices are 0-based as in Python).
    """
    if img.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale image, got shape {img.shape}."
        )

    # 1) Silhouette segmentation ------------------------------------------------
    # Otsu handles a wide range of lighting conditions robustly; the cone is
    # markedly darker than the background in every frame we've seen.
    thresh = threshold_otsu(img)
    binary = img < thresh

    labels, num = ndimage.label(binary)
    if num == 0:
        raise RuntimeError("No dark regions found -- thresholding failed.")

    # Component sizes (index 0 is the background label in `labels`, but
    # `ndimage.sum` returns sizes for all labels including 0, so we skip it).
    sizes = ndimage.sum(binary, labels, index=range(num + 1))
    biggest_label = int(np.argmax(sizes[1:])) + 1
    cone_mask = labels == biggest_label

    if cone_mask.sum() < min_cone_area:
        raise RuntimeError(
            f"Largest dark region has only {int(cone_mask.sum())} pixels "
            f"(< min_cone_area={min_cone_area}). Check threshold / image."
        )

    # Fill any internal holes so the mask's top/bottom rows genuinely
    # represent the outer silhouette.
    cone_mask = ndimage.binary_fill_holes(cone_mask)

    # 2) Canny edges on the original image -------------------------------------
    edges = canny(
        img.astype(float),
        sigma=canny_sigma,
        low_threshold=canny_low,
        high_threshold=canny_high,
        use_quantiles=False,
    )

    # 3) Keep only edges lying on the cone boundary ----------------------------
    boundary = cone_mask & ~ndimage.binary_erosion(cone_mask)
    boundary_band = ndimage.binary_dilation(
        boundary, iterations=max(1, boundary_band_width)
    )
    edges_on_cone = edges & boundary_band

    # 4) For every column that intersects the cone, take top-most and
    #    bottom-most edge pixel.  These automatically correspond to the
    #    upper and lower cone surfaces; the vertical base is skipped
    #    because its interior rows are neither topmost nor bottommost.
    col_has_cone = cone_mask.any(axis=0)
    cols = np.where(col_has_cone)[0]
    if len(cols) == 0:
        raise RuntimeError("Cone mask is empty after processing.")

    x_upper = np.empty(len(cols), dtype=np.int64)
    y_upper = np.empty(len(cols), dtype=np.int64)
    x_lower = np.empty(len(cols), dtype=np.int64)
    y_lower = np.empty(len(cols), dtype=np.int64)

    n_up = n_lo = 0
    for x in cols:
        edge_col = edges_on_cone[:, x]
        ys_edge = np.flatnonzero(edge_col)
        if ys_edge.size > 0:
            y_top, y_bot = ys_edge[0], ys_edge[-1]
        else:
            # Fall back to silhouette boundary in this column
            mask_col = cone_mask[:, x]
            ys_mask = np.flatnonzero(mask_col)
            if ys_mask.size == 0:
                continue
            y_top, y_bot = ys_mask[0], ys_mask[-1]

        x_upper[n_up] = x
        y_upper[n_up] = y_top
        n_up += 1
        x_lower[n_lo] = x
        y_lower[n_lo] = y_bot
        n_lo += 1

    return EdgeResult(
        x_upper=x_upper[:n_up].astype(float),
        y_upper=y_upper[:n_up].astype(float),
        x_lower=x_lower[:n_lo].astype(float),
        y_lower=y_lower[:n_lo].astype(float),
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_edge_file(path: str | os.PathLike, x: np.ndarray, y: np.ndarray) -> None:
    """Write ``x, y`` edge points to a tab-separated file, matching the
    format produced by the MATLAB ``find_cone_edge_points.m`` script.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt") as fid:
        fid.write("x\ty\n")
        for xi, yi in zip(x, y):
            fid.write(f"{xi:f}\t{yi:f}\n")


def process_image(
    image_path: str | os.PathLike,
    edge_dir: str | os.PathLike,
    frame_id: int | str,
    *,
    canny_sigma: float = 1.0,
    canny_low: float = 0.03,
    canny_high: float = 0.1,
    preview_dir: Optional[str | os.PathLike] = None,
) -> EdgeResult:
    """Process a single image and write the upper/lower edge files.

    Output filenames match the MATLAB script exactly:
    ``frame<frame_id>_upper.txt`` and ``frame<frame_id>_lower.txt``.

    If ``preview_dir`` is given, a ``frame<frame_id>_preview.png`` overlay
    is also written there (the directory is created if it doesn't exist).
    """
    image_path = Path(image_path)
    edge_dir = Path(edge_dir)

    img = np.array(Image.open(image_path))
    # Safety: if it's RGB, convert to grayscale
    if img.ndim == 3:
        img = np.array(Image.open(image_path).convert("L"))

    result = extract_cone_edges(
        img,
        canny_sigma=canny_sigma,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    upper_file = edge_dir / f"frame{frame_id}_upper.txt"
    lower_file = edge_dir / f"frame{frame_id}_lower.txt"
    write_edge_file(upper_file, result.x_upper, result.y_upper)
    write_edge_file(lower_file, result.x_lower, result.y_lower)

    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        _save_preview(img, result, preview_dir / f"frame{frame_id}_preview.png")

    return result


def _save_preview(img: np.ndarray, result: EdgeResult, out_path: Path) -> None:
    """Save a small PNG overlay so users can spot-check edge quality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img, cmap="gray")
    ax.plot(result.x_upper, result.y_upper, "r.", markersize=1, label="upper")
    ax.plot(result.x_lower, result.y_lower, "b.", markersize=1, label="lower")
    ax.legend(loc="upper right")
    ax.set_title(out_path.stem)
    ax.axis("off")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def _parse_frame_spec(spec: str) -> list[int]:
    """Parse ``'0-10'`` or ``'0,3,5-9'`` into an explicit list of frame numbers."""
    frames: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            frames.extend(range(int(a), int(b) + 1))
        else:
            frames.append(int(chunk))
    return frames


def process_directory(
    im_dir: str | os.PathLike,
    edge_dir: str | os.PathLike,
    *,
    pattern: str = "cone_{:03d}.tif",
    frames: Optional[Iterable[int]] = None,
    canny_sigma: float = 1.0,
    canny_low: float = 0.03,
    canny_high: float = 0.1,
    preview_dir: Optional[str | os.PathLike] = None,
    verbose: bool = True,
) -> None:
    """Batch-process an entire directory of cone images.

    Parameters
    ----------
    im_dir, edge_dir :
        Input image directory and output directory for edge text files.
    pattern :
        A Python format string taking a single frame number, e.g.
        ``'cone_{:03d}.tif'``.  If ``frames`` is None, the directory is
        scanned for files matching this pattern.
    frames :
        Optional explicit list/iterable of frame numbers to process.
    preview_dir :
        If given, also write a ``frame<N>_preview.png`` overlay image
        per frame into this directory, so edge quality can be spot-checked
        visually.  The directory is created if it doesn't exist.  If
        ``None`` (default), no previews are saved.
    """
    im_dir = Path(im_dir)
    edge_dir = Path(edge_dir)
    edge_dir.mkdir(parents=True, exist_ok=True)

    if frames is None:
        # Auto-discover frames from the pattern.  Build a regex from the
        # format string by replacing '{...}' with a capture group for digits.
        regex_str = re.escape(pattern)
        # Unescape and replace the escaped format placeholder
        regex_str = re.sub(r"\\\{[^}]*\\\}", r"(\\d+)", regex_str)
        regex = re.compile("^" + regex_str + "$")
        discovered: list[int] = []
        for name in sorted(os.listdir(im_dir)):
            m = regex.match(name)
            if m:
                discovered.append(int(m.group(1)))
        frames = sorted(discovered)
        if not frames:
            raise RuntimeError(
                f"No files matching pattern '{pattern}' found in {im_dir}"
            )

    frames = list(frames)
    n_ok = 0
    n_fail = 0
    for frame in frames:
        image_path = im_dir / pattern.format(frame)
        if not image_path.exists():
            if verbose:
                print(f"  [skip] {image_path} not found", file=sys.stderr)
            n_fail += 1
            continue
        try:
            res = process_image(
                image_path,
                edge_dir,
                frame_id=frame,
                canny_sigma=canny_sigma,
                canny_low=canny_low,
                canny_high=canny_high,
                preview_dir=preview_dir,
            )
            n_ok += 1
            if verbose:
                print(
                    f"  frame {frame}: upper={res.n_upper} pts, "
                    f"lower={res.n_lower} pts"
                )
        except Exception as e:
            n_fail += 1
            print(f"  [error] frame {frame}: {e}", file=sys.stderr)

    if verbose:
        print(f"Done. {n_ok} succeeded, {n_fail} failed.")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatic cone edge extraction (Python port of the "
        "MATLAB find_cone_edge_points.m workflow)."
    )
    p.add_argument(
        "--im-dir", required=True,
        help="Directory containing input TIFF images.",
    )
    p.add_argument(
        "--edge-dir", required=True,
        help="Directory in which to write per-frame edge text files.",
    )
    p.add_argument(
        "--pattern", default="cone_{:03d}.tif",
        help="Filename format string (default: 'cone_{:03d}.tif').",
    )
    p.add_argument(
        "--frames", default=None,
        help="Frames to process, e.g. '0-192' or '0,2,4-10'. "
             "If omitted, all matching files are processed.",
    )
    p.add_argument("--canny-sigma", type=float, default=1.0)
    p.add_argument("--canny-low", type=float, default=0.03)
    p.add_argument("--canny-high", type=float, default=0.1)
    p.add_argument(
        "--save-previews-dir", default=None, metavar="DIR",
        help="Directory in which to save a PNG overlay per frame for "
             "spot-checking.  Providing this flag enables preview "
             "generation; omit it to skip previews entirely.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-frame progress output.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    frames = _parse_frame_spec(args.frames) if args.frames else None
    process_directory(
        im_dir=args.im_dir,
        edge_dir=args.edge_dir,
        pattern=args.pattern,
        frames=frames,
        canny_sigma=args.canny_sigma,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        preview_dir=args.save_previews_dir,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
