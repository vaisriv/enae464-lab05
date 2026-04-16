"""
Visualize a folder of cone_<NUM>.tif images as a video with FPS slider.

Usage: python cone_viewer.py /path/to/folder
"""

import sys
import re
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import TimerBase
from matplotlib.widgets import Slider, Button
from PIL import Image


def natural_key(path: Path) -> int:
    """Sort by the number in cone_<NUM>.tif."""
    m = re.search(r"cone_(\d+)", path.stem)
    return int(m.group(1)) if m else 0


class Player:
    """Holds playback state so attribute types are unambiguous."""

    idx: int = 0
    playing: bool = True
    timer: Optional[TimerBase] = None


def main(folder_arg: str) -> None:
    folder = Path(folder_arg)
    files = sorted(folder.glob("cone_*.tif"), key=natural_key)
    if not files:
        sys.exit(f"No cone_*.tif files found in {folder}")

    print(f"Loading {len(files)} frames...")
    frames = [np.asarray(Image.open(f)) for f in files]
    n = len(frames)

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.22)
    im = ax.imshow(frames[0], cmap="gray")
    title = ax.set_title(f"Frame 1 / {n}  —  {files[0].name}")
    ax.axis("off")

    # Sliders (tuples, not lists — matplotlib's type stubs require tuple[float, float, float, float])
    ax_fps = plt.axes((0.15, 0.10, 0.65, 0.03))
    ax_frame = plt.axes((0.15, 0.05, 0.65, 0.03))
    s_fps = Slider(ax_fps, "FPS", 1, 60, valinit=10, valstep=1)
    s_frame = Slider(ax_frame, "Frame", 0, n - 1, valinit=0, valstep=1)

    # Play/pause button
    ax_btn = plt.axes((0.85, 0.05, 0.1, 0.08))
    btn = Button(ax_btn, "Pause")

    p = Player()

    def show(i: int) -> None:
        im.set_data(frames[i])
        title.set_text(f"Frame {i + 1} / {n}  —  {files[i].name}")
        fig.canvas.draw_idle()

    def tick(_event=None) -> None:
        if not p.playing:
            return
        p.idx = (p.idx + 1) % n
        # Update frame slider without retriggering playback reset
        s_frame.eventson = False
        s_frame.set_val(p.idx)
        s_frame.eventson = True
        show(p.idx)

    def make_timer() -> TimerBase:
        interval = int(1000 / s_fps.val)
        t = fig.canvas.new_timer(interval=interval)
        t.add_callback(tick)
        return t

    def restart_timer(_=None) -> None:
        if p.timer is not None:
            p.timer.stop()
        p.timer = make_timer()
        if p.playing:
            p.timer.start()

    def on_frame(val) -> None:
        p.idx = int(val)
        show(p.idx)

    def on_btn(_event) -> None:
        p.playing = not p.playing
        btn.label.set_text("Pause" if p.playing else "Play")
        if p.timer is None:
            return
        if p.playing:
            p.timer.start()
        else:
            p.timer.stop()

    s_fps.on_changed(restart_timer)
    s_frame.on_changed(on_frame)
    btn.on_clicked(on_btn)

    restart_timer()
    plt.show()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    main(folder)
