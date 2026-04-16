"""
Visualize a folder of cone_<NUM>.tif images as a video with FPS slider.

Usage: python cone_viewer.py /path/to/folder
"""

import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image


def natural_key(path):
    """Sort by the number in cone_<NUM>.tif."""
    m = re.search(r"cone_(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def main(folder):
    folder = Path(folder)
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

    # Sliders
    ax_fps = plt.axes([0.15, 0.10, 0.65, 0.03])
    ax_frame = plt.axes([0.15, 0.05, 0.65, 0.03])
    s_fps = Slider(ax_fps, "FPS", 1, 60, valinit=10, valstep=1)
    s_frame = Slider(ax_frame, "Frame", 0, n - 1, valinit=0, valstep=1)

    # Play/pause button
    ax_btn = plt.axes([0.85, 0.05, 0.1, 0.08])
    btn = Button(ax_btn, "Pause")

    state = {"idx": 0, "playing": True, "timer": None}

    def show(i):
        im.set_data(frames[i])
        title.set_text(f"Frame {i + 1} / {n}  —  {files[i].name}")
        fig.canvas.draw_idle()

    def tick(_event=None):
        if not state["playing"]:
            return
        state["idx"] = (state["idx"] + 1) % n
        # Update frame slider without retriggering playback reset
        s_frame.eventson = False
        s_frame.set_val(state["idx"])
        s_frame.eventson = True
        show(state["idx"])

    def make_timer():
        interval = int(1000 / s_fps.val)
        t = fig.canvas.new_timer(interval=interval)
        t.add_callback(tick)
        return t

    def restart_timer(_=None):
        if state["timer"] is not None:
            state["timer"].stop()
        state["timer"] = make_timer()
        if state["playing"]:
            state["timer"].start()

    def on_frame(val):
        state["idx"] = int(val)
        show(state["idx"])

    def on_btn(_event):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            state["timer"].start()
        else:
            state["timer"].stop()

    s_fps.on_changed(restart_timer)
    s_frame.on_changed(on_frame)
    btn.on_clicked(on_btn)

    restart_timer()
    plt.show()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    main(folder)
