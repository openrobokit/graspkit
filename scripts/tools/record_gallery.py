"""Record the Quickstart grasp gallery as a looping WebP: optimize one object for a hand and a gripper, then orbit both.

Select a free GPU, then run:

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync --with playwright scripts/tools/record_gallery.py
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image
from playwright.sync_api import sync_playwright

from graspkit.gallery.grasp_optim import GalleryViewer, cached_optimize, gallery_config

ROOT = Path(__file__).resolve().parents[2]


def main(
    output: Path = ROOT / "docs/assets/gallery/grasp_optim.webp",
    width: int = 480,
    seconds: float = 5.0,
    fps: int = 12,
    samples: int = 16,
    iterations: int = 300,
) -> None:
    assert torch.cuda.is_available(), "Select a free GPU and set CUDA_VISIBLE_DEVICES"
    assert width >= 64 and seconds > 0 and fps > 0 and round(seconds * fps) >= 4
    hands = ("xhand_right", "robotiq_2f85")
    object_key = "ycb/mustard_bottle"
    viewer = GalleryViewer(torch.device("cuda"), hand=hands[0], object_key=object_key)
    for hand in hands:
        viewer.hand_gui.value = hand
        config = replace(gallery_config(hand), num_samples_per_scene=samples, num_iters=iterations)
        run = cached_optimize(hand, [object_key], [viewer.scene(object_key)], config, viewer.device)[0]
        viewer.add_run(run)
    viewer.hand_gui.value = hands[0]
    viewer.refresh(select_last=True)

    height = width * 3 // 4
    frames: list[Image.Image] = []
    with sync_playwright() as playwright, ThreadPoolExecutor(max_workers=1) as renders:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"http://127.0.0.1:{viewer.server.get_port()}/?fixedDpr=1")
        deadline = time.monotonic() + 30
        while not viewer.server.get_clients():
            assert time.monotonic() < deadline, "Viser browser connection timed out"
            page.wait_for_timeout(100)
        client = next(iter(viewer.server.get_clients().values()))
        count = round(seconds * fps)
        for index in range(count):
            if index == count // 2:
                viewer.hand_gui.value = hands[1]
                viewer.refresh(select_last=True)
            progress = 2 * (index % (count // 2)) / (count // 2)
            viewer.grasp_gui.value = round((1 - abs(progress - 1)) * min(5, viewer.grasp_gui.max))
            viewer.show_grasp()
            phase = 2 * np.pi * index / count
            with client.atomic():
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.position = (0.38 + 0.06 * np.sin(phase), -0.48 + 0.06 * np.cos(phase), 0.32)
                client.camera.look_at = (0.0, 0.0, 0.10)
                client.camera.fov = np.deg2rad(40)
            page.wait_for_timeout(150 if index else 1500)
            render = renders.submit(client.get_render, height, width, transport_format="png")
            deadline = time.monotonic() + 45
            while not render.done():
                assert time.monotonic() < deadline, f"Viser render timed out at frame {index}"
                page.wait_for_timeout(10)
            frame = Image.fromarray(render.result()).convert("RGBA")
            frames.append(Image.alpha_composite(Image.new("RGBA", frame.size, "white"), frame).convert("RGB"))
            if index % fps == 0:
                print(f"Captured {index + 1}/{count}", flush=True)
        browser.close()
    viewer.server.stop()

    assert any(np.any(np.asarray(frame) != np.asarray(frames[0])) for frame in frames[1:]), "Scene did not move"
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [round((index + 1) * 1000 / fps) - round(index * 1000 / fps) for index in range(count)]
    for quality in (75, 60, 45, 30, 20):
        frames[0].save(
            output, save_all=True, append_images=frames[1:], duration=durations, loop=0, quality=quality, method=6
        )
        if output.stat().st_size <= 400_000:
            break
    assert output.stat().st_size <= 400_000, "Gallery preview exceeds 400 KB"
    with Image.open(output) as recorded:
        assert recorded.format == "WEBP" and recorded.is_animated and recorded.info["loop"] == 0
        assert recorded.size == (width, height)
    print(f"Wrote {output}: {count} frames, {output.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
