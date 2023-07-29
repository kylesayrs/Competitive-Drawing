# ln -s /opt/homebrew/lib/libcairo.2.dylib .
from typing import Optional

import os
import json
import tqdm
import numpy
import random
import argparse
import subprocess
import cairocffi as cairo
from concurrent.futures import ThreadPoolExecutor


parser = argparse.ArgumentParser()
parser.add_argument("output_dir_path")
parser.add_argument("--tmp_stroke_dir", default="/tmp")
parser.add_argument("--image_side", default=50)
parser.add_argument("--line_diameter", default=16)
parser.add_argument("--padding", default=0)
parser.add_argument("--parallel", default=False)
parser.add_argument("--filter", default=None)


def read_28x28_paths():
    all_paths = subprocess.Popen(
        "gsutil -m ls 'gs://quickdraw_dataset/full/numpy_bitmap'",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.read()
    all_paths = all_paths.decode("utf-8")
    all_paths = all_paths.split("\n")
    all_paths = all_paths[:-1]

    return all_paths


def read_stroke_paths():
    all_paths = subprocess.Popen(
        "gsutil -m ls 'gs://quickdraw_dataset/full/simplified'",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.read()
    all_paths = all_paths.decode("utf-8")
    all_paths = all_paths.split("\n")
    all_paths = all_paths[:-1]

    return all_paths


def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """

    original_side = 256.0

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2.0 + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2.0, total_padding / 2.0)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = numpy.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.0
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = numpy.copy(numpy.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images


def download_and_draw_strokes(
    category_path: str,
    output_dir_path: str,
    image_side: int,
    line_diameter: int,
    padding: int,
    progress: Optional[tqdm.std.tqdm]
):
    category_name = category_path.split("/")[-1].split(".")[0]
    category_name = category_name.replace(" ", "\ ")

    save_path = os.path.join(output_dir_path, f"{category_name}.npy")
    save_path = save_path.replace("\\", "")  # JANK
    if os.path.exists(save_path):
        progress.update(1)
        return

    stroke_output_path = os.path.join(args.tmp_stroke_dir, f"{category_name}.ndjson")
    process = subprocess.Popen(f"gsutil -m cp '{category_path}' {stroke_output_path}", shell=True)
    process.wait()

    stroke_output_path = stroke_output_path.replace("\\", "")  # JANK
    with open(stroke_output_path, "r") as stroke_file:
        vector_images = []
        for line in stroke_file:
            drawing_data = json.loads(line)
            if drawing_data["recognized"]:
                vector_images.append(drawing_data["drawing"])

    raster_images = vector_to_raster(
        vector_images,
        side=image_side,
        line_diameter=line_diameter,
        padding=padding,
    )
    raster_images = numpy.array(raster_images)

    numpy.save(save_path, raster_images)

    os.remove(stroke_output_path)
    progress.update(1)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.image_side == 28:
        paths = read_28x28_paths()

        for category_path in tqdm.tqdm(paths):
            category_name = category_path.split("/")[-1].split(".")[0]
            category_name = category_name.replace(" ", "\ ")

            output_path = os.path.join(args.output_dir_path, f"{category_name}.npy")
            process = subprocess.Popen(f"gsutil -m cp '{category_path}' {output_path}", shell=True)
            if not args.parallel:
                process.wait()

    else:
        paths = read_stroke_paths()
        if args.filter is not None:
            paths = [path for path in paths if args.filter in path]
        random.shuffle(paths)

        progress = tqdm.tqdm(total=len(paths))

        os.makedirs(args.output_dir_path, exist_ok=True)

        if args.parallel:
            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = [
                    executor.submit(
                        download_and_draw_strokes,
                        category_path,
                        args.output_dir_path,
                        args.image_side,
                        args.line_diameter,
                        args.padding,
                        progress
                    )
                    for category_path in paths
                ]

        else:
            for category_path in paths:
                try:
                    download_and_draw_strokes(
                        category_path,
                        args.output_dir_path,
                        args.image_side,
                        args.line_diameter,
                        args.padding,
                        progress
                    )
                except Exception as e:
                    print(e)
