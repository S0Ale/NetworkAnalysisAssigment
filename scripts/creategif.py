import os
import sys
from pathlib import Path
import imageio.v3 as iio

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <folder> <format:gif|mp4>")
        sys.exit(1)

    frames_folder = Path(sys.argv[1])
    save_format = sys.argv[2]

    images = sorted([img for img in frames_folder.iterdir() 
                 if img.suffix == ".png" and img.name.startswith("frame_")])
    frames = [iio.imread(img) for img in images]

    fps = 1
    duration = int(1000 / fps)

    output_path = frames_folder / f"simulation.{save_format}"

    if save_format == "gif":
        iio.imwrite(str(output_path), frames, plugin="pillow", duration=duration, loop=0)
    elif save_format == "mp4":
        iio.imwrite(str(output_path), frames, fps=fps)
    else:
        print("Unsupported format. Use 'gif' or 'mp4'.")
        sys.exit(1)

    print(f"Animation saved as {output_path}")