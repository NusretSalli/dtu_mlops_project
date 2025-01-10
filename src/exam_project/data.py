from pathlib import Path

import typer
from torch.utils.data import Dataset


from pathlib import Path
from collections import Counter
from PIL import Image
import typer

def get_image_shapes(data_dir: str):
    data_path = Path(data_dir)
    shapes = Counter()

    for image_path in data_path.glob("**/*"):
        if image_path.is_file() and image_path.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            with Image.open(image_path) as img:
                shapes[img.size] += 1

    for shape, count in shapes.items():
        print(f"Shape: {shape}, Count: {count}")

if __name__ == "__main__":
    typer.run(get_image_shapes)
