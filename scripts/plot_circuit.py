"""Plot the final CUDA spins as an image (one spin per pixel)."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spins_to_image(spins, image_shape, block_shape):
    """Fill the image block by block, moving left to right, then down."""
    image_rows, image_cols = image_shape
    block_rows, block_cols = block_shape

    image = np.zeros(image_shape, dtype=np.int8)
    spins_per_block = block_rows * block_cols
    spin_index = 0

    for row in range(0, image_rows, block_rows):
        for col in range(0, image_cols, block_cols):
            # For a 2 x 2 block, the next four spins become two pixel rows.
            next_index = spin_index + spins_per_block
            block = spins[spin_index:next_index].reshape(block_rows, block_cols)
            image[row:row + block_rows, col:col + block_cols] = block
            spin_index = next_index

    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spins", type=Path)
    parser.add_argument(
        "--shape", nargs=2, type=int, default=[100, 100],
        metavar=("ROWS", "COLS"),
    )
    parser.add_argument(
        "--block-shape", nargs=2, type=int, default=[1, 1],
        metavar=("ROWS", "COLS"),
        help="Consecutive spins fill a block; blocks run left to right, then down",
    )
    parser.add_argument("--output", type=Path, default=Path("results/circuit.png"))
    args = parser.parse_args()

    # Read final spins in their original circuit order.
    if args.spins.suffix == ".npy":
        spins = np.load(args.spins)
    else:
        spins = np.loadtxt(args.spins, dtype=np.int8)

    image = spins_to_image(spins, args.shape, args.block_shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(args.output, image, cmap="gray", vmin=-1, vmax=1)
    print(f"Final image: {args.output}")


if __name__ == "__main__":
    main()
