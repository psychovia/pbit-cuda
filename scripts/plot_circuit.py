"""Arrange final CUDA spins into 2 x 2 cells using the mentor's pixel mapping."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def indices_to_grid(v, W: int, H: int):
    """
    v: vector of length 4 * W * H, ordered cell by cell.
    W, H: number of 2 x 2 cells across and down.
    Returns a pixel grid of shape (2 * H, 2 * W).
    """
    v = np.asarray(v)
    assert v.size == 4 * W * H, "Vector length must be 4*W*H"

    grid = np.zeros((2 * H, 2 * W), dtype=v.dtype)

    for i in range(v.size):
        # Every four consecutive spins belong to one 2 x 2 cell.
        cell = i // 4
        local = i % 4

        cr, cc = divmod(cell, W)   # Cell row and column.
        lr, lc = divmod(local, 2)  # Pixel row and column inside the cell.

        # Convert cell coordinates into coordinates in the full image.
        r = 2 * cr + lr
        c = 2 * cc + lc
        grid[r, c] = v[i]

    return grid


def main():
    # W and H count cells, so 50 x 50 cells produce a 100 x 100 image.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spins", type=Path)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/circuit.png"))
    parser.add_argument("--show", action="store_true", help="Also display the image")
    args = parser.parse_args()


    if args.spins.suffix == ".npy":
        final_state = np.load(args.spins)
    else:
        final_state = np.loadtxt(args.spins, dtype=np.int8)

    img = indices_to_grid(final_state, args.width, args.height)

    # -1 - black / +1 - white.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(args.output, img, cmap="gray", vmin=-1, vmax=1)
    print(f"Final image: {args.output}")

    if args.show:
        plt.imshow(img, cmap="gray", vmin=-1, vmax=1)
        plt.show()


if __name__ == "__main__":
    main()
