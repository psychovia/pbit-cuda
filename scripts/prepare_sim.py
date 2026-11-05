
import argparse
from pathlib import Path

import numpy as np


# Grpah coloring 

def color_graph(J):

    n = len(J)

    # Connected p-bits cannot share a color
    adjacency = (J != 0) | (J.T != 0)

    np.fill_diagonal(adjacency, False)

    colors = np.full(n, -1, dtype=np.int32)

    # Process highest-degree p-bits first
    order = sorted(
        range(n),
        key=lambda i: (-int(adjacency[i].sum()), i)
    )

    for i in order:

        used = {
            int(colors[j])
            for j in np.flatnonzero(adjacency[i])
            if colors[j] >= 0
        }

        color = 0

        while color in used:
            color += 1

        colors[i] = color

    return colors


# format array for cuda headers

def format_float(values):

    result = []

    for value in np.ravel(values):

        text = f"{float(value):.9g}"

        if "." not in text and "e" not in text:
            text += ".0"

        result.append(text + "f")

    return ", ".join(result)


def format_int(values):

    return ", ".join(
        str(int(x)) for x in np.ravel(values)
    )


# ==================================================
# Read Makefile inputs
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--J", required=True)
    parser.add_argument("--h", required=True)
    parser.add_argument("--initial-state", default="random")

    parser.add_argument("--nt", type=int, default=1000)

    parser.add_argument(
        "--beta-schedule",
        choices=["constant", "linear"],
        default="linear"
    )

    parser.add_argument(
        "--beta-start",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--beta-end",
        type=float,
        default=5.0
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.nt < 1:
        parser.error("NT must be at least 1")

    if args.seed < 0:
        parser.error("SEED must be nonnegative")

    if not np.isfinite([args.beta_start, args.beta_end]).all():
        parser.error("Beta values must be finite")

    # Load coupling matrix and bias vector
    J = np.asarray(
        np.load(args.J),
        dtype=np.float32
    )

    h = np.asarray(
        np.load(args.h),
        dtype=np.float32
    ).reshape(-1)

    # Validate circuit
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be an N x N matrix")

    n = J.shape[0]

    if n < 1:
        raise ValueError("Circuit must have at least one p-bit")

    if h.shape != (n,):
        raise ValueError("h must contain N values")

    if not np.isfinite(J).all() or not np.isfinite(h).all():
        raise ValueError("J and h must contain finite values")

    if not np.allclose(J, J.T):
        raise ValueError("J must be symmetric")

    if not np.allclose(np.diag(J), 0):
        raise ValueError("J must have zero diagonal")

    # Initial p-bit states
    if args.initial_state == "random":

        rng = np.random.default_rng(args.seed)

        m = rng.choice(
            [-1, 1],
            size=n
        ).astype(np.int32)

    else:

        m = np.load(args.initial_state)

        if m.shape != (n,):
            raise ValueError(
                "Initial state must contain N values"
            )

        if not np.isin(m, [-1, 1]).all():
            raise ValueError(
                "Initial states must be -1 or +1"
            )

        m = m.astype(np.int32)

    # Beta schedule
    if args.beta_schedule == "constant":

        beta = np.full(
            args.nt,
            args.beta_start,
            dtype=np.float32
        )

    else:

        beta = np.linspace(
            args.beta_start,
            args.beta_end,
            args.nt,
            dtype=np.float32
        )

    # Graph coloring
    colors = color_graph(J)
    n_colors = int(colors.max()) + 1

    # Generate CUDA header
    header = f"""
#pragma once

#define N_PBITS {n}
#define N_COLORS {n_colors}
#define NT {args.nt}
#define SEED {args.seed}ULL

static const float HOST_J[N_PBITS * N_PBITS] = {{
    {format_float(J)}
}};

static const float HOST_H[N_PBITS] = {{
    {format_float(h)}
}};

static const int HOST_M[N_PBITS] = {{
    {format_int(m)}
}};

static const int HOST_COLORS[N_PBITS] = {{
    {format_int(colors)}
}};

static const float HOST_BETA[NT] = {{
    {format_float(beta)}
}};
"""

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.write_text(header)

    print("Circuit prepared")
    print("P-bits:", n)
    print("Colors:", n_colors)
    print("Sweeps:", args.nt)
    print("Beta schedule:", args.beta_schedule)
    print("Header:", output)


if __name__ == "__main__":
    main()
