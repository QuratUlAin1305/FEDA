# Locked colour scheme for the whole manuscript: teal - amber - slate
TEAL_DK   = "#0B4F51"
TEAL      = "#12797C"
TEAL_MD   = "#4AA3A5"
TEAL_LT   = "#9CCACB"
TEAL_PALE = "#E4F1F1"
AMBER_DK  = "#B06A05"
AMBER     = "#E0930F"
AMBER_LT  = "#F2C46A"
AMBER_PALE= "#FCEFD6"
SLATE_DK  = "#26323A"
SLATE     = "#485863"
SLATE_MD  = "#8A98A2"
SLATE_LT  = "#D3DBE0"
PAPER     = "#FFFFFF"
RUST      = "#B5502E"
RUST_DK   = "#8C3A1E"
RUST_LT   = "#E0A08C"
SERIES    = [TEAL, AMBER, RUST, TEAL_MD, AMBER_DK, SLATE]
# fixed colour for each descriptor block, used in every block-level figure
BLOCK = {"ESPH": TEAL, "ADM": AMBER, "AM": RUST, "FG": TEAL_MD}

import matplotlib as mpl
def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": SLATE_DK,
        "text.color": SLATE_DK,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.linewidth": 0.9,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
    })
