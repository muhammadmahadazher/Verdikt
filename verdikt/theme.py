"""Design tokens - one source of truth for terminal, HTML report and matplotlib figures.

Visual language: operational-futures industrial. Dark steel surfaces, structural hairlines,
amber as the warning signal, and colour used only to encode verdict state - never decoration.
The four verdict states map to the four exit codes, so a reader learns the palette once.
"""

from __future__ import annotations

# ---------------------------------------------------------------- surfaces
BG = "#0E1116"          # deep steel, page background
SURFACE = "#161B22"     # raised card
SURFACE_2 = "#1C232C"   # table header / inset
HAIRLINE = "#2A323D"    # 1px structural divider
TEXT = "#E6EDF3"        # primary
TEXT_DIM = "#9BA7B4"    # secondary
TEXT_FAINT = "#6B7785"  # tertiary / captions

# ------------------------------------------------------------ verdict states
# code -> (name, hex, terminal style)
VERDICT = {
    0: ("BETTER", "#3FB950", "bold green"),
    1: ("REGRESSION", "#F85149", "bold red"),
    2: ("UNDERPOWERED", "#D9A227", "bold yellow"),
    3: ("NOT COMPARABLE", "#A371F7", "bold magenta"),
}
OK = VERDICT[0][1]
BAD = VERDICT[1][1]
WARN = VERDICT[2][1]
CONFOUND = VERDICT[3][1]
ACCENT = "#58A6FF"      # informational highlight (never a verdict colour)

# ------------------------------------------------------------ provenance tags
# every printed metric declares how much it can be trusted
PROVENANCE = {
    "validated": ("#3FB950", "peer-reviewed method, reference implementation"),
    "prior-art": (ACCENT, "reimplemented from a cited tool"),
    "provisional": (WARN, "not yet calibrated - do not build decisions on this"),
}

# ---------------------------------------------------------------- series
# categorical colours for policy arms in figures (colour-blind safe ordering)
SERIES = ["#58A6FF", "#D9A227", "#3FB950", "#A371F7", "#F85149", "#39C5CF"]

FONT_STACK = "ui-monospace, 'Cascadia Code', 'JetBrains Mono', Consolas, monospace"
FONT_UI = "'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif"


def mpl_rc() -> dict:
    """matplotlib rcParams for figures that match the report and the README."""
    return {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_DIM,
        "axes.edgecolor": HAIRLINE,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "grid.color": HAIRLINE,
        "grid.alpha": 0.6,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "monospace",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.titlesize": 14,
        "legend.frameon": False,
    }
