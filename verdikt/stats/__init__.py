"""Statistics core. Importable as a library without touching the CLI."""

from .bayes import hdi_lift, prob_a_beats_b
from .intervals import clopper_pearson, interval, jeffreys, one_sided_lower, one_sided_upper, wilson
from .power import mde, normal_approx_n, power_exact, required_n
from .tests import bonferroni, compact_letters, holm, paired_p, unpaired_p

__all__ = [
    "bonferroni",
    "clopper_pearson",
    "compact_letters",
    "hdi_lift",
    "holm",
    "interval",
    "jeffreys",
    "mde",
    "normal_approx_n",
    "one_sided_lower",
    "one_sided_upper",
    "paired_p",
    "power_exact",
    "prob_a_beats_b",
    "required_n",
    "unpaired_p",
    "wilson",
]
