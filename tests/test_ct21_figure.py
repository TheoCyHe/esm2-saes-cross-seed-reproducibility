"""Tests for src/analysis/ct21_figure.py (amendment CT, §CT21)."""
import pytest

from src.analysis.ct21_figure import check


def _rows(dec070=0.28893591, act070=2.56555475):
    th = [round(0.30 + 0.05 * i, 2) for i in range(13)]
    return [{"theta": t, "decoder_fe_pp": dec070 if t == 0.70 else 0.1,
             "activation_fe_pp": act070 if t == 0.70 else 2.0} for t in th]


def test_the_record_values_pass():
    check(_rows())


def test_a_decoder_value_off_the_record_refuses():
    with pytest.raises(SystemExit, match="decoder_fe_pp"):
        check(_rows(dec070=0.2899))


def test_a_short_grid_refuses():
    with pytest.raises(SystemExit, match="13 ascending"):
        check(_rows()[:12])
