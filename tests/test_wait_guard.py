"""Tests for `src/cluster/wait_guard` -- PREREGISTRATION amendment BA 7.5.

THE FIXTURE PRINCIPLE, and it decides this whole file. AZ20 item 3's failure
was NOT a syntax error and NOT a malformed readout: both commands ran, both
produced well-formed output, and the wait reported completion anyway because
the condition was evaluated against the wrong shell's view. **So a test that
only feeds this module garbage cannot exhibit the disputed condition.**

The load-bearing fixture is therefore `test_refuses_a_FALSE_COMPLETION...`:
a perfectly well-formed EMPTY primary against a perfectly well-formed secondary
showing RUNNING. That is the real failure mode, byte for byte.
"""
from __future__ import annotations

import pytest

from src.cluster import wait_guard as wg


# ------------------------------------------------------- the real failure mode

def test_refuses_a_FALSE_COMPLETION_the_AZ20_case():
    """AZ20 item 3, reproduced exactly: an EMPTY primary (the locally-expanded
    `$(squeue ...)` produced nothing) against a secondary showing three live
    jobs. The wait printed ALL_DONE. This must refuse."""
    v = wg.corroborate("", "RUNNING\nRUNNING\nRUNNING\n")
    assert v["verdict"] == "UNCORROBORATED_COMPLETE"
    assert v["ok"] is False and v["complete"] is False
    assert v["n_secondary_live"] == 3


def test_a_silent_secondary_REFUSES_rather_than_clearing_the_wait():
    """Gotcha 28: silence and success are the same bytes, and empty output is
    the PRIMARY's own failure signature -- so it must not be the thing that
    clears it."""
    v = wg.corroborate("", "")
    assert v["verdict"] == "SECONDARY_UNAVAILABLE"
    assert v["ok"] is False and v["complete"] is False


def test_a_MISSING_secondary_is_also_a_refusal_not_a_completion():
    v = wg.corroborate("", None)
    assert v["verdict"] == "SECONDARY_UNAVAILABLE"
    assert v["complete"] is False


def test_a_failed_primary_cannot_clear_itself_by_producing_nothing():
    """`primary_raw=None` means the command failed. It must not read as
    'no jobs running' -- completion still requires the secondary to agree."""
    v = wg.corroborate(None, "RUNNING\n")
    assert v["verdict"] == "UNCORROBORATED_COMPLETE"
    assert v["complete"] is False


# ------------------------------------------------------- the passing outcomes

def test_corroborated_complete_is_the_only_outcome_that_proceeds():
    v = wg.corroborate("", "COMPLETED\nCOMPLETED\n")
    assert v["verdict"] == "CORROBORATED_COMPLETE"
    assert v["ok"] is True and v["complete"] is True


def test_incomplete_is_not_an_error():
    v = wg.corroborate("RUNNING\nPENDING\n", "RUNNING\nPENDING\n")
    assert v["verdict"] == "INCOMPLETE"
    assert v["ok"] is True and v["complete"] is False


def test_every_outcome_is_reachable_so_the_guard_can_do_more_than_one_thing():
    """Gotcha 25: 'can only fail' is as useless as 'can never fail'. Four
    distinct verdicts from four inputs."""
    got = {
        wg.corroborate("", "COMPLETED\n")["verdict"],
        wg.corroborate("RUNNING\n", "RUNNING\n")["verdict"],
        wg.corroborate("", "RUNNING\n")["verdict"],
        wg.corroborate("", "")["verdict"],
    }
    assert got == {"CORROBORATED_COMPLETE", "INCOMPLETE",
                   "UNCORROBORATED_COMPLETE", "SECONDARY_UNAVAILABLE"}


# ------------------------------------------------------------ state parsing

def test_sacct_cancellation_decoration_is_parsed_as_a_terminal_state():
    """`CANCELLED by 12345` is one line whose FIRST token is the state. Read
    whole it would match nothing in LIVE_STATES by accident; read as the first
    token it is correctly terminal."""
    v = wg.corroborate("", "CANCELLED by 3001\nCOMPLETED\n")
    assert v["verdict"] == "CORROBORATED_COMPLETE"


def test_blank_lines_do_not_count_as_output():
    """A readout of only whitespace produced no STATES, so it is unavailable
    rather than a clean queue."""
    v = wg.corroborate("", "\n  \n\n")
    assert v["verdict"] == "SECONDARY_UNAVAILABLE"


def test_states_are_case_normalised():
    assert wg.parse_states("running\n") == ["RUNNING"]
    v = wg.corroborate("", "running\n")
    assert v["verdict"] == "UNCORROBORATED_COMPLETE"


def test_a_terminal_secondary_state_does_not_block_completion():
    v = wg.corroborate("", "COMPLETED\nFAILED\nTIMEOUT\nOUT_OF_MEMORY\n")
    assert v["verdict"] == "CORROBORATED_COMPLETE"


# THE LIST IS WRITTEN OUT, NOT DERIVED FROM `wg.LIVE_STATES`. Parametrising
# over the module's own constant makes the test unable to fail: deleting a
# state deletes its case too, so the mutation "PENDING is no longer live"
# SURVIVED the first draft of this file. A test whose cases come from the thing
# under test cannot exhibit the disputed condition (gotcha 16).
EXPECTED_LIVE = ["PENDING", "RUNNING", "SUSPENDED", "CONFIGURING",
                 "COMPLETING", "RESIZING", "REQUEUED", "SIGNALING",
                 "STAGE_OUT", "STOPPED"]


@pytest.mark.parametrize("state", EXPECTED_LIVE)
def test_every_live_state_blocks_completion(state):
    v = wg.corroborate("", f"{state}\n")
    assert v["complete"] is False, f"{state} did not block completion"


def test_the_expected_live_states_are_all_actually_declared_live():
    """The companion to the literal list above: if a state is dropped from
    `LIVE_STATES`, this names it rather than silently testing one case fewer."""
    missing = [s for s in EXPECTED_LIVE if s not in wg.LIVE_STATES]
    assert not missing, f"declared live states have gone missing: {missing}"


# ------------------------------------------------------------- the exit path

def test_main_exit_codes_are_gateable(tmp_path, capsys):
    """Gotcha 13 extension 1: a validator whose failure is unreadable by the
    thing gating on it is worse than no validator. Each verdict must reach a
    DISTINCT exit code."""
    prim, sec = tmp_path / "p.txt", tmp_path / "s.txt"
    import sys

    def run(p_text: str, s_text: str) -> int:
        prim.write_text(p_text)
        sec.write_text(s_text)
        argv = sys.argv
        sys.argv = ["wait_guard", "--primary", str(prim), "--secondary",
                    str(sec)]
        try:
            wg.main()
        except SystemExit as e:
            return int(e.code)
        finally:
            sys.argv = argv
        raise AssertionError("main() did not exit")

    assert run("", "COMPLETED\n") == 0            # proceed
    assert run("RUNNING\n", "RUNNING\n") == 2     # keep waiting
    assert run("", "RUNNING\n") == 1              # REFUSE
    assert run("", "") == 1                       # REFUSE
    capsys.readouterr()
