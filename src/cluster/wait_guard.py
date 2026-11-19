"""BA12 -- THE REMOTE-WAIT GUARD. PREREGISTRATION amendment BA section 7.5.

=============================================================================
WHY THIS EXISTS, AND WHY "USE A SCRIPT FILE" WAS NOT ENOUGH
=============================================================================
The shell-expansion guard was deferred three times on the grounds that writing
the remote command to a script file removes the hazard. **It does not, and
AZ20 item 3 is the proof.** A wait loop written as

    until [ -z "$(squeue -u <netid> -h)" ]; do sleep 60; done; echo ALL_DONE

expanded `$(...)` on the LOCAL side, so the condition was evaluated against a
local shell's idea of the queue -- empty -- and the loop printed **`ALL_DONE`
while three jobs were still `RUNNING`**. **A false COMPLETION signal is the
dangerous direction**: it leads to reading incomplete results, which is worse
than the empty-log-header case (gotcha 28), where the failure at least looked
like nothing.

**It was caught only because the same invocation happened to also run `sacct`,
whose output contradicted it.** That accident is the whole design of this
module:

    THE RULE: pair any remote wait with an INDEPENDENT READOUT of the same
    state. A wait that reports completion from one source and is NOT
    corroborated by a second FAILS rather than proceeding.

=============================================================================
WHAT MAKES THE SECOND READOUT INDEPENDENT
=============================================================================
`squeue` reads the scheduler's live queue; `sacct` reads the accounting
database. A job that has left `squeue` for any reason other than finishing --
and, crucially, **a `squeue` invocation whose output never reached the
condition at all** -- still has a live state in `sacct`. The two fail in
different ways, which is the only property that matters here.

=============================================================================
AND SILENCE IS NOT CORROBORATION (gotcha 28)
=============================================================================
The primary hazard's own signature is EMPTY OUTPUT read as "nothing running".
So a secondary readout that produces nothing is `SECONDARY_UNAVAILABLE` and
**refuses**, never "no jobs running". An instrument that produced no outcome
must not be read as producing a negative one -- which is exactly the mistake
the primary made.
"""
from __future__ import annotations

import argparse
import json
import sys

# Slurm states that mean the job has NOT finished. Anything outside this set is
# terminal (COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, ...).
LIVE_STATES = frozenset({
    "PENDING", "RUNNING", "SUSPENDED", "CONFIGURING", "COMPLETING",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING",
    "STAGE_OUT", "STOPPED", "RESV_DEL_HOLD", "SPECIAL_EXIT",
})


def parse_states(text: str | None) -> list[str]:
    """Slurm state tokens from one readout's raw stdout.

    `sacct` decorates cancellations as `CANCELLED by 12345`, and both tools can
    emit blank lines. The first whitespace token of each non-blank line is the
    state; everything else is discarded.

    `None` and the empty string are DIFFERENT and both are preserved by
    returning an empty list -- the caller distinguishes them through
    `produced_output`, because that distinction is the entire point of the
    module.
    """
    if not text:
        return []
    out = []
    for line in text.splitlines():
        tok = line.strip().split()
        if tok:
            out.append(tok[0].upper())
    return out


def corroborate(primary_raw: str | None, secondary_raw: str | None,
                primary_source: str = "squeue",
                secondary_source: str = "sacct") -> dict:
    """The guard. Returns a verdict dict; `ok` is False whenever it refuses.

    THE FOUR OUTCOMES, and the predicate is total over both readouts
    (gotcha 24) INCLUDING the case where either produced nothing:

      * `SECONDARY_UNAVAILABLE`   -- the second readout produced NO OUTPUT.
        **Refuses.** Silence and success are the same bytes (gotcha 28), and
        this is the primary's own failure signature, so it must not be the
        thing that clears it.
      * `UNCORROBORATED_COMPLETE` -- the primary says done, the secondary shows
        live jobs. **Refuses.** This is AZ20's exact case and it is the reason
        the module exists.
      * `INCOMPLETE`              -- jobs are still live somewhere. Not an
        error; the caller keeps waiting.
      * `CORROBORATED_COMPLETE`   -- both readouts agree nothing is live. The
        only outcome that proceeds.

    Note `primary_raw=None` (the command itself failed) is NOT treated as "no
    jobs": it produces no live states, so completion still requires the
    secondary to have produced output and to agree. A wait cannot clear itself
    by failing to run.
    """
    p_states = parse_states(primary_raw)
    s_states = parse_states(secondary_raw)
    p_live = [s for s in p_states if s in LIVE_STATES]
    s_live = [s for s in s_states if s in LIVE_STATES]
    secondary_produced = bool(s_states)

    detail = {
        "primary_source": primary_source,
        "secondary_source": secondary_source,
        "primary_produced_output": bool(p_states),
        "secondary_produced_output": secondary_produced,
        "primary_live_states": sorted(set(p_live)),
        "secondary_live_states": sorted(set(s_live)),
        "n_primary_live": len(p_live),
        "n_secondary_live": len(s_live),
    }

    if not secondary_produced:
        return {
            "verdict": "SECONDARY_UNAVAILABLE", "ok": False, "complete": False,
            "why": f"the {secondary_source} readout produced no output. "
                   "Silence is not corroboration -- an instrument that "
                   "produced no outcome must not be read as producing a "
                   "negative one (gotcha 28), and empty output is the "
                   "PRIMARY's own failure signature.",
            **detail,
        }
    if p_live:
        return {"verdict": "INCOMPLETE", "ok": True, "complete": False,
                "why": f"{len(p_live)} live job state(s) in {primary_source}",
                **detail}
    if s_live:
        return {
            "verdict": "UNCORROBORATED_COMPLETE", "ok": False,
            "complete": False,
            "why": f"{primary_source} reports nothing live while "
                   f"{secondary_source} shows {len(s_live)}: "
                   f"{sorted(set(s_live))}. This is a FALSE COMPLETION "
                   "signal (AZ20 item 3) and the guard refuses rather than "
                   "letting the caller read incomplete results.",
            **detail,
        }
    return {"verdict": "CORROBORATED_COMPLETE", "ok": True, "complete": True,
            "why": "both readouts agree that no job is live", **detail}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary", required=True,
                   help="path to the PRIMARY readout's captured stdout "
                        "(squeue), or the literal '-' for stdin")
    p.add_argument("--secondary", required=True,
                   help="path to the INDEPENDENT readout's captured stdout "
                        "(sacct)")
    p.add_argument("--primary-source", default="squeue")
    p.add_argument("--secondary-source", default="sacct")
    p.add_argument("--json", action="store_true",
                   help="emit the verdict as JSON on stdout")
    args = p.parse_args()

    def read(path: str) -> str | None:
        if path == "-":
            return sys.stdin.read()
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            # A MISSING readout is not an empty one, and neither is a
            # completion: both fall through to the same refusal.
            return None

    v = corroborate(read(args.primary), read(args.secondary),
                    args.primary_source, args.secondary_source)
    print(json.dumps(v, indent=2) if args.json
          else f"{v['verdict']}: {v['why']}")
    # EXIT CODES, so an `&&` or an `afterok` chain can gate on this
    # (gotcha 13 extension 1 -- a validator whose failure is unreadable by the
    # thing gating on it is worse than no validator):
    #   0  complete and corroborated  -> proceed
    #   2  still running              -> keep waiting; NOT an error
    #   1  refused                    -> stop
    raise SystemExit(0 if v["complete"] else (2 if v["ok"] else 1))


if __name__ == "__main__":
    main()
