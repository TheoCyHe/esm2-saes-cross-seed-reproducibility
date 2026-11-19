"""THE MATCHING-RULE SENSITIVITY. Governed by PREREGISTRATION amendment CJ3,
committed at 6db4104 BEFORE the run it governs (job 22717004).

WHAT THIS IS, AND WHAT IT IS NOT. A sensitivity analysis on the matching RULE,
in the same register as the existing threshold sensitivity curve. **MNN on
decoder cosine remains the pre-registered primary and the figure of record.** No
rule is promoted, none is averaged with another, no selection procedure is built.

THE THREE RULES, attributed in docs/LITERATURE.md 2a. ATTRIBUTIONS FOR RULES 2
AND 3 WERE CORRECTED AT AMENDMENT CK AND AGAIN AT AMENDMENT CP (corrections
entries 211-212); the superseded ones are shown so the change is legible rather
than silent:
  rule 1  MNN on decoder cosine                    -- ours, and Borobia's
  rule 2  Hungarian one-to-one on decoder cosine   -- OURS (sensitivity arm; no
          published user. CK attributed it to Jedryszek & Crook, whose "fraction
          paired" is max_j |S_ij| > 0.7, a one-way argmax rate -- CP, entry 212.
          Before CK: "the clinical comparator's", which names NO matching
          algorithm at all, so its 21% cannot enter a rule table -- CK1.)
  rule 3  Hungarian, ENCODER AND DECODER agreeing  -- PAULO & BELROSE Sec. 3 (their
          formal definition) AND JEDRYSZEK & CROOK Sec. 2.3: the same criterion.
          (CK's "different criterion" read their Figure 2 caption -- CP, entry 211.
          Before CK: "the seed-dependence comparator's", which was right by their
          Sec. 3. See encoder_decoder_similarity.)

One-way argmax is carried as CONTEXT and is NOT one of the three.

WHY THE PAIRS ARE UNORDERED (190, not 380). All three rules are SYMMETRIC in the
pair -- mutual-best-ness and a one-to-one assignment are both direction-free --
and both dictionaries are 10,240 wide, so the fraction is identical either way.
One-way argmax is NOT symmetric and is computed A->B and labelled as such.

A FACT THE PAPERS DO NOT STATE, and rule 3 turns on it: decoder columns are
unit-norm (the trainer renormalises after every step) but ENCODER ROWS ARE NOT
(ours run 0.960 to 1.720). An encoder COSINE therefore needs an explicit row
normalisation; a raw W_enc @ W_enc.T is an inner product that silently ranks by
row magnitude. `cosine_matrix` normalises both sides for exactly this reason.

EXTENDED AT AMENDMENT CO, and the extension is what the CJ run could not do.
CJ reduced each rule's per-feature match vector to a MEAN on the line that
computed it, so no per-feature counts survived and the DISTRIBUTION of match
counts was unrecoverable without re-running. `mnn_partner`,
`encoder_decoder_partner` and `partner_view_from_b` expose the pairing itself,
so an unordered pair can be credited to BOTH its anchors and a per-(anchor,
feature) counter accumulated. Nothing about rules 1-3 changes.

X4's CONDITIONAL, AND WHY THIS RUNS AT FULL SCALE. X4 said Hungarian on a dense
10,240-square is O(n^3) and "may be impractical; if so" it runs subsampled. The
branch test was never run. Measured: 3.67 s for the decoder assignment and
5.35 s for the encoder one (job 22715989). The subsampled arm was unnecessary,
and amendment AB11's scale-invariance caveat was its standing price
(docs/PROJECT.md corrections entry 190, section 8 table row 193 of 230 -- the
number is reused; docs/CORRECTIONS.md entry 225).
"""
from __future__ import annotations

import numpy as np


NEG = -np.inf


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between the COLUMNS of A and the columns of B.

    Both sides are normalised here rather than assumed normalised -- see the
    module docstring on encoder rows.
    """
    An = A / np.linalg.norm(A, axis=0, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=0, keepdims=True)
    return An.T @ Bn


def oneway_similarity(C: np.ndarray) -> np.ndarray:
    """CONTEXT, not a rule. Row max: each A-feature's best B partner."""
    return C.max(axis=1)


def mnn_similarity(C: np.ndarray) -> np.ndarray:
    """RULE 1. Per-row mutual-nearest-neighbour similarity, or -inf.

    Mutual-best-ness does not depend on theta, so one vector decides the rule at
    EVERY threshold and the whole grid is a comparison rather than a
    recomputation (the implementation note in rf_marginal.py).
    """
    a_best_j = C.argmax(axis=1)
    b_best_i = C.argmax(axis=0)
    mutual = b_best_i[a_best_j] == np.arange(C.shape[0])
    return np.where(mutual, C.max(axis=1), NEG)


def hungarian_assignment(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RULE 2's solve. Returns (partner, similarity), both length n_rows.

    An unassigned row -- possible only when the matrix is not square -- carries
    partner -1 and similarity -inf.
    """
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(-C)
    partner = np.full(C.shape[0], -1, dtype=int)
    sim = np.full(C.shape[0], NEG, dtype=float)
    partner[r] = c
    sim[r] = C[r, c]
    return partner, sim


def mnn_partner(C: np.ndarray) -> np.ndarray:
    """RULE 1's PARTNER INDEX -- a SEPARATE function, not a wider return.

    `mnn_similarity` keeps its arity. Gotcha 27: when a helper needs to expose
    the permutation, add a function rather than grow a tuple, because the
    permutation is exactly the thing a positional check needs and a caller that
    ignores an extra return value gets no warning.

    Returns the B-column each A-row is mutually matched to, or -1 where the row
    has no mutual partner. NO theta is applied -- mutual-best-ness does not
    depend on it (see `mnn_similarity`), so one partner vector serves every
    threshold.
    """
    a_best_j = C.argmax(axis=1)
    b_best_i = C.argmax(axis=0)
    mutual = b_best_i[a_best_j] == np.arange(C.shape[0])
    return np.where(mutual, a_best_j, -1)


def partner_view_from_b(partner: np.ndarray, sim: np.ndarray,
                        n_b: int) -> np.ndarray:
    """THE OTHER HALF OF AN UNORDERED PAIR: the same matching, read off B.

    An unordered pair `{a, b}` contributes to BOTH anchors' per-feature counts.
    The pairing is one object, so B's vector is A's vector SCATTERED through
    the partner index -- never a second, independently solved matching, which
    could differ from A's under ties and would silently make the pair
    asymmetric.

    THE INJECTIVITY CHECK IS NOT DEFENSIVE PADDING. A partner array that
    repeats an index would overwrite on scatter and LOSE matched features
    without raising: a count that is silently too low, in a quantity whose
    whole content is a distribution over counts. Both matchings that feed this
    (mutual-nearest-neighbour and a one-to-one assignment) are injective by
    construction, so a repeat means the caller built the array wrongly.
    """
    partner = np.asarray(partner)
    ok = partner >= 0
    assigned = partner[ok]
    if np.unique(assigned).size != assigned.size:
        raise ValueError(
            f"partner repeats {assigned.size - np.unique(assigned).size} "
            "index(es) among its assigned entries; scattering would silently "
            "drop matched features. Both supported matchings are injective by "
            "construction, so this is a caller defect, not a case to absorb.")
    out = np.full(int(n_b), NEG, dtype=float)
    out[assigned] = np.asarray(sim)[ok]
    return out


def encoder_decoder_from_assignments(
    dec_partner: np.ndarray, dec_sim: np.ndarray,
    enc_partner: np.ndarray, enc_sim: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """RULE 3, from two assignments ALREADY SOLVED. The single definition.

    `encoder_decoder_similarity` and `encoder_decoder_partner` both delegate
    here, so rule 3 is defined in exactly one place. It exists because a caller
    that needs BOTH the similarity and the partner would otherwise solve the
    two Hungarian problems twice -- 9 s per pair at the full 10,240 square,
    which is ~28 minutes over 190 pairs, for an answer it already had.

    Returns `(partner, similarity)`, partner -1 and similarity -inf where the
    two matchings disagree.
    """
    agree = (dec_partner == enc_partner) & (dec_partner >= 0)
    sim = np.where(agree, np.minimum(dec_sim, enc_sim), NEG)
    return np.where(agree, dec_partner, -1), sim


def encoder_decoder_partner(Cd: np.ndarray, Ce: np.ndarray) -> np.ndarray:
    """RULE 3's PARTNER INDEX. Same reason as `mnn_partner` (gotcha 27)."""
    partner, _ = encoder_decoder_from_assignments(
        *hungarian_assignment(Cd), *hungarian_assignment(Ce))
    return partner


def encoder_decoder_similarity(
    Cd: np.ndarray, Ce: np.ndarray
) -> tuple[np.ndarray, float]:
    """RULE 3. Paulo & Belrose's formal definition of a shared latent (Section 3)
    and Jedryszek & Crook's "shared features" (Section 2.3): two Hungarian
    matchings that must agree, each clearing theta. Ours thresholds at >= (as
    their Section 3) on signed cosine (theirs absolute; P&B unstated).

    ATTRIBUTION CORRECTED AT AMENDMENT CK AND AGAIN AT AMENDMENT CP (corrections
    entries 211-212). CK re-attributed this function to Jedryszek & Crook alone,
    reading Paulo & Belrose's FIGURE 2 CAPTION -- "X is MATCHED TO a latent Y
    WITH WHICH it has cosine similarity greater than 0.7 according to BOTH the
    encoder and decoder weights", ONE matching then a two-sided threshold -- as
    their criterion. Their Section 3 defines it as two matchings that must
    agree, which is what this function does; the pre-CK attribution to "the
    seed-dependence comparator" was right by that definition.

    THE RELATION IS A STRICT SUBSET, and it is arithmetic rather than a
    hypothesis: where the two matchings agree on partner p, the values
    thresholded here ARE Cd[i, p] and Ce[i, p], so this set is the caption
    wording's set intersected with the agreement condition. Measured at 190
    pairs, the matchings agree for 87.73% of features, which bounds the caption
    wording on our data to [30.38%, 42.65%] (amendment CK3a; the FIGURE 2 CAPTION
    reading of the same paper, measured by `literal_two_sided_similarity` --
    variants D/E, 30.39/30.41%; docs/NUMBERS_OF_RECORD.md 13.2, 13.4).

    A feature counts only where the DECODER matching and the ENCODER matching
    assign it the SAME partner; its similarity is then the MINIMUM of the two,
    so a threshold on the result binds on BOTH.

    Returns (similarity, fraction_of_features_whose_matchings_agree).
    """
    partner, sim = encoder_decoder_from_assignments(
        *hungarian_assignment(Cd), *hungarian_assignment(Ce))
    return sim, float((partner >= 0).mean())


def literal_two_sided_similarity(
    Cd: np.ndarray, Ce: np.ndarray, match_on: str = "decoder"
) -> np.ndarray:
    """RULE 3-PRIME. PAULO & BELROSE's criterion, literally read (amendment CL).

    "X is MATCHED TO a latent Y WITH WHICH it has cosine similarity greater than
    0.7 according to BOTH the encoder and decoder weights" -- ONE matching, then
    a two-sided threshold on THAT pair. Distinct from rule 3
    (`encoder_decoder_similarity`), which requires TWO matchings to agree and is
    a strict subset of this.

    THE MATCHING'S COST MATRIX IS A SECOND UNSTATED DEGREE OF FREEDOM (CL1).
    Their wording names a matching but not its rule, and is symmetric in encoder
    and decoder, so nothing privileges either side. `match_on` is therefore
    REQUIRED to be stated by the caller and both readings are reported; there is
    no default that is defensible as "theirs".

    Returning the MINIMUM of the two similarities makes a threshold on the result
    a threshold on BOTH, which is what "according to BOTH" asks for.
    """
    if match_on == "decoder":
        partner, _ = hungarian_assignment(Cd)
    elif match_on == "encoder":
        partner, _ = hungarian_assignment(Ce)
    else:
        raise ValueError(
            f"match_on must be 'decoder' or 'encoder', got {match_on!r} -- "
            "the criterion does not determine it, so it cannot be guessed"
        )
    idx = np.arange(Cd.shape[0])
    assigned = partner >= 0
    safe = np.where(assigned, partner, 0)
    both = np.minimum(Cd[idx, safe], Ce[idx, safe])
    return np.where(assigned, both, NEG)


def fraction_at(sim: np.ndarray, theta: float) -> float:
    """Fraction of features whose matched similarity reaches theta.

    The denominator is ALWAYS the full feature count, never the matched count --
    that is what puts every rule on one scale (amendment CJ3.2).
    """
    return float((sim >= theta).mean())


def mnn_nested_in_oneway(C: np.ndarray, thetas) -> list:
    """CONTROL C2. Returns the thetas where MNN exceeds one-way -- a bug if any.

    Amendment W1 establishes MNN is nested inside one-way argmax, so a non-empty
    return is an implementation defect and the caller FAILS on it.
    """
    mnn, one = mnn_similarity(C), oneway_similarity(C)
    return [t for t in thetas if (mnn >= t).sum() > (one >= t).sum()]


def mnn_pairs_not_chosen_by_hungarian(C: np.ndarray, theta: float) -> int:
    """CONTROL C3. Counts MNN pairs at theta that Hungarian did NOT choose.

    W1's constructive counterexample: MNN is nested in one-way but NOT in
    Hungarian. A ZERO here at full dictionary scale would mean the instrument is
    returning a vacuous agreement, so this is checked rather than assumed
    (gotcha 28). Measured across 190 anchor pairs: 11,351 in total, and zero
    pairs with a zero count.
    """
    a_best_j = C.argmax(axis=1)
    mnn = mnn_similarity(C)
    partner, _ = hungarian_assignment(C)
    idx = np.where(mnn >= theta)[0]
    return int((partner[idx] != a_best_j[idx]).sum())
