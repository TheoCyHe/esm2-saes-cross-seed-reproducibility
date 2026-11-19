"""
Exact per-concept AP restricted to the concept's positive rows -- P1 restructure.

WHY THE OLD PATH IS SLOW. `average_precision_segmented` scans ALL non-zeros
(nnz ~ 567M at full-eval scale) once PER CONCEPT. Cost is
`nnz x n_concepts`, independent of how rare a concept is -- 822 sub-typed
concepts, most with a handful of positives, still pay the full nnz scan each.

THE EXACT REFORMULATION, not an approximation. In the closed form,
`contrib = lab * tp / rank` is IDENTICALLY ZERO wherever `lab == 0` (a
non-positive row). Summing only over the concept's POSITIVE rows therefore
computes the exact same sum with the (zero-valued) negative terms omitted --
not an approximation of it. `tp` at a touched (positive) entry is the
cumulative count of the concept's positives at or above that entry's rank
within its feature's block; since every entry considered here already IS a
positive, "cumulative count of positives seen so far in rank order" is
computed directly, without needing the negatives at all. Untouched features
(the concept fires on none of their positions) get exactly the closed-form
tail term (`n_positives / n_rows`, the base rate) -- identical to
`average_precision_segmented`'s silent-feature case.

COST. Building the one-time (per seed) ROW-major index costs ~nnz, the same
order as the sort `sparse_activations` already performs. Per concept, cost is
bounded by the total number of (feature, rank) touches its positive rows
have -- at most k per row (TopK's k), so `<= k * n_positives`, NOT nnz.

A ROW DOES NOT ALWAYS FIRE EXACTLY k FEATURES. `TopKSAE.forward` ReLUs
pre-activations BEFORE the top-k gate (`encode_pre`), so if fewer than k
neurons have strictly positive pre-activation, `topk` still selects k
indices but some carry value exactly 0.0 -- `z.nonzero()` (what
`sparse_activations` gathers) then returns FEWER than k entries for that row.
This is not a hypothetical: it is why the row index below is a proper CSR
structure (`row_offsets`, built via bincount + cumsum over the FULL 0..n_rows
range so a zero-count row is a valid, explicit empty slice) rather than a
fixed-width `(n_rows, k)` reshape, which would silently misalign the moment
any row fires fewer than k features.
"""

from __future__ import annotations

import torch


def argmax_lowest_index(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Deterministic argmax: on a tie, the LOWEST index wins -- explicit, not
    inherited from whatever the backend's argmax happens to do.

    g_f is a max over concepts, and its ARGMAX identity feeds downstream
    diagnostics (which concept a feature "aligns to"), so a tie-break that
    silently varies by device or library version would make that identity
    non-reproducible. Verified empirically (not assumed): torch.argmax and
    numpy.argmax both return the first (lowest-index) occurrence on CPU, but
    this is not exercised as a GUARANTEE here -- ties are real, not
    hypothetical (concepts with identical positive row sets, e.g. a pooled
    parent retained alongside its single surviving child, produce an EXACT
    AP tie), so the tie-break is constructed explicitly: mask every
    non-maximal position to a sentinel index one past the valid range, then
    take the ordinary argmin. The true max positions keep their real
    (distinct, in-range) indices, so the smallest surviving index is
    unambiguously the answer regardless of how ties among the SENTINEL value
    itself would be broken -- there is always at least one true max, so the
    sentinel is never the reported result.
    """
    n = x.shape[dim]
    idx_shape = [1] * x.dim()
    idx_shape[dim] = n
    idx = torch.arange(n, device=x.device).view(idx_shape).expand_as(x)
    max_val = x.max(dim=dim, keepdim=True).values
    is_max = x == max_val
    masked_idx = torch.where(is_max, idx, torch.full_like(idx, n))
    return masked_idx.argmin(dim=dim)


def build_row_index(
    feat: torch.Tensor,          # (nnz,) feature id, sorted (feature asc, value desc)
    pos: torch.Tensor,           # (nnz,) row id, aligned with feat
    seg_start: torch.Tensor,     # (n_features,) start of each feature's block
    n_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-time (per seed) transpose of the feature-major sparse layout into a
    row-major CSR structure, so a later per-concept step can look up a small
    set of query ROWS directly instead of scanning every non-zero.

    Returns (feat_by_row, rank_by_row, row_offsets):
      - feat_by_row, rank_by_row: (nnz,) non-zeros reordered by row ascending.
        `rank_by_row[i]` is that entry's rank (1-based) WITHIN ITS FEATURE'S
        ORIGINAL block -- i.e. computed against the full (feature-major)
        ordering, exactly what `average_precision_segmented` calls `rank`.
        Order WITHIN a row does not matter (nothing downstream depends on it).
      - row_offsets: (n_rows+1,) CSR offsets. Row r's entries are
        `feat_by_row[row_offsets[r]:row_offsets[r+1]]` -- 0 to k of them.
    """
    nnz = feat.numel()
    device = feat.device
    idx = torch.arange(nnz, device=device, dtype=torch.int64)
    rank = idx - seg_start[feat] + 1                      # 1-based, feature-block rank

    row_order = torch.argsort(pos, stable=True)
    feat_by_row = feat[row_order]
    rank_by_row = rank[row_order]
    pos_sorted = pos[row_order]

    counts = torch.bincount(pos_sorted, minlength=n_rows)
    row_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.cumsum(counts, 0),
    ])
    return feat_by_row, rank_by_row, row_offsets


def gather_touches(
    feat_by_row: torch.Tensor, rank_by_row: torch.Tensor, row_offsets: torch.Tensor,
    query_rows: torch.Tensor,    # (P,) row indices, need not be sorted or unique-checked
                                 # by this function (callers pass deduplicated concept
                                 # positive rows; duplicates would just double-count)
) -> tuple[torch.Tensor, torch.Tensor]:
    """(feature, rank) touches across a batch of query rows, vectorized -- no
    Python-level per-row loop. Output size = sum of each row's actual
    non-zero count (<= k each, via the CSR offsets, never assumed to be
    exactly k)."""
    start = row_offsets[query_rows]
    end = row_offsets[query_rows + 1]
    count = end - start
    total = int(count.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.int64, device=query_rows.device)
        return empty, empty

    touch_row = torch.repeat_interleave(
        torch.arange(query_rows.numel(), device=query_rows.device), count)
    row_start_cum = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=count.device),
        torch.cumsum(count, 0)[:-1],
    ])
    within_row = (torch.arange(total, device=query_rows.device)
                 - row_start_cum[touch_row])
    data_idx = start[touch_row] + within_row
    return feat_by_row[data_idx], rank_by_row[data_idx]


def average_precision_from_touches(
    touched_feat: torch.Tensor,  # (m,) feature id per touch (a concept-positive row
                                 # firing on that feature), any order
    touched_rank: torch.Tensor,  # (m,) that touch's rank within the feature's block
    n_features: int,
    n_positives: int,
    n_rows: int,
) -> torch.Tensor:
    """AP for every feature against ONE concept, from ONLY its positive-row
    touches -- exact, per the module docstring's `contrib` argument, not an
    approximation of `average_precision_segmented`.
    """
    device = touched_feat.device if touched_feat.numel() else torch.device("cpu")
    if n_positives <= 0:
        return torch.zeros(n_features, dtype=torch.float64, device=device)

    base_rate = n_positives / n_rows
    ap = torch.full((n_features,), base_rate, dtype=torch.float64, device=device)
    if touched_feat.numel() == 0:
        return ap

    # Sort by rank ascending first (recovers each feature's activation-DESC
    # order), THEN a stable sort by feature -- stability preserves the
    # rank-ascending order within each feature's group.
    o1 = torch.argsort(touched_rank, stable=True)
    tf, tr = touched_feat[o1], touched_rank[o1]
    o2 = torch.argsort(tf, stable=True)
    tf, tr = tf[o2], tr[o2].to(torch.float64)

    m = tf.numel()
    counts = torch.bincount(tf, minlength=n_features)
    seg = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.cumsum(counts, 0)[:-1],
    ])
    idx = torch.arange(m, device=device, dtype=torch.int64)
    # cumulative count of TOUCHED (= positive) entries at or before this one,
    # within this touch's feature -- exactly the original `tp` (every entry
    # here already satisfies lab=1, so "how many touches so far" IS "how many
    # positives at or above this rank").
    tp = (idx - seg[tf] + 1).to(torch.float64)

    contrib = tp / tr
    ap_fire = torch.zeros(n_features, dtype=torch.float64, device=device)
    ap_fire.index_add_(0, tf, contrib)
    p_fire = torch.zeros(n_features, dtype=torch.float64, device=device)
    p_fire.index_add_(0, tf, torch.ones(m, dtype=torch.float64, device=device))

    touched_mask = torch.zeros(n_features, dtype=torch.bool, device=device)
    touched_mask[tf.unique()] = True
    tail = (1.0 - p_fire / n_positives) * base_rate
    ap_exact = ap_fire / n_positives + tail
    ap = torch.where(touched_mask, ap_exact, ap)
    return ap


# ---------------------------------------------------------------------------
# THE NARROW, CHUNKED PATH (amendment CT, §CT29-D2 item 1, route (b1)).
#
# At k = 128 the int64 row index alone (two nnz-length arrays) filled a 32 GB
# device before any concept's arithmetic ran. Two changes keep the null's AP
# arithmetic on the GPU without changing a value:
#   - the DEVICE COPY of the index is narrowed (feature ids to int16 or int32,
#     ranks to int32), refusing unless every value fits, so nothing is rounded;
#   - the touch gather and the AP's sorts run in chunks of at most
#     `max_touches` touches. Each feature's AP is computed from exactly the
#     touches, in exactly the order, the unchunked path uses (a feature's
#     touches are never split across chunks), so on CPU -- where index_add_
#     sums in index order -- the two paths are bitwise equal. On CUDA
#     index_add_ is order-nondeterministic already (gotcha 35).
# Opt-in through null_pilot --narrow-row-index; nothing else calls these.
# ---------------------------------------------------------------------------

def narrow_row_index(feat_by_row: torch.Tensor, rank_by_row: torch.Tensor,
                     n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature ids to int16 (int32 from 32,768 features up) and ranks to int32,
    REFUSING unless every value is representable -- a narrowing that could
    wrap would silently re-attribute touches to other features. The int16
    limit is 32,767 features, not 32,768: the chunked AP compares ids against
    the range bound f1 = n_features, which must itself fit (at 32,768 it wraps
    to -32,768 and the last range's touches vanish -- found by review)."""
    feat_dtype = torch.int16 if n_features <= 2 ** 15 - 1 else torch.int32
    if n_features > 2 ** 31:
        raise ValueError(f"n_features {n_features} does not fit int32")
    if feat_by_row.numel():
        lo, hi = int(feat_by_row.min()), int(feat_by_row.max())
        if lo < 0 or hi >= n_features:
            raise ValueError(f"feature ids span [{lo}, {hi}], outside "
                             f"[0, {n_features}): refusing to narrow")
        rlo, rhi = int(rank_by_row.min()), int(rank_by_row.max())
        if rlo < 1 or rhi >= 2 ** 31:
            raise ValueError(f"ranks span [{rlo}, {rhi}], outside [1, 2^31): "
                             "refusing to narrow")
    return feat_by_row.to(feat_dtype), rank_by_row.to(torch.int32)


def gather_touches_chunked(
    feat_by_row: torch.Tensor, rank_by_row: torch.Tensor,
    row_offsets: torch.Tensor, query_rows: torch.Tensor, max_touches: int,
    stats: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`gather_touches` over chunks of query rows holding at most `max_touches`
    touches each (a single row is never split; it holds at most k). The
    pieces are concatenated in row order, which is the sequence the unchunked
    call returns. Outputs keep the index's dtypes."""
    if max_touches < 1:
        raise ValueError("max_touches must be >= 1")
    count = row_offsets[query_rows + 1] - row_offsets[query_rows]
    total = int(count.sum().item())
    if total <= max_touches:
        return gather_touches(feat_by_row, rank_by_row, row_offsets, query_rows)
    if stats is not None:
        stats["chunked_gathers"] = stats.get("chunked_gathers", 0) + 1
    cum = torch.cumsum(count, 0)
    fs, rs = [], []
    start, base, n = 0, 0, query_rows.numel()
    while start < n:
        end = int(torch.searchsorted(cum, base + max_touches, right=True).item())
        end = max(end, start + 1)
        piece_total = int(cum[end - 1].item()) - base
        if piece_total > 0:
            f, r = gather_touches(feat_by_row, rank_by_row, row_offsets,
                                  query_rows[start:end])
            fs.append(f)
            rs.append(r)
        base += piece_total
        start = end
    return torch.cat(fs), torch.cat(rs)


def feature_chunk_bounds(per_feature: torch.Tensor,
                         max_touches: int) -> list[tuple[int, int]]:
    """Greedy, contiguous feature ranges [f0, f1) whose touch counts sum to at
    most `max_touches`; a feature holding more than that is a range alone.
    Every feature lies in exactly one range."""
    if max_touches < 1:
        raise ValueError("max_touches must be >= 1")
    cum = torch.cumsum(per_feature.to(torch.int64).cpu(), 0)
    n = cum.numel()
    bounds, f0, base = [], 0, 0
    while f0 < n:
        f1 = int(torch.searchsorted(cum, base + max_touches, right=True).item())
        f1 = max(f1, f0 + 1)
        bounds.append((f0, f1))
        base = int(cum[f1 - 1].item())
        f0 = f1
    return bounds


def average_precision_from_touches_chunked(
    touched_feat: torch.Tensor, touched_rank: torch.Tensor, n_features: int,
    n_positives: int, n_rows: int, max_touches: int,
    stats: dict | None = None,
) -> torch.Tensor:
    """`average_precision_from_touches`, with the sorts run over feature ranges
    of at most `max_touches` touches when the draw has more than that. Each
    range's features see exactly their own touches, in the order the
    unchunked path sorts them, so the per-feature arithmetic is unchanged.
    Touches are widened to int64 before the arithmetic."""
    m = touched_feat.numel()
    if m <= max_touches:
        return average_precision_from_touches(
            touched_feat.to(torch.int64), touched_rank.to(torch.int64),
            n_features, n_positives, n_rows)
    if n_positives <= 0:
        raise ValueError(f"{m} touches with n_positives={n_positives}")
    per_feature = torch.bincount(touched_feat, minlength=n_features)
    bounds = feature_chunk_bounds(per_feature, max_touches)
    if stats is not None:
        stats["chunked_ap_evaluations"] = stats.get("chunked_ap_evaluations", 0) + 1
        stats["max_feature_chunks"] = max(stats.get("max_feature_chunks", 0),
                                          len(bounds))
    ap = torch.empty(n_features, dtype=torch.float64, device=touched_feat.device)
    for f0, f1 in bounds:
        sel = ((touched_feat >= f0) & (touched_feat < f1)).nonzero().squeeze(1)
        part = average_precision_from_touches(
            touched_feat[sel].to(torch.int64) - f0,
            touched_rank[sel].to(torch.int64), f1 - f0, n_positives, n_rows)
        ap[f0:f1] = part.to(ap.device)
    return ap
