"""Section 2, Path A: a ball position on every scored frame, and how far to trust it.

The tracker sees the ball on a bit over half the frames. Later sections need a position on every
scored frame, plus a number that says when that position is a guess. This module provides both, as
evidence only: it never says who has the ball.

    import ball
    out = ball.load()       # dict: ball_xy (n, 2), ball_conf (n,), ball_method (n,)

v0, in three steps:

  1. Outliers. A ball point is dropped when it is too far from BOTH its previous and next observed
     neighbours to be reached at MAX_BALL_SPEED_MS (plus OUTLIER_SLACK_M of tracker jitter). The
     allowance grows with the time between the points, so a point after a long gap is not punished
     for the gap. Only isolated spikes go: a real fast pass is consistent with its neighbours.
     Points beyond OFFPITCH_MARGIN_M outside the pitch lines are dropped too. Points outside the
     lines but inside the margin are KEPT, because Section 3 reads "ball off the pitch" from them.
  2. Gaps. Inside each scored half block, a gap between two observations is a straight line. At a
     block edge with an observation on one side only, the nearest observation is held. Nothing is
     ever carried across halftime, and frames outside the scored blocks are not filled.
  3. Confidence. 1.0 where observed. In a gap (v0, to be calibrated in v1):

         conf = CONF_FLOOR + (1 - CONF_FLOOR) * exp(-d / CONF_TAU_DIST_S) * exp(-L / CONF_TAU_LEN_S)

     d is the time in seconds to the nearest observation, L the gap length in seconds (for a held
     run, the length of that run). It falls smoothly from about 0.95 next to an observation to the
     floor in the middle of a long gap. Outside the scored blocks it is 1.0 where observed, else 0.

ball_method says how each frame got its value: 0 observed, 1 linear, 2 hold, 3 player-based (in
v0 only used when a whole block has no observation, as a last resort: the player centroid), 255
outside the scored blocks.

The result is cached in cache/ (gitignored: it is frame-level tracking). The signature records the
tracking source, VERSION and a hash of the Params block, and is written last, so a change in any
of them, or an interrupted build, means a rebuild.

    python starter/ball.py            # build or load the cache, print shapes
    python starter/ball.py --report   # aggregate gap report, safe to share
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
import load_data  # noqa: E402

# Params
VERSION = "ball-v0"
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]

MAX_BALL_SPEED_MS = 40.0   # faster than any real shot over a frame or two
OUTLIER_SLACK_M = 1.0      # tracker jitter allowed on top of the speed limit
OUTLIER_MAX_PASSES = 5     # removal is repeated so a point is judged against accepted neighbours
OFFPITCH_MARGIN_M = 8.0    # beyond this outside the lines a ball position is not credible

CONF_FLOOR = 0.05
CONF_TAU_DIST_S = 1.0      # v0 guess, to be calibrated in v1
CONF_TAU_LEN_S = 4.0       # v0 guess, to be calibrated in v1

GAP_BUCKETS = ((1, 5), (6, 12), (13, 25), (26, 75), (76, 250), (251, None))

CACHE_DIR = load_data.CACHE_DIR
SIGNATURE_FILE = "ball_signature.json"
ARRAYS = ("ball_xy", "ball_conf", "ball_method")
DTYPE = np.float32

OBSERVED, LINEAR, HOLD, PLAYERS, UNSCORED = 0, 1, 2, 3, 255
METHOD_NAMES = {OBSERVED: "observed", LINEAR: "linear", HOLD: "hold", PLAYERS: "players",
                UNSCORED: "unscored"}

PARAMS = {
    "MAX_BALL_SPEED_MS": MAX_BALL_SPEED_MS, "OUTLIER_SLACK_M": OUTLIER_SLACK_M,
    "OUTLIER_MAX_PASSES": OUTLIER_MAX_PASSES, "OFFPITCH_MARGIN_M": OFFPITCH_MARGIN_M,
    "CONF_FLOOR": CONF_FLOOR, "CONF_TAU_DIST_S": CONF_TAU_DIST_S,
    "CONF_TAU_LEN_S": CONF_TAU_LEN_S, "FPS": FPS, "PITCH": list(PITCH),
}


def scored_blocks(game=GAME):
    """Scored half blocks as half-open (start, end). games.py gives inclusive ends."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def _segments(n_frames, blocks):
    """The blocks plus the stretches between them, so nothing links frames across halftime."""
    cuts = sorted({0, n_frames} | {edge for block in blocks for edge in block})
    return [(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:]) if hi > lo]


def _runs(mask):
    """Half-open (start, end) of every run of True in a 1-d bool array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def find_gaps(ball_raw, blocks):
    """Every run of missing ball inside the blocks, half-open, in frame order."""
    missing = np.isnan(ball_raw[:, 0])
    gaps = []
    for lo, hi in blocks:
        gaps.extend((lo + s, lo + e) for s, e in _runs(missing[lo:hi]))
    return gaps


def beyond_margin(xy, margin=OFFPITCH_MARGIN_M, pitch=PITCH):
    """True where a position is more than `margin` outside the pitch lines. NaN gives False."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return ((xy[:, 0] < -margin) | (xy[:, 0] > length + margin)
                | (xy[:, 1] < -margin) | (xy[:, 1] > width + margin))


def outside_lines(xy, pitch=PITCH):
    """True where a position is outside the pitch lines at all. NaN gives False."""
    return beyond_margin(xy, 0.0, pitch)


def _spike_pass(clean, lo, hi, fps):
    """One pass of the isolated-spike rule over [lo, hi). Returns the number removed."""
    idx = np.flatnonzero(~np.isnan(clean[lo:hi, 0])) + lo
    if len(idx) < 3:
        return 0
    points = clean[idx].astype(np.float64)
    step = np.hypot(*np.diff(points, axis=0).T)
    allowed = MAX_BALL_SPEED_MS * np.diff(idx) / fps + OUTLIER_SLACK_M
    excess = step - allowed
    # Interior point k is a spike candidate if both of its steps are too long.
    score = np.minimum(excess[:-1], excess[1:])
    candidate = score > 0
    if not candidate.any():
        return 0
    # Where candidates touch, drop only the worst of them this pass; the next pass re-judges the
    # others against the neighbours that remain.
    padded = np.concatenate(([-np.inf], np.where(candidate, score, -np.inf), [-np.inf]))
    worst = candidate & (score >= padded[:-2]) & (score > padded[2:])
    clean[idx[1:-1][worst]] = np.nan
    return int(worst.sum())


def _remove_outliers(ball_raw, blocks, fps=FPS):
    """(ball_clean, n_spikes, n_beyond_margin). See remove_outliers."""
    clean = np.array(ball_raw, dtype=DTYPE, copy=True)
    far = beyond_margin(clean)
    clean[far] = np.nan
    n_spikes = 0
    for lo, hi in _segments(len(clean), blocks):
        for _ in range(OUTLIER_MAX_PASSES):
            removed = _spike_pass(clean, lo, hi, fps)
            n_spikes += removed
            if not removed:
                break
    return clean, n_spikes, int(far.sum())


def remove_outliers(ball_raw, blocks):
    """(ball_clean, n_removed): ball_raw without isolated spikes and far off-pitch points.

    Neighbours are looked for within a block (or within one stretch between blocks), never across
    a block edge. The first and last observation of a stretch have only one neighbour, so the
    two-sided rule cannot judge them and they are kept.
    """
    clean, n_spikes, n_far = _remove_outliers(ball_raw, blocks)
    return clean, n_spikes + n_far


def gap_confidence(distance_frames, gap_frames, fps=FPS):
    """v0 confidence for a filled frame. See the module docstring. To be calibrated in v1."""
    d = np.asarray(distance_frames, dtype=np.float64) / fps
    length = np.asarray(gap_frames, dtype=np.float64) / fps
    trust = np.exp(-d / CONF_TAU_DIST_S) * np.exp(-length / CONF_TAU_LEN_S)
    return (CONF_FLOOR + (1.0 - CONF_FLOOR) * trust).astype(DTYPE)


def _player_centroid(players):
    """Mean of all tracked players per frame, pitch centre where nobody is tracked."""
    present = ~np.isnan(players[..., 0])
    count = present.sum(axis=1)
    total = np.where(present[..., None], players, 0.0).sum(axis=1)
    centre = np.array(PITCH, dtype=np.float64) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count[:, None]
    return np.where(count[:, None] > 0, mean, centre).astype(DTYPE)


def _fill_block(clean, players, lo, hi, xy, conf, method, fps):
    """Fill [lo, hi) in place: line between observations, hold at the edges."""
    frames = np.arange(lo, hi)
    observed = ~np.isnan(clean[lo:hi, 0])
    idx = frames[observed]
    method[lo:hi][observed] = OBSERVED
    if len(idx) == 0:
        # No observation in the whole block: the only estimate left is where the players are.
        xy[lo:hi] = _player_centroid(players[lo:hi])
        conf[lo:hi] = CONF_FLOOR
        method[lo:hi] = PLAYERS
        return

    missing = ~observed
    for axis in (0, 1):
        # np.interp is a straight line between observations and holds the end values beyond them.
        xy[lo:hi, axis][missing] = np.interp(frames[missing], idx, clean[idx, axis])

    big = hi + 1
    prev = np.maximum.accumulate(np.where(observed, frames, -1))
    nxt = np.minimum.accumulate(np.where(observed, frames, big)[::-1])[::-1]
    has_prev, has_next = prev >= 0, nxt < big
    linear = missing & has_prev & has_next
    hold = missing & ~linear

    distance = np.minimum(np.where(has_prev, frames - prev, big),
                          np.where(has_next, nxt - frames, big))
    run = np.where(linear, nxt - prev - 1, np.where(has_prev, hi - 1 - prev, nxt - lo))

    method[lo:hi][linear] = LINEAR
    method[lo:hi][hold] = HOLD
    conf[lo:hi][missing] = gap_confidence(distance[missing], run[missing], fps)


def estimate_ball(tracking, blocks=None, fps=FPS):
    """dict(ball_xy, ball_conf, ball_method) for one match. See the module docstring.

    tracking is the dict from load_data.load_tracking() (only ball_raw and players are read).
    blocks are half-open scored blocks; by default the ones in games.py.
    """
    blocks = scored_blocks() if blocks is None else [tuple(b) for b in blocks]
    clean, _, _ = _remove_outliers(tracking["ball_raw"], blocks, fps)
    n_frames = len(clean)

    xy = clean.copy()
    conf = np.where(np.isnan(clean[:, 0]), 0.0, 1.0).astype(DTYPE)
    method = np.full(n_frames, UNSCORED, dtype=np.uint8)
    for lo, hi in blocks:
        lo, hi = max(0, lo), min(n_frames, hi)
        if hi > lo:
            _fill_block(clean, tracking["players"], lo, hi, xy, conf, method, fps)
    return {"ball_xy": xy, "ball_conf": conf, "ball_method": method}


def params_hash():
    text = json.dumps(PARAMS, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_signature(blocks=None):
    """What a cached result was built from. Needs the match data (for the tracking signature)."""
    blocks = scored_blocks() if blocks is None else blocks
    return {"tracking": load_data.source_signature(), "version": VERSION,
            "params": params_hash(), "blocks": [list(b) for b in blocks]}


def _normalise(signature):
    # JSON turns tuples into lists; compare like with like.
    return json.loads(json.dumps(signature, sort_keys=True))


def _cache_is_current(signature, cache_dir):
    path = os.path.join(cache_dir, SIGNATURE_FILE)
    if not os.path.isfile(path):
        return False
    if not all(os.path.isfile(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS):
        return False
    with open(path) as handle:
        return json.load(handle) == _normalise(signature)


def build(tracking=None, refresh=False, cache_dir=CACHE_DIR, blocks=None, signature=None):
    """Compute the ball arrays and cache them, or return the cache if it is current.

    With no arguments this reads the match through load_data. A tracking dict passed in is taken
    to be load_data.load_tracking() output for the default game. Tests pass their own tracking,
    blocks, cache_dir and signature, so they never touch the match data.
    """
    if signature is None:
        signature = cache_signature(blocks)
    if not refresh and _cache_is_current(signature, cache_dir):
        return {name: np.load(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS}

    if tracking is None:
        tracking = load_data.load_tracking()
    out = estimate_ball(tracking, blocks)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    for name in ARRAYS:
        np.save(os.path.join(cache_dir, name + ".npy"), out[name])
    # Written last, so an interrupted build never looks complete.
    with open(os.path.join(cache_dir, SIGNATURE_FILE), "w") as handle:
        json.dump(_normalise(signature), handle, sort_keys=True)
    return out


def load(refresh=False):
    """The one line for everyone: dict(ball_xy, ball_conf, ball_method), rebuilt if stale."""
    return build(refresh=refresh)


# ---------------------------------------------------------------------------------------------
# Report. Aggregates only: shapes, counts, shares. Nothing per frame, so it is safe to share.
# ---------------------------------------------------------------------------------------------

def _bucket_label(low, high):
    return "{}-{}".format(low, high) if high else ">{}".format(low - 1)


def gap_table(gaps):
    """[(label, n_gaps, n_frames)] by GAP_BUCKETS."""
    lengths = np.array([end - start for start, end in gaps], dtype=np.int64)
    rows = []
    for low, high in GAP_BUCKETS:
        sel = (lengths >= low) & ((lengths <= high) if high else True)
        rows.append((_bucket_label(low, high), int(sel.sum()), int(lengths[sel].sum())))
    return rows


def report_lines(tracking, out, blocks=None):
    """The --report block as a list of lines."""
    blocks = scored_blocks() if blocks is None else blocks
    ball_raw = tracking["ball_raw"]
    n_frames = len(ball_raw)
    scored = np.zeros(n_frames, dtype=bool)
    for lo, hi in blocks:
        scored[lo:hi] = True
    n_scored = int(scored.sum())

    clean, n_spikes, n_far = _remove_outliers(ball_raw, blocks)
    raw_obs = ~np.isnan(ball_raw[:, 0])
    gaps = find_gaps(clean, blocks)
    n_missing = sum(end - start for start, end in gaps)
    pct = lambda part, whole: 100.0 * part / whole if whole else float("nan")  # noqa: E731

    lines = ["=== ball.py --report ({}) ===".format(VERSION)]
    for name in ARRAYS:
        lines.append("{:<12} shape {} dtype {}".format(name, out[name].shape, out[name].dtype))
    lines.append("frames total {}  scored {}  blocks {}".format(n_frames, n_scored, len(blocks)))
    lines.append("raw observed: all frames {:.1f}%  scored {:.1f}%".format(
        pct(raw_obs.sum(), n_frames), pct((raw_obs & scored).sum(), n_scored)))
    lines.append("outliers removed: spikes {}  beyond {:.0f} m margin {}".format(
        n_spikes, OFFPITCH_MARGIN_M, n_far))
    off = outside_lines(ball_raw) & ~beyond_margin(ball_raw)
    lines.append("raw observed outside pitch lines: within margin {}  beyond margin {}".format(
        int(off.sum()), n_far))
    lines.append("gaps in scored blocks (after outlier removal): {}  missing frames {} "
                 "({:.1f}% of scored)".format(len(gaps), n_missing, pct(n_missing, n_scored)))
    lines.append("  {:<8} {:>7} {:>10}".format("bucket", "n_gaps", "%missing"))
    for label, count, frames in gap_table(gaps):
        lines.append("  {:<8} {:>7} {:>9.1f}%".format(label, count, pct(frames, n_missing)))
    lines.append("scored frames by ball_method:")
    method = out["ball_method"][scored]
    for code in (OBSERVED, LINEAR, HOLD, PLAYERS, UNSCORED):
        lines.append("  {:<9} {:>6.2f}%".format(METHOD_NAMES[code],
                                                 pct((method == code).sum(), n_scored)))
    conf = out["ball_conf"][scored]
    q = np.percentile(conf, [10, 25, 50, 75, 90]) if n_scored else [np.nan] * 5
    lines.append("ball_conf on scored frames: p10 {:.2f} p25 {:.2f} p50 {:.2f} p75 {:.2f} "
                 "p90 {:.2f}".format(*q))
    lines.append("ball_conf range all frames: min {:.3f} max {:.3f}  NaN {}".format(
        float(out["ball_conf"].min()), float(out["ball_conf"].max()),
        int(np.isnan(out["ball_conf"]).sum())))
    n_nan = int(np.isnan(out["ball_xy"][scored]).any(axis=1).sum())
    lines.append("NaN in ball_xy on scored frames: {}  ({})".format(
        n_nan, "OK" if n_nan == 0 else "FAIL"))
    lines.append("=== end ===")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print the aggregate gap report")
    parser.add_argument("--refresh", action="store_true", help="rebuild the cache")
    args = parser.parse_args()

    out = build(refresh=args.refresh)
    if args.report:
        print("\n".join(report_lines(load_data.load_tracking(), out)))
    else:
        for name in ARRAYS:
            print("{:<12} {} {}".format(name, out[name].shape, out[name].dtype))
