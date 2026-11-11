"""Section 2, Path B: Smart Tagging turned into per-frame hints.

Tags are time windows; later sections work frame by frame. This module writes two arrays of one
value per tactical video frame. Both are hints, not labels: the tags are automatic, they have false
positives and misses, and their windows (about 5 s) open before the event they describe.

    import tagging
    hints = tagging.load()      # dict: restart (n,) float32, st_team (n,) int8

v0:

  restart  1.0 over [start_frame, end_frame) of every "set pieces" tag (all six types), 0
           elsewhere, max where windows overlap. The window opens early, so do not read its first
           frame as the restart. v1 replaces the box with a soft bump that peaks near the restart.
  st_team  0 home, 1 away, -1 unknown. Every phase tag (every code except set pieces and goals)
           writes its team over its window. A frame covered by tags of both teams is -1. Tags
           with team "n/a", and so every throw-in, leave frames at -1.

Tag team strings are mapped to home and away by fuzzy matching against the club names in the ATD
metadata, so no club name is written in this file and "atletic club" still finds its club.

Known v0 caveat: st_team takes the tag's team at face value. For the defending codes (for example
"defending in middle third") the tagged team is probably the one WITHOUT the ball. v1 checks the
polarity of every code against the tracking. Until then, treat st_team as "team named by the tag",
not "team in possession". The report prints how much of st_team comes from defending codes only.

    python starter/tagging.py            # build or load the cache, print shapes
    python starter/tagging.py --report   # aggregate tag report and time-base check, safe to share
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, require, source_frames  # noqa: E402
from baseline import team_names  # noqa: E402
import ball  # noqa: E402
import load_data  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
VERSION = "tagging-v0"

SET_PIECE_CODE = "set pieces"
NON_PHASE_CODES = (SET_PIECE_CODE, "goals")
# Codes whose tagged team is probably the defending side. Used only for a report line in v0.
DEFENDING_CODES = ("defending in middle third", "defending in defensive third",
                   "defending in attacking third", "defensive transition")

NO_TEAM = ("n/a", "na", "none", "")
# Words that say "this is a club" rather than which club, dropped before matching names.
GENERIC_TOKENS = ("fc", "cf", "club", "cd", "sd", "ud", "ac", "sc", "afc", "de", "the")
TEAM_MATCH_MIN = 0.60      # best similarity needed to accept a match
TEAM_MATCH_MARGIN = 0.15   # and by how much it must beat the other club

HOME, AWAY, UNKNOWN = 0, 1, -1

CACHE_DIR = load_data.CACHE_DIR
SIGNATURE_FILE = "tagging_signature.json"
ARRAYS = ("restart", "st_team")

PARAMS = {
    "SET_PIECE_CODE": SET_PIECE_CODE, "NON_PHASE_CODES": list(NON_PHASE_CODES),
    "NO_TEAM": list(NO_TEAM), "GENERIC_TOKENS": list(GENERIC_TOKENS),
    "TEAM_MATCH_MIN": TEAM_MATCH_MIN, "TEAM_MATCH_MARGIN": TEAM_MATCH_MARGIN,
}


# ---------------------------------------------------------------------------------------------
# Team mapping
# ---------------------------------------------------------------------------------------------

def _tokens(name):
    words = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower()).split()
    core = [w for w in words if w not in GENERIC_TOKENS]
    return core or words


def name_similarity(a, b):
    """0..1. The better of whole-name similarity and the mean best-token similarity.

    The token score lets "barcelona" match "fc barcelona" and "atletic" match "athletic bilbao"
    without either name being written down anywhere.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    whole = difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    per_token = np.mean([max(difflib.SequenceMatcher(None, x, y).ratio() for y in tb)
                         for x in ta])
    return float(max(whole, per_token))


def map_teams(tag_team_values, club_names):
    """{tag team string: (code, best score)} for (home, away) club names.

    "n/a", None and empty give -1. A string that matches neither club well, or both about
    equally, gives -1 too. Two different strings that land on the same side raise, because then
    the tags would have only one team and every hint would be wrong.
    """
    out = {}
    for value in sorted({v for v in tag_team_values if v is not None}):
        text = str(value).strip().lower()
        if text in NO_TEAM:
            out[value] = (UNKNOWN, 0.0)
            continue
        scores = [name_similarity(text, club) for club in club_names]
        best = int(np.argmax(scores))
        ordered = sorted(scores, reverse=True)
        ok = ordered[0] >= TEAM_MATCH_MIN and ordered[0] - ordered[1] >= TEAM_MATCH_MARGIN
        out[value] = (best if ok else UNKNOWN, round(ordered[0], 3))

    sides = {}
    for value, (code, _) in out.items():
        if code != UNKNOWN:
            sides.setdefault(code, []).append(value)
    clash = {code: names for code, names in sides.items() if len(names) > 1}
    if clash:
        raise ValueError("tag team strings {} all map to the same side ({}) against clubs {}. "
                         "Check the ATD metadata and the tag export.".format(
                             sorted(next(iter(clash.values()))),
                             "home" if next(iter(clash)) == HOME else "away", list(club_names)))
    for value in tag_team_values:
        if value is None:
            out[None] = (UNKNOWN, 0.0)
    return out


def team_codes(tag_team_values, meta_path):
    """{tag team string: 0 home, 1 away, -1 unknown}, matched against the ATD club names."""
    clubs = team_names(meta_path)
    if len(clubs) != 2:
        raise ValueError("{}: expected 2 team names, found {}".format(meta_path, len(clubs)))
    return {value: code for value, (code, _) in map_teams(tag_team_values, clubs).items()}


# ---------------------------------------------------------------------------------------------
# Per-frame arrays
# ---------------------------------------------------------------------------------------------

def _clipped(tags, n_frames):
    """(start, end) arrays clipped to [0, n_frames); rows that end up empty have start == end."""
    start = np.clip(tags["start_frame"].to_numpy(dtype=np.int64), 0, n_frames)
    end = np.clip(tags["end_frame"].to_numpy(dtype=np.int64), 0, n_frames)
    return start, np.maximum(start, end)


def _coverage(start, end, n_frames):
    """How many of the half-open windows cover each frame. Vectorised with a difference array."""
    diff = np.zeros(n_frames + 1, dtype=np.int64)
    np.add.at(diff, start, 1)
    np.add.at(diff, end, -1)
    return np.cumsum(diff[:-1])


def build_restart(tags, n_frames):
    """float32 (n_frames,): 1.0 inside any set-piece window, 0 elsewhere. v0 box."""
    sel = (tags["code"] == SET_PIECE_CODE).to_numpy()
    start, end = _clipped(tags[sel], n_frames)
    return (_coverage(start, end, n_frames) > 0).astype(np.float32)


def team_votes(tags, n_frames, home_away, codes=None):
    """(home_count, away_count) per frame from phase tags, optionally only from `codes`."""
    code = tags["code"]
    sel = ~code.isin(NON_PHASE_CODES)
    if codes is not None:
        sel &= code.isin(codes)
    phase = tags[sel.to_numpy()]
    team = np.array([home_away.get(v, UNKNOWN) for v in phase["team"]], dtype=np.int64)
    start, end = _clipped(phase, n_frames)
    home = _coverage(start[team == HOME], end[team == HOME], n_frames)
    away = _coverage(start[team == AWAY], end[team == AWAY], n_frames)
    return home, away


def build_st_team(tags, n_frames, home_away):
    """int8 (n_frames,): team named by the phase tags, -1 where none or where they disagree."""
    home, away = team_votes(tags, n_frames, home_away)
    st_team = np.full(n_frames, UNKNOWN, dtype=np.int8)
    st_team[(home > 0) & (away == 0)] = HOME
    st_team[(away > 0) & (home == 0)] = AWAY
    return st_team


# ---------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------

def params_hash():
    text = json.dumps(PARAMS, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def source_signature(game=GAME, data_dir=None):
    """The tag file and the ATD metadata (club names come from it), by size and mtime."""
    paths = require(["smart_tagging", "atd_meta"], game, data_dir)
    return {"game": game,
            "files": [[os.path.basename(p), os.path.getsize(p), os.path.getmtime(p)]
                      for p in paths]}


def cache_signature(n_frames=None):
    return {"tags": source_signature(), "version": VERSION, "ball_version": ball.VERSION,
            "params": params_hash(), "n_frames": n_frames or source_frames("tactical")}


def _normalise(signature):
    return json.loads(json.dumps(signature, sort_keys=True))


def _cache_is_current(signature, cache_dir):
    path = os.path.join(cache_dir, SIGNATURE_FILE)
    if not os.path.isfile(path):
        return False
    if not all(os.path.isfile(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS):
        return False
    with open(path) as handle:
        return json.load(handle) == _normalise(signature)


def build(refresh=False, cache_dir=CACHE_DIR, tags=None, home_away=None, n_frames=None,
          signature=None):
    """Compute restart and st_team and cache them, or return the cache if it is current.

    With no arguments this reads the match. Tests pass tags, home_away, n_frames, cache_dir and
    signature, so they never touch the match data.
    """
    n_frames = n_frames or source_frames("tactical")
    if signature is None:
        signature = cache_signature(n_frames)
    if not refresh and _cache_is_current(signature, cache_dir):
        return {name: np.load(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS}

    if tags is None:
        tags = load_data.load_tags()
    if home_away is None:
        home_away = team_codes(tags["team"].tolist(), require(["atd_meta"])[0])
    out = {"restart": build_restart(tags, n_frames),
           "st_team": build_st_team(tags, n_frames, home_away)}
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    for name in ARRAYS:
        np.save(os.path.join(cache_dir, name + ".npy"), out[name])
    # Written last, so an interrupted build never looks complete.
    with open(os.path.join(cache_dir, SIGNATURE_FILE), "w") as handle:
        json.dump(_normalise(signature), handle, sort_keys=True)
    return out


def load(refresh=False):
    """The one line for everyone: dict(restart, st_team), rebuilt if stale."""
    return build(refresh=refresh)


# ---------------------------------------------------------------------------------------------
# Report. Aggregates only: counts, percentiles, shares. Nothing per frame.
# ---------------------------------------------------------------------------------------------

def report_lines(tags, out, mapping, blocks, time_base=None):
    """The --report block as a list of lines. mapping is map_teams() output."""
    n_frames = len(out["restart"])
    scored = np.zeros(n_frames, dtype=bool)
    for lo, hi in blocks:
        scored[lo:hi] = True
    n_scored = int(scored.sum())
    pct = lambda part, whole: 100.0 * part / whole if whole else float("nan")  # noqa: E731
    label = lambda v: "None" if v is None else v  # noqa: E731

    lines = ["=== tagging.py --report ({}) ===".format(VERSION)]
    for name in ARRAYS:
        lines.append("{:<8} shape {} dtype {}".format(name, out[name].shape, out[name].dtype))
    lines.append("tags {}".format(len(tags)))
    lines.append("team mapping (tag string -> side, similarity):")
    for value, (code, score) in sorted(mapping.items(), key=lambda kv: str(kv[0])):
        side = {HOME: "home", AWAY: "away"}.get(code, "unknown")
        lines.append("  {:<16} -> {:<7} {:.2f}".format(label(value), side, score))

    lines.append("by code:")
    for code, n in tags["code"].value_counts().sort_index().items():
        lines.append("  {:<32} {:>4}".format(code, n))
    lines.append("by team:")
    for team, n in tags["team"].fillna("None").value_counts().sort_index().items():
        lines.append("  {:<16} {:>4}".format(team, n))
    sp = tags[tags["code"] == SET_PIECE_CODE]
    lines.append("set pieces by type (team n/a count):")
    for kind, group in sp.groupby(sp["type"].fillna("None")):
        lines.append("  {:<24} {:>3}  n/a {:>3}".format(
            kind, len(group), int(group["team"].isin(NO_TEAM).sum())))

    lines.append("window length (s) by code: n  p10  p50  p90")
    length = (tags["end_s"] - tags["start_s"]).to_numpy(dtype=np.float64)
    for code in sorted(tags["code"].unique()):
        values = length[(tags["code"] == code).to_numpy()]
        q = np.percentile(values, [10, 50, 90])
        lines.append("  {:<32} {:>4} {:>5.1f} {:>5.1f} {:>5.1f}".format(code, len(values), *q))

    home_away = {v: c for v, (c, _) in mapping.items()}
    st = out["st_team"][scored]
    lines.append("st_team on scored frames: home {:.1f}%  away {:.1f}%  unknown {:.1f}%".format(
        pct((st == HOME).sum(), n_scored), pct((st == AWAY).sum(), n_scored),
        pct((st == UNKNOWN).sum(), n_scored)))
    home, away = team_votes(tags, n_frames, home_away)
    clash = (home > 0) & (away > 0)
    lines.append("disagreement frames (both teams tagged): {} all, {} scored".format(
        int(clash.sum()), int((clash & scored).sum())))
    dh, da = team_votes(tags, n_frames, home_away, codes=DEFENDING_CODES)
    from_defending = ((dh + da) > 0) & ((home - dh + away - da) == 0) & (out["st_team"] >= 0)
    lines.append("st_team set only by defending codes: {:.1f}% of scored frames "
                 "(polarity unchecked in v0)".format(pct((from_defending & scored).sum(),
                                                         n_scored)))
    lines.append("restart > 0 on scored frames: {:.2f}%".format(
        pct((out["restart"][scored] > 0).sum(), n_scored)))
    if time_base is not None:
        lines.append("time base check: {} kick off (start) tags, {}".format(
            time_base["n_kickoff_tags"], "OK" if time_base["agrees"] else "MISMATCH"))
    lines.append("=== end ===")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print the aggregate tag report")
    parser.add_argument("--refresh", action="store_true", help="rebuild the cache")
    args = parser.parse_args()

    out = build(refresh=args.refresh)
    if args.report:
        tags = load_data.load_tags()
        mapping = map_teams(tags["team"].tolist(), team_names(require(["atd_meta"])[0]))
        time_base = load_smart_tagging.check_time_base(load_smart_tagging.load())
        print("\n".join(report_lines(tags, out, mapping, ball.scored_blocks(), time_base)))
    else:
        for name in ARRAYS:
            print("{:<8} {} {}".format(name, out[name].shape, out[name].dtype))
