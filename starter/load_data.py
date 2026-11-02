"""The shared data layer: tracking as per-frame arrays, Smart Tagging as one tidy table.

Every module of our solution starts here, so that one parser decides what the data means:

    import load_data
    tracking = load_data.load_tracking()    # dict of arrays, indexed by tactical video frame
    tags = load_data.load_tags()            # pandas table, one row per Smart Tagging instance

Every per-frame array has one row per video frame, so array[f] is row f of the submission and
nobody has to remember that the ATD counts from 1. Positions are in metres, because the pitch is
not square and distances in normalised coordinates quietly favour one axis. Missing observations
are NaN, never a sentinel, so they cannot be mistaken for a place on the pitch.

Parsing the raw ATD is quick but not instant, so the arrays are cached in cache/. That folder is
gitignored for a reason beyond tidiness: it is frame-level tracking, and the terms of use forbid
publishing it. The cache records the size and modification time of its source files, so a changed
bundle rebuilds it rather than serving stale arrays.

    python starter/load_data.py         # build the cache and print a summary
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from load_smart_tagging import load as load_raw_tags  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
ARRAYS = ("players", "speeds", "ball_raw", "team_of_track")
SIGNATURE_FILE = "tracking_signature.json"

# Raw ATD line: "<frame>:<track>;<track>;...:<ball>", each group "x,y,speed".
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","

# float32 is ample: it resolves about 10 micrometres at 105 m, and it halves the size of the cache.
DTYPE = np.float32

# Tag column -> group name in the XML. Looked up case-insensitively, because the export's casing
# is a detail of the file rather than something the rest of the code should depend on.
TAG_GROUPS = {
    "team": "team",
    "half": "half",
    "type": "type",
    "side": "side",
    "direction": "direction of ball entry",
    "max_players_box": "max players in the box",
}
TAG_COLUMNS = (("code",) + tuple(TAG_GROUPS)
               + ("start_s", "end_s", "start_frame", "end_frame"))


def parse_tracking(game=GAME, data_dir=None):
    """Read the tactical ATD into arrays. One pass over the raw file, no cache.

    Unlike starter/baseline.py this refuses a malformed line instead of skipping it: a skipped
    line here would be a silent row of NaN in every section downstream.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team_of_track = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", game)
    n_tracks = len(team_of_track)
    length, width = GAMES[game]["pitch"]

    players = np.full((n_frames, n_tracks, 2), np.nan, dtype=DTYPE)
    speeds = np.full((n_frames, n_tracks), np.nan, dtype=DTYPE)
    ball_raw = np.full((n_frames, 2), np.nan, dtype=DTYPE)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError("{} line {}: expected 3 fields".format(raw_path, number))
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_tracks:
                raise ValueError("{} line {}: expected {} tracks, found {}".format(
                    raw_path, number, n_tracks, len(groups)))
            if not 0 <= frame < n_frames or seen[frame]:
                raise ValueError("{} line {}: bad or repeated frame id".format(raw_path, number))
            seen[frame] = True

            for slot, group in enumerate(groups):
                values = group.split(VALUE_SEP)
                players[frame, slot] = float(values[0]) * length, float(values[1]) * width
                speeds[frame, slot] = float(values[2])
            values = parts[2].split(VALUE_SEP)
            ball_raw[frame] = float(values[0]) * length, float(values[1]) * width

    if not seen.all():
        raise ValueError("{}: {} frames have no record".format(raw_path, int((~seen).sum())))
    return {"players": players, "speeds": speeds, "ball_raw": ball_raw,
            "team_of_track": team_of_track}


def source_signature(game=GAME, data_dir=None):
    """What the cache was built from. Cheap, because hashing 195 MB on every load is not."""
    paths = require(["atd_meta", "atd_raw"], game, data_dir)
    return {"game": game,
            "files": [[os.path.basename(path), os.path.getsize(path), os.path.getmtime(path)]
                      for path in paths]}


def cache_is_current(signature, cache_dir):
    """True only if every array exists and the signature matches."""
    path = os.path.join(cache_dir, SIGNATURE_FILE)
    if not os.path.isfile(path):
        return False
    if not all(os.path.isfile(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS):
        return False
    with open(path) as handle:
        return json.load(handle) == signature


def load_tracking(game=GAME, data_dir=None, cache_dir=CACHE_DIR, refresh=False):
    """The tracking arrays, from cache where possible. See the module docstring.

        players        (n_frames, n_tracks, 2)   x, y in metres, NaN where the slot is empty
        speeds         (n_frames, n_tracks)      m/s, NaN where empty, and on about 9.8% of
                                                 tracked positions: test players for presence,
                                                 not speeds
        ball_raw       (n_frames, 2)             x, y in metres, NaN where the ball is missing
        team_of_track  (n_tracks,)               0 = home, 1 = away
    """
    signature = source_signature(game, data_dir)
    if not refresh and cache_is_current(signature, cache_dir):
        return {name: np.load(os.path.join(cache_dir, name + ".npy")) for name in ARRAYS}

    tracking = parse_tracking(game, data_dir)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    for name in ARRAYS:
        np.save(os.path.join(cache_dir, name + ".npy"), tracking[name])
    # Written last, so an interrupted build never looks complete.
    with open(os.path.join(cache_dir, SIGNATURE_FILE), "w") as handle:
        json.dump(signature, handle)
    return tracking


def clean(value):
    """Lower-cased and stripped, so nobody downstream gets an empty match from a casing change."""
    return value.strip().lower() if isinstance(value, str) else None


def load_tags(game=GAME, data_dir=None):
    """Smart Tagging as a table, one row per instance, in file order.

    All text is lower case: code "set pieces", team "barcelona", "atletic club" or "n/a". A
    column is None where the instance has no such tag. "n/a" is kept as it is, because deciding
    what a tag without a team means is a modelling choice and belongs to starter/tagging.py.

    end_frame is the first frame after the interval, so tag frames are start_frame:end_frame.
    Intervals are windows around an event rather than the instant of it -- see DATA.md.
    """
    rows = []
    for event in load_raw_tags(game, data_dir):
        tags = {clean(group): value for group, value in event["tags"].items()}
        row = {column: clean(tags.get(group)) for column, group in TAG_GROUPS.items()}
        row.update(code=clean(event["code"]), start_s=event["start_s"], end_s=event["end_s"],
                   start_frame=event["start_frame"], end_frame=event["end_frame"])
        rows.append(row)
    return pd.DataFrame(rows, columns=list(TAG_COLUMNS))


if __name__ == '__main__':
    tracking = load_tracking()
    for name in ARRAYS:
        print("{:<14} {}".format(name, tracking[name].shape))
    print("ball located on {:.1f}% of frames".format(
        100.0 * float(np.mean(~np.isnan(tracking["ball_raw"][:, 0])))))
    tags = load_tags()
    print("{} tags, {} set pieces".format(len(tags), int((tags["code"] == "set pieces").sum())))
