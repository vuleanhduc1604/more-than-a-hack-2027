"""Run the whole pipeline and write the possession CSV.

    python starter/make_submission.py                 # writes evaluation/results/section4_v2.csv
    python starter/make_submission.py --name mine

Pipeline, each step cached in cache/ and rebuilt when its inputs change:

    load_data   tracking arrays                          Section 1
    ball        ball_xy, ball_conf                       Section 2
    dead_ball   p_dead                                   Section 3
    team        p_home (ball evidence + team shape)      Section 4
    decode      Viterbi over H / A / D inside each half  Section 4

Everything is deterministic: no random numbers are drawn after the sample games are degraded with
fixed seeds, so two runs write byte-identical files (the file's SHA-256 is printed to compare).
The result is checked with the same validation the official scorer applies.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, GAMES  # noqa: E402
import ball  # noqa: E402
import dead_ball  # noqa: E402
import decode  # noqa: E402
import team  # noqa: E402
import possession_metrics as pm  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
DEFAULT_NAME = "section4_v2"


def make_labels():
    """Letters for every video frame of the match: H, A, D inside the halves, X outside."""
    p_dead = dead_ball.load()
    p_home = team.load()
    codes = decode.decode(p_home, p_dead, ball.scored_blocks())
    return decode.to_letters(codes)


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME, help="file name, without .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    labels = make_labels()
    path = write_csv(os.path.join(args.out, args.name + ".csv"), labels)

    frame = pm.read_label_csv(path)
    pm.validate_submission(frame, GAMES[GAME]["n_video_frames"])   # raises if the scorer would
    codes = pm.label_codes(frame)
    n_spells = len(pm.possession_spells(codes, pm.valid_runs(codes)))
    print("wrote {} rows to {}".format(len(labels), os.path.relpath(path, ROOT)))
    print("  " + "  ".join("{} {}".format(k, int(np.sum(labels == k))) for k in "HADX"))
    print("  {} possessions".format(n_spells))
    print("  sha256 {}".format(sha256(path)))
    print("\nvalidated; score it with:  python evaluation/run_evaluation.py")
