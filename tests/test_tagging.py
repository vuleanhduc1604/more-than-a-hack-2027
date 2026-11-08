"""tagging.py on synthetic tags only. Runs on a fresh clone with no data/ folder."""

import hashlib
import os

import numpy as np
import pytest

import synthetic
import tagging

N = synthetic.N_FRAMES
CLUBS = ("FC Barcelona", "Athletic Club")
SIGNATURE = {"source": "synthetic-tags"}


@pytest.fixture(scope="module")
def tags():
    return synthetic.make_tags(N)


@pytest.fixture(scope="module")
def home_away(tags):
    return {v: c for v, (c, _) in tagging.map_teams(tags["team"].tolist(), CLUBS).items()}


def test_fuzzy_team_mapping(home_away):
    assert home_away["barcelona"] == tagging.HOME
    assert home_away["atletic club"] == tagging.AWAY
    assert home_away["n/a"] == tagging.UNKNOWN


def test_mapping_handles_other_club_spellings():
    mapping = tagging.map_teams(["barcelona", "atletic club", None, "someone else"],
                                ("Barcelona", "Athletic Club de Bilbao"))
    assert mapping["barcelona"][0] == tagging.HOME
    assert mapping["atletic club"][0] == tagging.AWAY
    assert mapping[None][0] == tagging.UNKNOWN
    assert mapping["someone else"][0] == tagging.UNKNOWN


def test_two_strings_on_one_side_raise():
    with pytest.raises(ValueError):
        tagging.map_teams(["barcelona", "barca"], CLUBS)


def test_team_codes_reads_atd_metadata(tmp_path):
    meta = tmp_path / "meta.xml"
    meta.write_text("<Root><Teams>"
                    "<Team id='T1'><Name>{}</Name></Team>"
                    "<Team id='T2'><Name>{}</Name></Team>"
                    "</Teams></Root>".format(*CLUBS))
    codes = tagging.team_codes(["barcelona", "atletic club", "n/a"], str(meta))
    assert codes == {"barcelona": 0, "atletic club": 1, "n/a": -1}


def test_restart_box_half_open_and_clipped(tags):
    restart = tagging.build_restart(tags, N)
    assert restart.shape == (N,) and restart.dtype == np.float32
    assert set(np.unique(restart)) <= {0.0, 1.0}
    assert restart[479] == 0 and restart[480] == 1      # kick off window opens
    assert restart[724] == 1 and restart[725] == 0      # throw-in window, end is exclusive
    assert restart[1124] == 1 and restart[1125] == 0    # corner
    assert restart[N - 30:].all()                       # goal kick runs past the end, clipped
    assert restart[1200:1450].sum() == 0                # phase tags do not raise restart


def test_st_team(tags, home_away):
    st = tagging.build_st_team(tags, N, home_away)
    assert st.shape == (N,) and st.dtype == np.int8
    assert np.all(st[1200:1400] == tagging.HOME)        # build up, with an agreeing overlap
    assert np.all(st[1400:1450] == tagging.UNKNOWN)     # conflict with progression
    assert np.all(st[1450:1600] == tagging.AWAY)        # progression alone
    assert st[1199] == tagging.UNKNOWN and st[1600] == tagging.UNKNOWN  # half-open edges
    assert np.all(st[600:725] == tagging.UNKNOWN)       # throw-in stays unknown
    assert np.all(st[1000:1125] == tagging.UNKNOWN)     # set pieces do not write st_team in v0
    assert np.all(st[2000:2200] == tagging.AWAY)        # goals tag ignored, defending tag kept
    assert np.all(st[3000:3100] == tagging.UNKNOWN)     # phase tag with team n/a
    assert np.all(st[N - 50:] == tagging.AWAY)          # long ball clipped at the end


def test_negative_start_is_clipped(tags, home_away):
    early = tags.iloc[[3]].copy()
    early["start_frame"], early["end_frame"] = -40, 10
    st = tagging.build_st_team(early, N, home_away)
    assert np.all(st[:10] == tagging.HOME) and st[10] == tagging.UNKNOWN


def _digest(folder):
    return {name: hashlib.sha256(open(os.path.join(folder, name + ".npy"), "rb").read())
            .hexdigest() for name in tagging.ARRAYS}


def test_build_is_deterministic_and_cached(tmp_path, tags, home_away):
    first, second = str(tmp_path / "a"), str(tmp_path / "b")
    for folder in (first, second):
        tagging.build(cache_dir=folder, tags=tags, home_away=home_away, n_frames=N,
                      signature=SIGNATURE)
    assert _digest(first) == _digest(second)
    cached = tagging.build(cache_dir=first, tags=None, home_away=None, n_frames=N,
                           signature=SIGNATURE)
    assert cached["st_team"].dtype == np.int8 and cached["restart"].shape == (N,)


def test_report_block(tags, home_away):
    mapping = tagging.map_teams(tags["team"].tolist(), CLUBS)
    out = {"restart": tagging.build_restart(tags, N),
           "st_team": tagging.build_st_team(tags, N, home_away)}
    lines = tagging.report_lines(tags, out, mapping, synthetic.BLOCKS,
                                 {"n_kickoff_tags": 1, "agrees": True})
    assert lines[0].startswith("=== tagging.py --report") and lines[-1] == "=== end ==="
    assert any(line.endswith("OK") for line in lines if "time base" in line)
