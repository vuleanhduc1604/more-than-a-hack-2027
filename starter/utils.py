"""Plotting helpers for the starter notebooks.

Nothing here is needed to produce a submission -- baseline.py writes one with numpy alone. This is
for looking at the data, which is the fastest way to understand what the tracking is and is not
telling you.

    pip install -r requirements-starter.txt

matplotlib is imported inside the functions rather than at module scope, so importing this module
costs nothing in an environment that only wants the loaders.
"""

import numpy as np

# Params
# Metres. The ATD metadata says 105 x 68; games.py carries 105.3 x 68.0 measured from the
# corrected tracking. The difference matters only if you are computing distances in metres.
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = "#d62728", "#1f77b4", "#ffffff"
GRASS, LINE = "#3f8f4f", "#ffffff"


def draw_pitch(ax=None, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """A plain football pitch in metres, origin at a corner. Returns the axes.

    Drawn here with matplotlib rather than pulled from a pitch-plotting package, because it is
    thirty lines and it means the notebooks need one less dependency to install and one less
    unmaintained package to age badly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))

    ax.add_patch(Rectangle((0, 0), length, width, facecolor=GRASS, edgecolor=LINE, zorder=0))
    ax.plot([length / 2, length / 2], [0, width], color=LINE, zorder=1)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, fill=False, edgecolor=LINE, zorder=1))

    for side in (0, 1):
        # Penalty area 16.5 m deep and 40.3 m wide, six-yard box 5.5 m by 18.3 m.
        for depth, box_width in ((16.5, 40.3), (5.5, 18.3)):
            x = 0 if side == 0 else length - depth
            ax.add_patch(Rectangle((x, (width - box_width) / 2), depth, box_width,
                                   fill=False, edgecolor=LINE, zorder=1))

    ax.set_xlim(-3, length + 3)
    ax.set_ylim(-3, width + 3)
    ax.set_aspect("equal")
    ax.set_xlabel("metres")
    return ax


def plot_frame(dataset, frame_index, ax=None, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Draw one tracking frame: both teams and the ball.

    The ball is drawn last and outlined so it stays visible in a crowd -- and when it is missing,
    which happens on roughly four frames in ten, the title says so rather than leaving you to
    wonder why there is no white dot.
    """
    ax = draw_pitch(ax, length, width)
    frame = dataset.frames[frame_index]
    home_team = dataset.metadata.teams[0]

    for player, position in frame.players_coordinates.items():
        if position is None:
            continue
        ax.scatter(position.x * length, position.y * width, s=90, zorder=3,
                   color=HOME_COLOUR if player.team == home_team else AWAY_COLOUR,
                   edgecolors="black", linewidths=0.5)

    ball = frame.ball_coordinates
    has_ball = ball is not None and ball.x is not None and not np.isnan(ball.x)
    if has_ball:
        ax.scatter(ball.x * length, ball.y * width, s=70, zorder=4,
                   color=BALL_COLOUR, edgecolors="black", linewidths=1.2)

    ax.set_title("frame {}{}".format(frame_index, "" if has_ball else "   -- ball not tracked"))
    return ax


def plot_frame_speeds(players, speeds, ball_raw, team_of_track, frame_index, ax=None,
                      ball_xy=None, ball_conf=None, fps=25, length=105.3, width=68.0,
                      step=2):
    """One frame from the metre arrays of load_data, with each player's and the ball's speed.

    Each dot is labelled with its speed in m/s and carries an arrow showing where it would be in
    one second at its current velocity. The direction comes from positions `step` frames either
    side, and the label from `speeds`. A tracked player with no speed in `speeds` (about 8% of
    graded positions) is labelled from the positions instead, with a trailing "*".

    The ball's speed is never in the data, so it is always derived from positions. When the ball
    is missing and ball_xy (Section 2's filled ball) is given, the filled estimate is drawn as a
    hollow circle with its ball_conf in the title, so a guessed position never passes for a seen one.
    """
    import matplotlib.pyplot as plt

    ax = draw_pitch(ax, length, width)
    n = len(players)
    lo, hi = max(0, frame_index - step), min(n - 1, frame_index + step)
    dt = (hi - lo) / float(fps)

    def velocity(track):
        """(n, 2) m/s from a central difference; NaN where either end is missing."""
        return (track[hi] - track[lo]) / dt if dt > 0 else np.full_like(track[frame_index], np.nan)

    pos = players[frame_index].astype(np.float64)
    vel = velocity(players.astype(np.float64))
    team = np.asarray(team_of_track)
    present = ~np.isnan(pos[:, 0])
    for side, colour in ((0, HOME_COLOUR), (1, AWAY_COLOUR)):
        sel = present & (team == side)
        ax.scatter(pos[sel, 0], pos[sel, 1], s=90, zorder=3, color=colour,
                   edgecolors="black", linewidths=0.5)
    for k in np.flatnonzero(present):
        v = vel[k]
        if not np.isnan(v[0]):
            ax.annotate("", xy=pos[k] + v, xytext=pos[k], zorder=2,
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
        tracked = speeds[frame_index, k]
        label = "{:.1f}".format(tracked) if not np.isnan(tracked) else (
            "{:.1f}*".format(np.hypot(*v)) if not np.isnan(v[0]) else "?")
        ax.text(pos[k, 0] + 0.8, pos[k, 1] + 0.8, label, fontsize=7, zorder=5, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0))

    b = ball_raw[frame_index].astype(np.float64)
    seen = not np.isnan(b[0])
    note = ""
    if seen:
        bv = velocity(ball_raw.astype(np.float64))
        ax.scatter(*b, s=70, zorder=6, color=BALL_COLOUR, edgecolors="black", linewidths=1.2)
        if not np.isnan(bv[0]):
            ax.annotate("", xy=b + bv, xytext=b, zorder=6,
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
            note = "   ball {:.1f} m/s".format(np.hypot(*bv))
        else:
            note = "   ball speed unknown (neighbours missing)"
    elif ball_xy is not None and not np.isnan(ball_xy[frame_index, 0]):
        est = ball_xy[frame_index].astype(np.float64)
        ev = velocity(ball_xy.astype(np.float64))
        ax.scatter(*est, s=90, zorder=6, facecolors="none", edgecolors="yellow", linewidths=1.8)
        if not np.isnan(ev[0]):
            ax.annotate("", xy=est + ev, xytext=est, zorder=6,
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2, ls="--"))
        conf = "" if ball_conf is None else ", conf {:.2f}".format(ball_conf[frame_index])
        note = "   ball not tracked: hollow = filled estimate ({:.1f} m/s{})".format(
            np.hypot(*ev) if not np.isnan(ev[0]) else float("nan"), conf)
    else:
        note = "   ball not tracked"

    ax.set_title("frame {}{}".format(frame_index, note), fontsize=10)
    ax.text(0, -2.6, "labels: speed m/s (* = from positions)   arrows: 1 s of travel",
            fontsize=7, color="black")
    ax.set_xlim(-4, length + 4)
    ax.set_ylim(-4, width + 4)
    return ax


def animate_frames(dataset, start_frame, end_frame, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """A matplotlib animation of a span of tracking frames.

    In a notebook:

        from IPython.display import HTML
        HTML(animate_frames(dataset, 11185, 11435).to_html5_video())

    Needs ffmpeg for to_html5_video(); to_jshtml() works without it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    figure, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_pitch(ax, length, width)
    home = ax.scatter([], [], s=90, color=HOME_COLOUR, edgecolors="black", linewidths=0.5, zorder=3)
    away = ax.scatter([], [], s=90, color=AWAY_COLOUR, edgecolors="black", linewidths=0.5, zorder=3)
    ball = ax.scatter([], [], s=70, color=BALL_COLOUR, edgecolors="black", linewidths=1.2, zorder=4)
    title = ax.set_title("")
    plt.close(figure)

    home_team = dataset.metadata.teams[0]

    def positions(frame):
        """(home_xy, away_xy, ball_xy) for one frame, each an (n, 2) array."""
        sides = {True: [], False: []}
        for player, position in frame.players_coordinates.items():
            if position is not None:
                sides[player.team == home_team].append(
                    (position.x * length, position.y * width))
        point = frame.ball_coordinates
        visible = point is not None and point.x is not None and not np.isnan(point.x)
        return (np.array(sides[True]).reshape(-1, 2), np.array(sides[False]).reshape(-1, 2),
                np.array([[point.x * length, point.y * width]]) if visible
                else np.empty((0, 2)))

    def update(step):
        index = start_frame + step
        home_xy, away_xy, ball_xy = positions(dataset.frames[index])
        home.set_offsets(home_xy)
        away.set_offsets(away_xy)
        # set_offsets on an empty array is how the ball simply disappears on frames where the
        # tracking lost it, which is the honest way to draw a missing observation.
        ball.set_offsets(ball_xy)
        title.set_text("frame {}{}".format(index, "" if len(ball_xy) else "   -- no ball"))
        return home, away, ball, title

    return FuncAnimation(figure, update, frames=end_frame - start_frame, interval=40, blit=False)
