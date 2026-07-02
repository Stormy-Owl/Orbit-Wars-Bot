"""
orbit_wars.py — Inference agent for Orbit Wars.

Architecture
────────────
  3 Minor NNUEs  →  candidate move generation + ordering
  Move filters   →  prune sun-crossing / impossible / interceptable moves
  Search         →  Alpha-Beta (2-player) | lightweight MCTS (3-4 player)
  Main NNUE      →  leaf evaluation

No GPU usage; designed to fit inside a 1-second per-turn budget.
"""

import math, os, json, pickle, time
import numpy as np

# ══════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════
SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
BOARD              = 100.0
R_CAP              = 2.0
TOP_K              = 5       # top target planets
TOP_M              = 3       # ship-count candidates per (origin, target) pair
TOP_L              = 3       # timing candidates per (origin, target) pair
MAX_N              = 300     # max intercept look-ahead steps
MAX_PLANETS        = 40
MAX_FLEETS         = 20
WEIGHTS_FILE       = "nnue_weights.pkl"

# Move-time budget (seconds) – leave headroom for Python overhead
MOVE_BUDGET        = 0.75
AB_DEPTH_2P        = 3      # Alpha-Beta depth for 2-player
MCTS_SIMS          = 250   # MCTS simulations for 4-player
MCTS_C             = 1.4    # UCB exploration constant

# Field indices
ID, OWNER, X, Y, RADIUS, SHIPS, PROD = 0, 1, 2, 3, 4, 5, 6
FL_ID, FL_OWNER, FL_X, FL_Y, FL_ANGLE, FL_FROM, FL_SHIPS = 0, 1, 2, 3, 4, 5, 6


# ══════════════════════════════════════════════
#  NNUE (pure-NumPy, no GPU)
# ══════════════════════════════════════════════
class NNUE:
    """3-layer ReLU network with optional accumulator cache."""

    def __init__(self, in_dim, l1, l2, out_dim=1, lr=0.005):
        s = 0.1
        self.lr   = lr
        self.W1   = np.random.randn(in_dim, l1).astype(np.float32) * s
        self.b1   = np.zeros(l1, dtype=np.float32)
        self.W2   = np.random.randn(l1, l2).astype(np.float32) * s
        self.b2   = np.zeros(l2, dtype=np.float32)
        self.W3   = np.random.randn(l2, out_dim).astype(np.float32) * s
        self.b3   = np.zeros(out_dim, dtype=np.float32)
        self._cache = None   # (x, z1, h1, z2, h2)

    # ── forward (single sample) ────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        out = h2 @ self.W3 + self.b3
        self._cache = (x, z1, h1, z2, h2)
        return out

    def forward_scalar(self, x: np.ndarray) -> float:
        return float(self.forward(x)[0])

    # ── batched forward (N×in_dim) → (N,) ─────
    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        h1  = np.maximum(0.0, X @ self.W1 + self.b1)
        h2  = np.maximum(0.0, h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3).ravel()

    # ── backward (single sample, uses cached intermediates) ──
    def backward(self, grad_out: np.ndarray):
        x, z1, h1, z2, h2 = self._cache
        dW3     = h2[:, None] * grad_out[None, :]
        grad_h2 = grad_out @ self.W3.T
        grad_z2 = grad_h2 * (z2 > 0)
        dW2     = h1[:, None] * grad_z2[None, :]
        grad_h1 = grad_z2 @ self.W2.T
        grad_z1 = grad_h1 * (z1 > 0)
        dW1     = x[:, None] * grad_z1[None, :]
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * grad_z1
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * grad_z2
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * grad_out

    # ── weight serialisation ───────────────────
    def save(self, d: dict, key: str):
        d[key] = (self.W1.copy(), self.b1.copy(),
                  self.W2.copy(), self.b2.copy(),
                  self.W3.copy(), self.b3.copy())

    def load(self, d: dict, key: str):
        if key in d:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = \
                [a.astype(np.float32) for a in d[key]]

    # ── snapshot / restore for search ─────────
    def snapshot(self):
        return self._cache  # read-only reference; callers must not mutate

    def restore(self, snap):
        self._cache = snap


# ── Global NNUE instances ──────────────────────
planet_nnue = NNUE(9,   16, 8,  1)
ship_nnue   = NNUE(12,  16, 8,  1)
step_nnue   = NNUE(10,  16, 8,  1)
main_nnue   = NNUE(340, 256, 32, 1, lr=0.001)


def load_weights(path: str = WEIGHTS_FILE):
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        d = pickle.load(f)
    planet_nnue.load(d, "planet")
    ship_nnue.load(d,   "ship")
    step_nnue.load(d,   "step")
    main_nnue.load(d,   "main")


def save_weights(path: str = WEIGHTS_FILE):
    d = {}
    planet_nnue.save(d, "planet")
    ship_nnue.save(d,   "ship")
    step_nnue.save(d,   "step")
    main_nnue.save(d,   "main")
    with open(path, "wb") as f:
        pickle.dump(d, f)


load_weights()


# ══════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════
def dist2(x1, y1, x2, y2) -> float:
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def dist(x1, y1, x2, y2) -> float:
    return math.sqrt(dist2(x1, y1, x2, y2))


def is_orbiting(p) -> bool:
    return (dist(p[X], p[Y], SUN_X, SUN_Y) + p[RADIUS]) < 50.0


def fleet_speed(ships: int) -> float:
    return 1.0 + 5.0 * math.pow(math.log(max(ships, 1)) / math.log(1000), 1.5)


def sun_angle_score(px, py) -> float:
    r = dist(px, py, SUN_X, SUN_Y)
    if r <= SUN_R:
        return 0.0
    blocked = 2.0 * math.asin(min(SUN_R / r, 1.0))
    return max(0.0, 1.0 - blocked / (math.pi / 2.0))


def position_score(px, py, peak=30.0) -> float:
    r     = dist(px, py, SUN_X, SUN_Y)
    sigma = 8.0 if r < peak else 14.0
    return math.exp(-((r - peak) ** 2) / (2.0 * sigma ** 2))


def find_intercept(ox, oy, tp, w, steps=80):
    """Return best (ix, iy, fleet_time) for an orbiting target."""
    r_orb   = dist(tp[X], tp[Y], SUN_X, SUN_Y)
    phi     = math.atan2(tp[Y] - SUN_Y, tp[X] - SUN_X)
    angles  = phi + np.linspace(0, 2.0 * math.pi, steps, endpoint=False)
    px      = SUN_X + r_orb * np.cos(angles)
    py      = SUN_Y + r_orb * np.sin(angles)
    ftimes  = np.sqrt((ox - px) ** 2 + (oy - py) ** 2) / fleet_speed(50)
    ptimes  = ((angles - phi) % (2.0 * math.pi)) / (w + 1e-9)
    best    = int(np.argmin(np.abs(ftimes - ptimes)))
    return float(px[best]), float(py[best]), float(ftimes[best])


def intercept_tol(tp, w) -> float:
    return R_CAP / (dist(tp[X], tp[Y], SUN_X, SUN_Y) * w + 1e-9)


# ══════════════════════════════════════════════
#  MOVE FILTERS
# ══════════════════════════════════════════════
_TWO_PI = 2.0 * math.pi


def _segment_sun_hit(x1, y1, x2, y2) -> bool:
    """Return True if segment (x1,y1)→(x2,y2) passes within SUN_R of sun."""
    dx = x2 - x1;  dy = y2 - y1
    fx = x1 - SUN_X; fy = y1 - SUN_Y
    a  = dx * dx + dy * dy
    if a < 1e-12:
        return (fx * fx + fy * fy) <= SUN_R * SUN_R
    b  = 2.0 * (fx * dx + fy * dy)
    c  = fx * fx + fy * fy - SUN_R * SUN_R
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sq   = math.sqrt(disc)
    t1   = (-b - sq) / (2.0 * a)
    t2   = (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def _out_of_bounds(x, y) -> bool:
    return x < 0.0 or x > BOARD or y < 0.0 or y > BOARD

def fleet_missed(op, tp, ships, angle,N):
    speed=1.0+5.0*(math.pow(math.log(ships)/math.log(1000),1.5))
    xf_N=op[X]+math.cos(angle)*(speed*N)
    yf_N=op[Y]+math.sin(angle)*(speed*N)
    dist_fc=math.hypot(xf_N-tp[X],yf_N-tp[Y])
    return dist_fc>tp[RADIUS]

def filter_move_orb(op, tp, ships, angle, N) -> bool:
    """
    Return True if the move should be KEPT (passes all filters).

    Rejected if:
      - ships > planet garrison (impossible)
      - launch angle sends fleet straight into sun
      - direct path to target crosses sun (static targets)
      - destination is already out-of-bounds
    """
    if ships <= 0 or ships > op[SHIPS]:
        return False
    # Fleet spawns just outside planet radius
    spawn_dist = op[RADIUS] + 0.5
    sx = op[X] + math.cos(angle) * spawn_dist
    sy = op[Y] + math.sin(angle) * spawn_dist
    if _out_of_bounds(sx, sy):
        return False
    # Check sun crossing to the target
    if fleet_missed(op, tp, ships, angle,N):
        return False
    # For orbiting target use intercept point
    ix = sx + math.cos(angle) * 60.0  # rough far point
    iy = sy + math.sin(angle) * 60.0
    if _segment_sun_hit(sx, sy, ix, iy):
        return False
    
    return True

def filter_move_sta(op, tp, ships, angle):
    """
    Return True if the move should be KEPT (passes all filters).

    Rejected if:
      - ships > planet garrison (impossible)
      - launch angle sends fleet straight into sun
      - direct path to target crosses sun (static targets)
      - destination is already out-of-bounds
    """
    if ships <= 0 or ships > op[SHIPS]:
        return False
    # Fleet spawns just outside planet radius
    spawn_dist = op[RADIUS] + 0.5
    sx = op[X] + math.cos(angle) * spawn_dist
    sy = op[Y] + math.sin(angle) * spawn_dist
    if _out_of_bounds(sx, sy):
        return False
    if _segment_sun_hit(sx, sy, tp[X], tp[Y]):
        return False
    
    return True

def _enemy_fleets_targeting(fleets, tp_id, our_player, fleet_cache):
    """Ships in enemy fleets heading toward tp_id."""
    total = 0
    for fl in fleets:
        if fl[FL_OWNER] == our_player:
            continue
        if fleet_cache.get(str(fl[FL_ID]), {}).get("target") == tp_id:
            total += fl[FL_SHIPS]
    return total


# ══════════════════════════════════════════════
#  FLEET CACHE  (in-memory only; flushed to disk in train.py)
# ══════════════════════════════════════════════
class FleetCache:
    """Tracks fleet targets; purely in-memory during inference."""

    def __init__(self):
        self._cache: dict = {}          # str(fid) → {target, owner, inferred}
        self.pending: dict = {}         # from_planet_id → target_planet_id
        self._events: list = []         # active fleet events
        self._done_events: list = []    # closed fleet events

    def reset(self):
        self._cache.clear()
        self.pending.clear()
        self._events.clear()
        self._done_events.clear()

    def load(self, d: dict):
        self._cache = d.get("cache", {})
        self._events = d.get("events", [])
        self._done_events = d.get("done_events", [])

    def dump(self) -> dict:
        return {
            "cache":       self._cache,
            "events":      self._events,
            "done_events": self._done_events,
        }

    def get_target(self, fid) -> int | None:
        return self._cache.get(str(fid), {}).get("target")

    def add(self, fid, target_id, owner, inferred: bool):
        self._cache[str(fid)] = {"target": target_id, "owner": owner,
                                  "inferred": inferred}

    def remove(self, fid):
        self._cache.pop(str(fid), None)

    def update(self, fleets, planets, player, w, step):
        current_ids = {str(fl[FL_ID]) for fl in fleets}
        planet_counts: dict = {}
        for p in planets:
            planet_counts[p[OWNER]] = planet_counts.get(p[OWNER], 0) + 1

        # Close events for departed fleets
        for fid in [k for k in self._cache if k not in current_ids]:
            self._close_event(int(fid), step, planet_counts, player)
            self.remove(fid)

        # Register new fleets
        for fl in fleets:
            fid = str(fl[FL_ID])
            if fid not in self._cache:
                if fl[FL_OWNER] == player:
                    tid = self.pending.pop(fl[FL_FROM], None)
                    if tid is not None:
                        self.add(fl[FL_ID], tid, player, False)
                        self._open_event(fl[FL_ID], step, player, tid)
                    else:
                        tid = infer_fleet_target(fl, planets, w)
                        self.add(fl[FL_ID], tid, player, True)
                        if tid is not None:
                            self._open_event(fl[FL_ID], step, player, tid)
                else:
                    tid = infer_fleet_target(fl, planets, w)
                    self.add(fl[FL_ID], tid, fl[FL_OWNER], True)

    def fleet_pressure(self, fleets, tp, player):
        e_sh = f_sh = e_wd = f_wd = 0.0
        for fl in fleets:
            if self.get_target(fl[FL_ID]) != tp[ID]:
                continue
            d = dist(fl[FL_X], fl[FL_Y], tp[X], tp[Y])
            if fl[FL_OWNER] == player:
                f_sh += fl[FL_SHIPS];  f_wd += fl[FL_SHIPS] * d
            else:
                e_sh += fl[FL_SHIPS];  e_wd += fl[FL_SHIPS] * d
        return e_sh, f_sh, e_wd, f_wd

    # ── event helpers ──────────────────────────
    def _open_event(self, fleet_id, t_launch, player, target_id):
        self._events.append({
            "fleet_id": fleet_id, "t_launch": t_launch,
            "t_end": None, "player": player,
            "target_id": target_id, "score": None,
        })

    def _close_event(self, fleet_id, t_end, planet_counts, player):
        for ev in list(self._events):
            if ev["fleet_id"] == fleet_id:
                our       = planet_counts.get(player, 0)
                max_other = max(
                    (v for k, v in planet_counts.items()
                     if k != player and k != -1), default=0)
                ev["t_end"] = t_end
                ev["score"] = 1.0 if our > max_other else -1.0
                self._done_events.append(ev)
                self._events.remove(ev)
                return

    def net_score_for_step(self, t) -> float | None:
        contribs = [
            (ev["score"], ev["t_end"] - t)
            for ev in self._done_events
            if ev["t_end"] is not None and ev["score"] is not None
            and ev["t_launch"] <= t <= ev["t_end"]
        ]
        if not contribs:
            return None
        total_w = sum(w for _, w in contribs) + 1e-9
        return sum(s * w for s, w in contribs) / total_w


# Singleton used by the agent
_fc = FleetCache()


# ══════════════════════════════════════════════
#  ENEMY FLEET INFERENCE
# ══════════════════════════════════════════════
def infer_fleet_target(fl, planets, w) -> int | None:
    ux = math.cos(fl[FL_ANGLE])
    uy = math.sin(fl[FL_ANGLE])
    best_pid, best_res = None, 1e9
    for p in planets:
        if is_orbiting(p):
            r_orb = dist(p[X], p[Y], SUN_X, SUN_Y)
            fx_s  = fl[FL_X] - SUN_X;  fy_s = fl[FL_Y] - SUN_Y
            b_    = 2.0 * (fx_s * ux + fy_s * uy)
            c_    = fx_s ** 2 + fy_s ** 2 - r_orb ** 2
            disc  = b_ ** 2 - 4.0 * c_
            if disc < 0:
                continue
            for sign in (1, -1):
                t_fl = (-b_ + sign * math.sqrt(disc)) / 2.0
                if t_fl < 0:
                    continue
                meet_a = math.atan2(fl[FL_Y] + uy * t_fl - SUN_Y,
                                    fl[FL_X] + ux * t_fl - SUN_X)
                phi    = math.atan2(p[Y] - SUN_Y, p[X] - SUN_X)
                t_pl   = ((meet_a - phi) % _TWO_PI) / (w + 1e-9)
                res    = abs(t_fl - t_pl)
                if res < intercept_tol(p, w) and res < best_res:
                    best_res = res;  best_pid = p[ID];  break
        else:
            angle_to = math.atan2(p[Y] - fl[FL_Y], p[X] - fl[FL_X])
            diff     = abs((fl[FL_ANGLE] - angle_to + math.pi) % _TWO_PI - math.pi)
            if diff < 0.15 and diff < best_res:
                best_res = diff;  best_pid = p[ID]
    return best_pid


# ══════════════════════════════════════════════
#  FEATURE BUILDERS  (Minor NNUEs)
# ══════════════════════════════════════════════
def _planet_feat(p, all_planets, player, nb_r=20.0) -> np.ndarray:
    n_fr = n_en = n_neu = 0
    d_fr = d_en = 1e9
    for q in all_planets:
        if q[ID] == p[ID]:
            continue
        d = dist(p[X], p[Y], q[X], q[Y])
        if d < nb_r:
            if   q[OWNER] == player:  n_fr += 1
            elif q[OWNER] == -1:      n_neu += 1
            else:                      n_en += 1
        if q[OWNER] == player and d < d_fr:             d_fr = d
        if q[OWNER] not in (-1, player) and d < d_en:  d_en = d
    return np.array([
        p[X] / 100, p[Y] / 100, p[RADIUS] / 3, p[PROD] / 5,
        n_fr / 10,  n_en / 10,  n_neu / 10,
        min(d_fr, 100) / 100, min(d_en, 100) / 100,
    ], dtype=np.float32)


def score_planets(all_planets, player):
    """Return list of (score, planet, feat) sorted descending."""
    results = []
    for p in all_planets:
        feat    = _planet_feat(p, all_planets, player)
        nnue_sc = planet_nnue.forward_scalar(feat)
        base_sc = (1.0 + math.log(max(p[PROD], 1))) \
                * sun_angle_score(p[X], p[Y]) \
                * position_score(p[X], p[Y])
        results.append((base_sc + nnue_sc, p, feat))
    results.sort(key=lambda x: -x[0])
    return results


# ── ship-count scoring ─────────────────────────
def _ship_batch_feats(op, tp, e_sh, f_sh, d_ot, candidates) -> np.ndarray:
    base = np.array([
        op[X] / 100, op[Y] / 100, op[SHIPS] / 500, op[PROD] / 5,
        tp[X] / 100, tp[Y] / 100, tp[SHIPS] / 500, tp[PROD] / 5,
        e_sh / 200,  f_sh / 200,  d_ot / 141,
    ], dtype=np.float32)
    max_s = max(op[SHIPS], 1)
    ratios = np.array([s / max_s for s in candidates], dtype=np.float32)
    return np.column_stack([
        np.tile(base, (len(candidates), 1)),
        ratios[:, None],
    ])


def top_ship_counts(op, tp, e_sh, f_sh, d_ot):
    max_s = max(op[SHIPS], 1)
    if max_s <= 32:
        candidates = list(range(1, max_s + 1))
    else:
        log_idx    = np.unique(np.round(
            np.exp(np.linspace(0, np.log(max_s), 28))).astype(int))
        candidates = sorted(set(np.clip(log_idx, 1, max_s).tolist() + [1, max_s]))
    feats  = _ship_batch_feats(op, tp, e_sh, f_sh, d_ot, candidates)
    scores = ship_nnue.forward_batch(feats)
    idx    = np.argsort(-scores)[:TOP_M]
    return [(candidates[i], float(scores[i]), feats[i]) for i in idx]


def best_ship_count(op, tp, e_sh, f_sh, d_ot):
    return top_ship_counts(op, tp, e_sh, f_sh, d_ot)[0]


# ── timing scoring ─────────────────────────────
def _timing_base_feat(op, tp, w, e_wd, f_wd) -> np.ndarray:
    r_orb = dist(tp[X], tp[Y], SUN_X, SUN_Y)
    phi   = math.atan2(tp[Y] - SUN_Y, tp[X] - SUN_X)
    return np.array([
        op[X] / 100, op[Y] / 100,
        tp[X] / 100, tp[Y] / 100,
        r_orb / 50, phi / math.pi, w / 0.05,
        e_wd / 10000, f_wd / 10000,
    ], dtype=np.float32)


def top_timings(op, tp, w, e_wd, f_wd):
    base   = _timing_base_feat(op, tp, w, e_wd, f_wd)
    n_samp = min(36, MAX_N)
    ns     = np.unique(np.round(np.linspace(1, MAX_N, n_samp)).astype(int))
    feats  = np.column_stack([
        np.tile(base, (len(ns), 1)),
        (ns / MAX_N)[:, None],
    ]).astype(np.float32)
    scores = step_nnue.forward_batch(feats)
    idx    = np.argsort(-scores)[:TOP_L]
    return [(int(ns[i]), float(scores[i]), feats[i]) for i in idx]


def best_timing(op, tp, w, e_wd, f_wd):
    return top_timings(op, tp, w, e_wd, f_wd)[0]


# ══════════════════════════════════════════════
#  BOARD ENCODER  (Main NNUE)
# ══════════════════════════════════════════════
def encode_board(planets, fleets, player, fleet_cache: FleetCache) -> np.ndarray:
    pf   = np.zeros(MAX_PLANETS * 5, dtype=np.float32)
    pmap = {}
    for i, p in enumerate(sorted(planets, key=lambda p: p[ID])[:MAX_PLANETS]):
        b       = i * 5
        pf[b]   = 1.0 if p[OWNER] == player else 0.0
        pf[b+1] = 1.0 if p[OWNER] not in (-1, player) else 0.0
        pf[b+2] = 1.0 if p[OWNER] == -1 else 0.0
        pf[b+3] = p[SHIPS] / 500
        pf[b+4] = p[PROD]  / 5
        pmap[p[ID]] = p

    ff   = np.zeros(MAX_FLEETS * 7, dtype=np.float32)
    fc   = fleet_cache._cache
    for i, fl in enumerate(sorted(fleets, key=lambda f: f[FL_ID])[:MAX_FLEETS]):
        b        = i * 7
        ff[b]    = 1.0 if fl[FL_OWNER] == player else 0.0
        ff[b+1]  = 1.0 if fl[FL_OWNER] != player else 0.0
        ff[b+2]  = fl[FL_SHIPS] / 500
        ff[b+3]  = dist(fl[FL_X], fl[FL_Y], SUN_X, SUN_Y) / 70.0
        entry    = fc.get(str(fl[FL_ID]), {})
        tid      = entry.get("target")
        if tid is not None and tid in pmap:
            ff[b+4] = pmap[tid][X] / 100
            ff[b+5] = pmap[tid][Y] / 100
        ff[b+6]  = 1.0 if entry.get("inferred", True) else 0.0
    return np.concatenate([pf, ff])


def eval_board(planets, fleets, player,
               fleet_cache: FleetCache | None = None) -> tuple[float, np.ndarray]:
    fc   = fleet_cache if fleet_cache is not None else _fc
    feat = encode_board(planets, fleets, player, fc)
    val  = main_nnue.forward_scalar(feat)
    return val, feat


# ══════════════════════════════════════════════
#  CAPTURE ACTION BUILDERS
# ══════════════════════════════════════════════
def Cap_inn_orbs_raw(op, tp, ships, N, w) -> list:
    r_tp  = dist(tp[X], tp[Y], SUN_X, SUN_Y)
    phi   = math.atan2(tp[Y] - SUN_Y, tp[X] - SUN_X)
    xtp_N = SUN_X + r_tp * math.cos(w * N + phi)
    ytp_N = SUN_Y + r_tp * math.sin(w * N + phi)
    return [op[ID], math.atan2(ytp_N - op[Y], xtp_N - op[X]), ships]


def Cap_out_orbs_raw(op, tp, ships) -> list:
    return [op[ID], math.atan2(tp[Y] - op[Y], tp[X] - op[X]), ships]


# ══════════════════════════════════════════════
#  CANDIDATE MOVE GENERATOR
# ══════════════════════════════════════════════
def generate_candidates(planets, fleets, player, w,
                         fleet_cache: FleetCache) -> list[list[list]]:
    """
    Return a list of move-sets (each move-set = list of actions for one turn).
    Index 0 is the greedy-best set; subsequent sets are single-planet deviations.
    """
    scored   = score_planets(planets, player)
    owned_s  = [(s, p, f) for s, p, f in scored
                if p[OWNER] == player and p[SHIPS] >= 5]
    other_s  = [(s, p, f) for s, p, f in scored if p[OWNER] != player]
    if not owned_s or not other_s:
        return [[]]

    # Fill the first TOP_K-1 slots with the highest-scored non-owned planets.
    # The last slot is always the highest-scored *static* (non-orbiting) planet
    # that is not already present in those first TOP_K-1 slots.
    top_dynamic = [p for _, p, _ in other_s[:TOP_K - 1]]
    top_dynamic_ids = {p[ID] for p in top_dynamic}

    id_to_sc = {p[ID]: s for s, p, _ in scored}

    per_planet: dict[int, list] = {}   # planet_id → [(action, score, tp_id)]

    for _, op, _ in sorted(owned_s, key=lambda x: -x[0]):
        static_pin = next(
            (p for _, p, _ in other_s if not is_orbiting(p) and not _segment_sun_hit(op[X],op[Y],p[X],p[Y]) and p[ID] not in top_dynamic_ids),
            None,
        )

        if static_pin is not None:
            targets = top_dynamic + [static_pin]
        else:
        # Fallback: no eligible static planet — keep the original TOP_K list
            targets = [p for _, p, _ in other_s[:TOP_K]]
        cands = []
        for tp in targets:
            d_ot             = dist(op[X], op[Y], tp[X], tp[Y])
            e_sh, f_sh, e_wd, f_wd = fleet_cache.fleet_pressure(fleets, tp, player)
            ship_opts        = top_ship_counts(op, tp, e_sh, f_sh, d_ot)

            if is_orbiting(tp):
                time_opts = top_timings(op, tp, w, e_wd, f_wd)
                for ships, s_sc, _ in ship_opts:
                    for N, t_sc, _ in time_opts:
                        act   = Cap_inn_orbs_raw(op, tp, ships, N, w)
                        angle = act[1]
                        if filter_move_orb(op, tp, ships, angle, N):
                            pri = id_to_sc.get(tp[ID], 0) + s_sc + t_sc \
                                  - e_sh * 0.1 + f_sh * 0.05
                            cands.append((act, pri, tp[ID]))
            else:
                for ships, s_sc, _ in ship_opts:
                    act   = Cap_out_orbs_raw(op, tp, ships)
                    angle = act[1]
                    if filter_move_sta(op, tp, ships, angle):
                        pri = id_to_sc.get(tp[ID], 0) + s_sc \
                              - e_sh * 0.1 + f_sh * 0.05
                        cands.append((act, pri, tp[ID]))

        if cands:
            cands.sort(key=lambda x: -x[1])
            per_planet[op[ID]] = cands

    if not per_planet:
        return [[]]

    # Best move-set: greedy choice per planet
    best_set = [c[0][0] for c in per_planet.values()]
    all_sets = [best_set]

    # Alternative sets: swap one planet's action
    for op_id, cands in per_planet.items():
        for alt_act, _, _ in cands[1:3]:
            alt_set = []
            for oid2, c2 in per_planet.items():
                alt_set.append(alt_act if oid2 == op_id else c2[0][0])
            all_sets.append(alt_set)

    return all_sets


# ══════════════════════════════════════════════
#  SIMULATION HELPERS  (for search)
# ══════════════════════════════════════════════
def sim_apply_move(planets, fleets, actions, player):
    """
    Lightweight in-place simulation step.
    Returns (new_planets, new_fleets) as new lists (shallow-copy rows).
    """
    import copy as _copy
    planets = [list(p) for p in planets]
    fleets  = [list(f) for f in fleets]
    pmap    = {p[ID]: p for p in planets}
    new_fid = max((fl[FL_ID] for fl in fleets), default=0) + 1
    for act in actions:
        from_id, angle, ships = act
        if from_id not in pmap:
            continue
        op = pmap[from_id]
        ships = min(ships, op[SHIPS])
        if ships <= 0:
            continue
        op[SHIPS] -= ships
        fleets.append([new_fid, player,
                        op[X], op[Y], angle, from_id, ships])
        new_fid += 1
    for p in planets:
        if p[OWNER] == player:
            p[SHIPS] += p[PROD]
    return planets, fleets


def _count_players(planets):
    return len({p[OWNER] for p in planets if p[OWNER] != -1})


def _next_enemy(player, planets):
    owners = sorted({p[OWNER] for p in planets
                     if p[OWNER] not in (-1, player)})
    return owners[0] if owners else player


# ══════════════════════════════════════════════
#  ALPHA-BETA  (2-player)
# ══════════════════════════════════════════════
def alpha_beta(planets, fleets, player, depth,
               alpha, beta, maximising, w,
               fleet_cache: FleetCache, deadline: float):
    if depth == 0 or time.monotonic() >= deadline:
        val, _ = eval_board(planets, fleets, player, fleet_cache)
        return val, None

    cur = player if maximising else _next_enemy(player, planets)
    move_sets = generate_candidates(planets, fleets, cur, w, fleet_cache)
    if not move_sets or move_sets == [[]]:
        val, _ = eval_board(planets, fleets, player, fleet_cache)
        return val, None

    best_moves = None
    if maximising:
        val = -1e9
        for ms in move_sets:
            np_, fp_ = sim_apply_move(planets, fleets, ms, cur)
            sc, _    = alpha_beta(np_, fp_, player, depth - 1,
                                  alpha, beta, False, w, fleet_cache, deadline)
            if sc > val:
                val = sc;  best_moves = ms
            alpha = max(alpha, val)
            if alpha >= beta or time.monotonic() >= deadline:
                break
    else:
        val = 1e9
        for ms in move_sets:
            np_, fp_ = sim_apply_move(planets, fleets, ms, cur)
            sc, _    = alpha_beta(np_, fp_, player, depth - 1,
                                  alpha, beta, True, w, fleet_cache, deadline)
            if sc < val:
                val = sc;  best_moves = ms
            beta = min(beta, val)
            if alpha >= beta or time.monotonic() >= deadline:
                break

    return val, best_moves


# ══════════════════════════════════════════════
#  MCTS  (3-4 player)
# ══════════════════════════════════════════════
class _MCTSNode:
    __slots__ = ("planets", "fleets", "player", "w", "parent",
                 "move_set", "children", "visits", "value",
                 "untried", "fleet_cache")

    def __init__(self, planets, fleets, player, w,
                 fleet_cache: FleetCache, parent=None, move_set=None):
        self.planets     = planets
        self.fleets      = fleets
        self.player      = player
        self.w           = w
        self.parent      = parent
        self.move_set    = move_set     # action that led here
        self.children: list  = []
        self.visits      = 0
        self.value       = 0.0
        self.fleet_cache = fleet_cache
        # Lazily computed candidate move-sets
        self.untried: list | None = None

    def _ensure_untried(self):
        if self.untried is None:
            self.untried = generate_candidates(
                self.planets, self.fleets,
                self.player, self.w, self.fleet_cache)

    def is_fully_expanded(self) -> bool:
        self._ensure_untried()
        return len(self.untried) == 0

    def best_child(self, c: float) -> "_MCTSNode":
        log_n = math.log(self.visits + 1)
        def ucb(node):
            if node.visits == 0:
                return 1e9
            return node.value / node.visits + c * math.sqrt(log_n / node.visits)
        return max(self.children, key=ucb)

    def expand(self) -> "_MCTSNode":
        self._ensure_untried()
        ms           = self.untried.pop(0)
        np_, fp_     = sim_apply_move(self.planets, self.fleets, ms, self.player)
        child        = _MCTSNode(np_, fp_, self.player, self.w,
                                 self.fleet_cache, parent=self, move_set=ms)
        self.children.append(child)
        return child

    def rollout_value(self) -> float:
        val, _ = eval_board(self.planets, self.fleets,
                            self.player, self.fleet_cache)
        return val

    def backprop(self, value: float):
        self.visits += 1
        self.value  += value
        if self.parent:
            self.parent.backprop(value)


def mcts_search(planets, fleets, player, w,
                fleet_cache: FleetCache,
                n_sims: int, deadline: float) -> list:
    root = _MCTSNode(planets, fleets, player, w, fleet_cache)

    for _ in range(n_sims):
        if time.monotonic() >= deadline:
            break

        # Selection
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(MCTS_C)

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        # Evaluation (use main NNUE as rollout value)
        value = node.rollout_value()

        # Backpropagation
        node.backprop(value)

    if not root.children:
        # Fallback: greedy best
        cands = generate_candidates(planets, fleets, player, w, fleet_cache)
        return cands[0] if cands else []

    best = max(root.children,
               key=lambda n: n.visits if n.visits > 0 else -1e9)
    return best.move_set if best.move_set else []


# ══════════════════════════════════════════════
#  PER-GAME STATE  (reset between games)
# ══════════════════════════════════════════════
_current_step    = 0
_move_log: list  = []
_history: dict   = {}


def reset_game_state():
    global _current_step, _move_log, _history
    _current_step = 0
    _move_log.clear()
    _history.clear()
    _fc.reset()


# ── record helpers ─────────────────────────────
def record_move(step, player, from_pid, target_pid, decisions):
    _move_log.append({
        "step":             step,
        "player":           player,
        "from_planet_id":   from_pid,
        "target_planet_id": target_pid,
        "decisions":        decisions,
    })


def record_history(step, planets):
    _history[step] = {str(p[ID]): p[OWNER] for p in planets}


def get_move_log()  -> list:  return _move_log
def get_history()   -> dict:  return _history
def get_fleet_cache() -> FleetCache: return _fc


# ══════════════════════════════════════════════
#  MAIN AGENT
# ══════════════════════════════════════════════
def agent(obs, config=None):
    global _current_step
    t_start  = time.monotonic()
    deadline = t_start + MOVE_BUDGET

    step    = _current_step
    _current_step += 1

    player  = obs["player"]
    planets = obs["planets"]
    fleets  = obs.get("fleets", [])
    w       = obs.get("angular_velocity", 0.03)

    record_history(step, planets)

    owned  = [p for p in planets if p[OWNER] == player]
    others = [p for p in planets if p[OWNER] != player]
    if not owned or not others:
        return []

    _fc.update(fleets, planets, player, w, step)

    n_players = _count_players(planets)

    # ── Score board once ────────────────────────
    scored   = score_planets(planets, player)
    owned_s  = [(s, p, f) for s, p, f in scored
                if p[OWNER] == player and p[SHIPS] >= 5]
    other_s  = [(s, p, f) for s, p, f in scored if p[OWNER] != player]
    if not owned_s or not other_s:
        return []
    id_to_sc = {p[ID]: s for s, p, _ in scored}

    # ── Board eval for logging ──────────────────
    main_sc, main_feat = eval_board(planets, fleets, player, _fc)

    # ── Search ──────────────────────────────────
    if n_players <= 2:
        # Alpha-Beta
        _, best_move_set = alpha_beta(
            planets, fleets, player,
            AB_DEPTH_2P, -1e9, 1e9, True, w, _fc, deadline)
    else:
        # MCTS
        best_move_set = mcts_search(
            planets, fleets, player, w, _fc, MCTS_SIMS, deadline)

    # Fallback to greedy if search returned nothing
    if not best_move_set:
        best_move_set = generate_candidates(
            planets, fleets, player, w, _fc)[0]

    # ── Record decisions for training ───────────
    for act in best_move_set:
        from_id = act[0]
        # Find which target this corresponds to
        angle   = act[1]
        ships   = act[2]
        # Identify target planet by closest angle match
        best_tp = None;  best_da = 1e9
        for _, tp, _ in other_s:
            da = abs(math.atan2(tp[Y] - next(
                (p for p in planets if p[ID] == from_id), planets[0])[Y],
                tp[X] - next(
                (p for p in planets if p[ID] == from_id), planets[0])[X])
                - angle)
            if da < best_da:
                best_da = da;  best_tp = tp

        op = next((p for p in planets if p[ID] == from_id), None)
        if op is None or best_tp is None:
            continue

        p_feat  = _planet_feat(op, planets, player)
        d_ot    = dist(op[X], op[Y], best_tp[X], best_tp[Y])
        e_sh, f_sh, e_wd, f_wd = _fc.fleet_pressure(fleets, best_tp, player)
        s_ships, s_sc, s_feat   = best_ship_count(op, best_tp, e_sh, f_sh, d_ot)

        if is_orbiting(best_tp):
            N, t_sc, t_feat = best_timing(op, best_tp, w, e_wd, f_wd)
        else:
            N, t_sc, t_feat = 1, 0.0, np.zeros(10, dtype=np.float32)

        record_move(step, player, from_id, best_tp[ID], {
            "planet": {"feat": p_feat.tolist(),  "out": float(id_to_sc.get(op[ID], 0))},
            "ship":   {"feat": s_feat.tolist(),  "out": float(s_sc)},
            "step":   {"feat": t_feat.tolist(),  "out": float(t_sc)},
            "main":   {"feat": main_feat.tolist(),"out": float(main_sc)},
        })
        _fc.pending[from_id] = best_tp[ID]

    return best_move_set