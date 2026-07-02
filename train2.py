"""
train.py — Orbit Wars NNUE training.

Two independent training modes:
  1. replay_train(json_path)   — learn directly from a saved replay JSON
                                 (the 81506445.json format from Kaggle)
  2. train(n_games)            — parallel self-play (6 workers, MCTS-only)

Both modes share the same training machinery:
  • Hop-chain signal  → Minor NNUEs  (planet / ship / step)
  • Event-weighted signal → Main NNUE  (GPU-batched if PyTorch available)
  • Replay buffer  (capacity REPLAY_CAP games)  → mix old + new samples
  • Checkpoint tournament every TOURNAMENT_EVERY games; promote if win > 55 %

Hop-chain uses depth-discounted propagation (γ = HOP_GAMMA ≈ 0.8).

Run:
  python train.py                       # self-play, 200 games
  python train.py 500                   # self-play, 500 games
  python train.py --replay game.json    # train from one replay file
  python train.py --replay game.json 50 # replay bootstrap then 50 self-play games
"""

from __future__ import annotations

import sys, os, json, copy, math, time, pickle, importlib.util, collections, random
import multiprocessing as mp
import numpy as np

# ── Optional GPU ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS & HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

AGENT_PATH       = os.path.join(_HERE, "Orbit_wars.py")
CURRENT_WEIGHTS  = "nnue_weights.pkl"
BEST_WEIGHTS     = "nnue_weights_best.pkl"

N_WORKERS        = 6
TOURNAMENT_EVERY = 20      # self-play games between tournaments
TOURNAMENT_GAMES = 16      # games per tournament match
WIN_THRESHOLD    = 0.55    # promotion bar

REPLAY_CAP       = 50      # max games kept in replay buffer
REPLAY_OLD_FRAC  = 0.3     # fraction of training batch from old games

MAIN_BATCH_SIZE  = 64      # mini-batch size for main NNUE
HOP_GAMMA        = 0.80    # depth discount for hop-chain recursion

# ══════════════════════════════════════════════════════════════════════════════
#  AGENT MODULE LOADER
# ══════════════════════════════════════════════════════════════════════════════
def _load_agent_module(path: str = AGENT_PATH):
    spec = importlib.util.spec_from_file_location("orbit_wars_agent", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reset_agent(ow):
    """Reset all mutable per-game state on an agent module."""
    ow._current_step = 0
    ow._move_log.clear()
    ow._history.clear()
    ow.get_fleet_cache().reset()


# ══════════════════════════════════════════════════════════════════════════════
#  GPU-ACCELERATED MAIN NNUE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
class TorchMainNNUE(nn.Module if _TORCH else object):
    """
    Drop-in GPU wrapper for the numpy main NNUE.
    Degrades gracefully to numpy-only when PyTorch is unavailable.
    """

    def __init__(self, numpy_nnue):
        self._np = numpy_nnue
        if not _TORCH:
            return
        super().__init__()
        in_dim  = numpy_nnue.W1.shape[0]
        l1      = numpy_nnue.W1.shape[1]
        l2      = numpy_nnue.W2.shape[1]
        out_dim = numpy_nnue.W3.shape[1]
        self.fc1 = nn.Linear(in_dim, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, out_dim)
        self._sync_from_numpy()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)
        self._opt = torch.optim.Adam(self.parameters(), lr=numpy_nnue.lr)

    # ── numpy ↔ torch sync ────────────────────────────────────────────────────
    def _sync_from_numpy(self):
        if not _TORCH:
            return
        self.fc1.weight.data = torch.tensor(self._np.W1.T.copy())
        self.fc1.bias.data   = torch.tensor(self._np.b1.copy())
        self.fc2.weight.data = torch.tensor(self._np.W2.T.copy())
        self.fc2.bias.data   = torch.tensor(self._np.b2.copy())
        self.fc3.weight.data = torch.tensor(self._np.W3.T.copy())
        self.fc3.bias.data   = torch.tensor(self._np.b3.copy())

    def _sync_to_numpy(self):
        if not _TORCH:
            return
        self._np.W1 = self.fc1.weight.data.cpu().numpy().T.astype(np.float32)
        self._np.b1 = self.fc1.bias.data.cpu().numpy().astype(np.float32)
        self._np.W2 = self.fc2.weight.data.cpu().numpy().T.astype(np.float32)
        self._np.b2 = self.fc2.bias.data.cpu().numpy().astype(np.float32)
        self._np.W3 = self.fc3.weight.data.cpu().numpy().T.astype(np.float32)
        self._np.b3 = self.fc3.bias.data.cpu().numpy().astype(np.float32)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    def train_batch(self, feats: np.ndarray, targets: np.ndarray):
        """feats: (N, in_dim) float32 | targets: (N,) float32"""
        if not _TORCH:
            for feat, tgt in zip(feats, targets):
                pred = self._np.forward_scalar(feat)
                grad = np.array([pred - tgt], dtype=np.float32)
                self._np.backward(grad)
            return
        X = torch.tensor(feats,   dtype=torch.float32, device=self._device)
        Y = torch.tensor(targets, dtype=torch.float32, device=self._device).unsqueeze(1)
        self._opt.zero_grad()
        loss = nn.functional.mse_loss(self.forward(X), Y)
        loss.backward()
        self._opt.step()
        self._sync_to_numpy()

    def eval_batch_np(self, feats: np.ndarray) -> np.ndarray:
        if not _TORCH:
            return self._np.forward_batch(feats)
        with torch.no_grad():
            X = torch.tensor(feats, dtype=torch.float32, device=self._device)
            return self.forward(X).cpu().numpy().ravel()


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Circular buffer of game result dicts.
    Each entry: {"move_log", "history", "fc_dump", "n_steps", "winner", "rewards"}.
    Mixes REPLAY_OLD_FRAC of old entries with new ones during training.
    """

    def __init__(self, capacity: int = REPLAY_CAP):
        self._buf: collections.deque = collections.deque(maxlen=capacity)

    def add(self, result: dict):
        self._buf.append(result)

    def sample_mixed(self, new_results: list[dict]) -> list[dict]:
        """Return new_results + a random sample of old buffer entries."""
        old = list(self._buf)
        n_old = max(1, int(len(new_results) * REPLAY_OLD_FRAC / (1 - REPLAY_OLD_FRAC + 1e-9)))
        n_old = min(n_old, len(old))
        sampled_old = random.sample(old, n_old) if n_old > 0 else []
        return new_results + sampled_old

    def __len__(self):
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
#  HOP-CHAIN SIGNAL  (depth-discounted)
# ══════════════════════════════════════════════════════════════════════════════
def compute_hop_signal(target_pid: int, launch_step: int, player: int,
                        history: dict, move_log: list, total_steps: int,
                        gamma: float = HOP_GAMMA) -> float:
    """
    Reward for sending a fleet at launch_step toward target_pid.

    direct_reward = (owned_steps - enemy_steps) / total_steps
    hop reward    = γ^depth × hop_score(child)

    Clips final signal to [-1, 1].
    """
    visited: set = set()

    def hop_score(pid: int, from_step: int, depth: int) -> float:
        if pid in visited:
            return 0.0
        visited.add(pid)
        spid = str(pid)

        # First step ≥ from_step where *player* owns this planet
        capture_step = None
        for t in range(from_step, total_steps):
            if history.get(str(t), {}).get(spid) == player:
                capture_step = t
                break
        if capture_step is None:
            return 0.0

        owned_steps = enemy_steps = 0
        for t in range(capture_step, total_steps):
            owner = history.get(str(t), {}).get(spid, -1)
            if   owner == player: owned_steps += 1
            elif owner != -1:     enemy_steps += 1

        direct = (owned_steps - enemy_steps) / (total_steps + 1e-9)

        # Recurse into fleets launched FROM this planet post-capture
        hop_total = 0.0
        for rec in move_log:
            if rec["step"] < capture_step:       continue
            if rec["player"] != player:          continue
            if rec.get("from_planet_id") != pid: continue
            child_tid = rec.get("target_planet_id")
            if child_tid is not None:
                hop_total += (gamma ** (depth + 1)) * hop_score(child_tid,
                                                                  rec["step"],
                                                                  depth + 1)
        return direct + hop_total

    raw = hop_score(target_pid, launch_step, depth=0)
    return float(np.clip(raw, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  MINOR NNUE TRAINING  (hop-chain, CPU)
# ══════════════════════════════════════════════════════════════════════════════
def train_minor_nnues(ow, move_log: list, history: dict,
                      player: int, total_steps: int):
    nnues = {
        "planet": ow.planet_nnue,
        "ship":   ow.ship_nnue,
        "step":   ow.step_nnue,
    }
    for rec in move_log:
        if rec["player"] != player:
            continue
        tid = rec.get("target_planet_id")
        if tid is None:
            continue
        signal = compute_hop_signal(
            tid, rec["step"], player, history, move_log, total_steps)
        for tag, nnue in nnues.items():
            dec = rec["decisions"].get(tag)
            if dec is None:
                continue
            feat      = np.array(dec["feat"], dtype=np.float32)
            pred      = nnue.forward_scalar(feat)
            loss_grad = np.array([pred - signal], dtype=np.float32)
            nnue.backward(loss_grad)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NNUE TRAINING  (event-weighted, batched)
# ══════════════════════════════════════════════════════════════════════════════
def collect_main_samples(ow, move_log: list, player: int):
    """Return (feats, targets) or (None, None) when nothing to train on."""
    fc = ow.get_fleet_cache()
    feats, targets = [], []
    for rec in move_log:
        if rec["player"] != player:
            continue
        dec = rec["decisions"].get("main")
        if dec is None:
            continue
        signal = fc.net_score_for_step(rec["step"])
        if signal is None:
            continue
        feats.append(dec["feat"])
        targets.append(signal)
    if not feats:
        return None, None
    return (np.array(feats, dtype=np.float32),
            np.array(targets, dtype=np.float32))


def train_main_nnue(gpu_nnue: TorchMainNNUE, ow,
                    move_log: list, player: int,
                    batch_size: int = MAIN_BATCH_SIZE):
    feats, targets = collect_main_samples(ow, move_log, player)
    if feats is None:
        return
    idx = np.random.permutation(len(feats))
    for start in range(0, len(idx), batch_size):
        bi = idx[start:start + batch_size]
        gpu_nnue.train_batch(feats[bi], targets[bi])


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN ON A SINGLE GAME RESULT
# ══════════════════════════════════════════════════════════════════════════════
def train_on_result(ow_main, gpu_nnue: TorchMainNNUE, res: dict,
                    n_players: int = 2):
    """Apply one full training pass (all players) from a result dict."""
    move_log = res["move_log"]
    history  = res["history"]
    n_steps  = res["n_steps"]

    ow_main.get_fleet_cache().load(res["fc_dump"])

    for player in range(n_players):
        train_minor_nnues(ow_main, move_log, history, player, n_steps)
        train_main_nnue(gpu_nnue, ow_main, move_log, player)


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY JSON → TRAINING RESULT
#  Converts the Kaggle episode JSON into the same dict format used by workers.
# ══════════════════════════════════════════════════════════════════════════════
def _result_from_replay_json(replay: dict, ow) -> dict:
    """
    Parse a Kaggle replay JSON and re-run the agent over every step so we
    collect move_log / history / fleet-cache events — identical to what a
    self-play worker would produce.

    The replay is a 2-player game (rewards list has 2 entries).
    We infer:
      • angular_velocity from the first observation
      • winner from final rewards
      • n_steps from number of steps

    Returns a result dict compatible with train_on_result().
    """
    steps    = replay["steps"]          # list of per-step agent arrays
    rewards  = replay["rewards"]        # [r0, r1]
    n_steps  = len(steps)
    n_agents = len(rewards)

    # Detect winner
    winner = int(np.argmax(rewards)) if any(r is not None for r in rewards) else 0

    _reset_agent(ow)

    for step_idx, step_agents in enumerate(steps):
        # step_agents is a list with one entry per agent
        # Each entry: {action, info, observation, reward, status}
        for agent_idx in range(n_agents):
            entry = step_agents[agent_idx]
            obs   = entry.get("observation", {})
            if not obs:
                continue

            # Override player field to match the agent slot
            obs_copy = dict(obs)
            obs_copy["player"] = agent_idx

            # Skip dead agents (no planets owned)
            planets = obs_copy.get("planets", [])
            owned   = [p for p in planets if p[1] == agent_idx]
            others  = [p for p in planets if p[1] != agent_idx]
            if not owned or not others:
                continue

            # Run agent — we don't use the returned action,
            # we only need the side-effects: record_history, record_move,
            # fleet-cache update.
            try:
                ow.agent(obs_copy, config=None)
            except Exception:
                pass   # never let a bad step crash training

    move_log = copy.deepcopy(ow.get_move_log())
    history  = copy.deepcopy(ow.get_history())
    fc_dump  = ow.get_fleet_cache().dump()

    return {
        "seed":     replay.get("info", {}).get("seed", -1),
        "move_log": move_log,
        "history":  history,
        "fc_dump":  fc_dump,
        "n_steps":  n_steps,
        "winner":   winner,
        "rewards":  rewards,
        "n_agents": n_agents,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC: TRAIN FROM A REPLAY JSON FILE
# ══════════════════════════════════════════════════════════════════════════════
def replay_train(json_path: str,
                 weights_in:  str = CURRENT_WEIGHTS,
                 weights_out: str = CURRENT_WEIGHTS,
                 verbose: bool = True) -> dict:
    """
    Bootstrap NNUE weights from a single saved replay file.

    Steps
    ─────
    1. Load agent module + weights.
    2. Walk every step of the replay, calling ow.agent() to populate
       move_log / history / fleet-cache (same data a self-play worker produces).
    3. Run the standard training pass (minor NNUEs + main NNUE).
    4. Save updated weights.

    Returns the result dict for optional further use (e.g. seeding replay buffer).
    """
    if verbose:
        print(f"[replay_train] Loading replay: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        replay = json.load(f)

    rewards  = replay.get("rewards", [0, 0])
    n_agents = len(rewards)

    if verbose:
        print(f"  agents={n_agents}  steps={len(replay['steps'])}"
              f"  rewards={rewards}")

    # ── Load agent + weights ──────────────────────────────────────────────────
    ow = _load_agent_module()
    ow.load_weights(weights_in)
    gpu_nnue = TorchMainNNUE(ow.main_nnue)

    # ── Parse replay → result dict ────────────────────────────────────────────
    res = _result_from_replay_json(replay, ow)

    if verbose:
        print(f"  move_log entries : {len(res['move_log'])}")
        print(f"  history  steps   : {len(res['history'])}")
        print(f"  winner           : P{res['winner']}")

    # ── Train ─────────────────────────────────────────────────────────────────
    train_on_result(ow, gpu_nnue, res, n_players=n_agents)

    if _TORCH:
        gpu_nnue._sync_to_numpy()

    # ── Save ──────────────────────────────────────────────────────────────────
    ow.save_weights(weights_out)
    if not os.path.exists(BEST_WEIGHTS):
        ow.save_weights(BEST_WEIGHTS)

    if verbose:
        print(f"  Weights saved → {weights_out}")

    return res


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-PLAY WORKER  (subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def _worker(worker_id: int, seeds: list, weights_path: str,
            result_queue: mp.Queue):
    """
    Run a batch of self-play games (MCTS always; no Alpha-Beta in training).
    Each game result is pushed onto result_queue.
    """
    # Import kaggle_environments lazily inside the worker to avoid
    # serialisation issues with the env registration in the main process.
    import importlib.util as _ilu, json as _json
    import kaggle_environments as _ke

    _ENV_DIR = os.environ.get("ORBIT_WARS_ENV_DIR", "")
    if _ENV_DIR:
        spec_mod = _ilu.spec_from_file_location(
            "orbit_wars_env", os.path.join(_ENV_DIR, "orbit_wars.py"))
        ow_env = _ilu.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(ow_env)
        with open(os.path.join(_ENV_DIR, "orbit_wars.json")) as _f:
            _spec = _json.load(_f)
        _ke.register("orbit_wars", {
            "specification": _spec,
            "interpreter":   ow_env.interpreter,
            "renderer":      ow_env.renderer,
            "html_renderer": ow_env.html_renderer,
        })

    ow = _load_agent_module()
    ow.load_weights(weights_path)

    # Force MCTS regardless of player count (training always uses MCTS)
    _orig_ab = ow.AB_DEPTH_2P
    ow.AB_DEPTH_2P = 0   # depth-0 alpha-beta degrades to MCTS fallback

    for seed in seeds:
        _reset_agent(ow)

        try:
            env = _ke.make("orbit_wars", configuration={"seed": seed}, debug=False)

            def make_agent(pid):
                def _a(obs, config=None):
                    o = dict(obs)
                    o["player"] = pid
                    return ow.agent(o, config)
                return _a

            n_agents = 4
            agents   = [make_agent(i) for i in range(n_agents)]
            env.run(agents)

            final   = env.steps[-1]
            rewards = [final[i].reward for i in range(n_agents)]
            winner  = int(np.argmax(rewards))
            n_steps = len(env.steps)

        except Exception as exc:
            # On any env error, push a minimal placeholder so the main
            # process doesn't hang waiting for a result.
            result_queue.put({
                "seed": seed, "move_log": [], "history": {},
                "fc_dump": {"cache": {}, "events": [], "done_events": []},
                "n_steps": 0, "winner": -1, "rewards": [],
                "n_agents": 4, "error": str(exc),
            })
            continue

        result_queue.put({
            "seed":     seed,
            "move_log": copy.deepcopy(ow.get_move_log()),
            "history":  copy.deepcopy(ow.get_history()),
            "fc_dump":  ow.get_fleet_cache().dump(),
            "n_steps":  n_steps,
            "winner":   winner,
            "rewards":  rewards,
            "n_agents": n_agents,
        })

    ow.AB_DEPTH_2P = _orig_ab


# ══════════════════════════════════════════════════════════════════════════════
#  TOURNAMENT
# ══════════════════════════════════════════════════════════════════════════════
def _register_env_main():
    """Register the Kaggle env in the main process (only needed for tournament)."""
    import kaggle_environments as _ke
    _ENV_DIR = os.environ.get("ORBIT_WARS_ENV_DIR", "")
    if not _ENV_DIR:
        return False
    try:
        spec_mod = importlib.util.spec_from_file_location(
            "orbit_wars_env", os.path.join(_ENV_DIR, "orbit_wars.py"))
        ow_env = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(ow_env)
        with open(os.path.join(_ENV_DIR, "orbit_wars.json")) as f:
            _spec = json.load(f)
        _ke.register("orbit_wars", {
            "specification": _spec,
            "interpreter":   ow_env.interpreter,
            "renderer":      ow_env.renderer,
            "html_renderer": ow_env.html_renderer,
        })
        return True
    except Exception as e:
        print(f"  [warn] env registration failed: {e}")
        return False


def run_tournament(current_path: str, best_path: str,
                   n_games: int = TOURNAMENT_GAMES) -> float:
    """
    Return win-rate of *current* weights vs *best* weights over n_games.
    current = players 0, 2  |  best = players 1, 3.
    Falls back to 0.0 if the kaggle env is not registered.
    """
    try:
        import kaggle_environments as _ke
        from kaggle_environments import make as _make
    except ImportError:
        print("  [warn] kaggle_environments not available — skipping tournament")
        return 0.0

    wins = 0
    for g in range(n_games):
        ow_cur  = _load_agent_module()
        ow_best = _load_agent_module()
        ow_cur.load_weights(current_path)
        ow_best.load_weights(best_path)
        _reset_agent(ow_cur)
        _reset_agent(ow_best)

        def make_cur(pid):
            def _a(obs, cfg=None):
                o = dict(obs); o["player"] = pid
                return ow_cur.agent(o, cfg)
            return _a

        def make_best(pid):
            def _a(obs, cfg=None):
                o = dict(obs); o["player"] = pid
                return ow_best.agent(o, cfg)
            return _a

        try:
            env = _make("orbit_wars",
                        configuration={"seed": 10_000 + g}, debug=False)
            agents = [make_cur(0), make_best(1), make_cur(2), make_best(3)]
            env.run(agents)
            final      = env.steps[-1]
            rewards    = [final[i].reward for i in range(4)]
            cur_score  = rewards[0] + rewards[2]
            best_score = rewards[1] + rewards[3]
            if cur_score > best_score:
                wins += 1
        except Exception as exc:
            print(f"  [warn] tournament game {g} failed: {exc}")

    return wins / n_games if n_games > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-PLAY TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train(n_games: int = 200, save_every: int = 10,
          verbose: bool = True,
          seed_results: list | None = None):
    """
    Parallel self-play training loop.

    seed_results: optional list of result dicts to pre-load the replay buffer
                  (e.g. from replay_train).
    """
    print(f"[train] self-play: {n_games} games, {N_WORKERS} workers")
    if _TORCH:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  GPU main NNUE: {dev}")
    else:
        print("  PyTorch not available — CPU-only training")

    ow_main  = _load_agent_module()
    ow_main.load_weights(CURRENT_WEIGHTS)
    gpu_nnue = TorchMainNNUE(ow_main.main_nnue)

    if not os.path.exists(BEST_WEIGHTS):
        ow_main.save_weights(BEST_WEIGHTS)

    replay_buf = ReplayBuffer(capacity=REPLAY_CAP)
    if seed_results:
        for r in seed_results:
            replay_buf.add(r)
        print(f"  Replay buffer seeded with {len(replay_buf)} result(s)")

    # Try to register env for tournaments
    env_ok = _register_env_main()

    games_done = 0

    while games_done < n_games:
        batch_size = min(N_WORKERS * 2, n_games - games_done)
        seeds      = list(range(games_done, games_done + batch_size))
        per_worker = math.ceil(batch_size / N_WORKERS)
        worker_batches = [
            seeds[i * per_worker:(i + 1) * per_worker]
            for i in range(N_WORKERS)
            if seeds[i * per_worker:(i + 1) * per_worker]
        ]

        result_q: mp.Queue = mp.Queue()
        procs = []
        for wid, wseeds in enumerate(worker_batches):
            p = mp.Process(
                target=_worker,
                args=(wid, wseeds, CURRENT_WEIGHTS, result_q),
                daemon=True,
            )
            p.start()
            procs.append(p)

        new_results = []
        for _ in range(batch_size):
            new_results.append(result_q.get())
        for p in procs:
            p.join(timeout=10)

        # Mix new + old replay samples
        mixed = replay_buf.sample_mixed(new_results)

        for res in mixed:
            if res.get("error"):
                if verbose:
                    print(f"  [skip] seed={res['seed']} error: {res['error']}")
                continue
            n_agents = res.get("n_agents", 4)
            train_on_result(ow_main, gpu_nnue, res, n_players=n_agents)

        # Add only fresh results to the replay buffer
        for res in new_results:
            if not res.get("error"):
                replay_buf.add(res)

        games_done += len(new_results)

        if _TORCH:
            gpu_nnue._sync_to_numpy()

        if verbose:
            for res in new_results:
                tag = f"  Game {games_done:4d} | seed={res['seed']}"
                if res.get("error"):
                    print(tag + f" [ERROR: {res['error']}]")
                else:
                    print(tag + f" | winner=P{res['winner']}"
                                f" | steps={res['n_steps']}"
                                f" | buf={len(replay_buf)}")

        if games_done % save_every == 0:
            ow_main.save_weights(CURRENT_WEIGHTS)
            print(f"  → Weights saved at game {games_done}")

        if env_ok and games_done % TOURNAMENT_EVERY == 0:
            ow_main.save_weights(CURRENT_WEIGHTS)
            print(f"\n  → Tournament @ game {games_done} ...", end=" ", flush=True)
            win_rate = run_tournament(CURRENT_WEIGHTS, BEST_WEIGHTS)
            print(f"win-rate={win_rate:.2f}", end="  ")
            if win_rate >= WIN_THRESHOLD:
                ow_main.save_weights(BEST_WEIGHTS)
                print("PROMOTED ✓")
            else:
                ow_main.load_weights(BEST_WEIGHTS)
                if _TORCH:
                    gpu_nnue._sync_from_numpy()
                print("kept previous best")
            print()

    # Final save + optional promotion
    ow_main.save_weights(CURRENT_WEIGHTS)
    if env_ok:
        wr = run_tournament(CURRENT_WEIGHTS, BEST_WEIGHTS, n_games=TOURNAMENT_GAMES)
        if wr >= WIN_THRESHOLD:
            ow_main.save_weights(BEST_WEIGHTS)
            print(f"Final weights promoted (win-rate={wr:.2f})")
        else:
            print(f"Final weights NOT promoted (win-rate={wr:.2f})")
    else:
        print("Tournament skipped (env not registered) — saved as current only.")

    print("\n[train] Complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = sys.argv[1:]

    # ── Mode: --replay <file> [n_self_play_games] ─────────────────────────────
    if args and args[0] == "--replay":
        if len(args) < 2:
            print("Usage: python train.py --replay <replay.json> [n_games]")
            sys.exit(1)
        json_path = args[1]
        n_sp      = int(args[2]) if len(args) >= 3 else 0

        res = replay_train(json_path, verbose=True)

        if n_sp > 0:
            print(f"\n[train] Switching to self-play for {n_sp} games ...")
            train(n_games=n_sp, seed_results=[res], verbose=True)

    # ── Mode: pure self-play ──────────────────────────────────────────────────
    else:
        n = int(args[0]) if args else 200
        train(n_games=n, verbose=True)