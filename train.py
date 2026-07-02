"""
train.py — Parallel self-play training for Orbit Wars.

Architecture
────────────
  6 parallel self-play workers (multiprocessing)
  │
  ├─ Each worker:
  │    3 Minor NNUEs → top-k candidates
  │    Move filters
  │    MCTS (main NNUE evaluates leaves)
  │    Collect (state, value, move) samples
  │
  ├─ Main process:
  │    Aggregate samples from all workers
  │    Train Minor NNUEs  (hop-chain signal, CPU)
  │    Train Main NNUE    (event-weighted signal, GPU if available)
  │
  └─ Checkpoint tournament every N games:
       Compare current vs best; promote only if win-rate > 55 %

Run: python train.py [n_games]
"""

import sys, os, json, copy, math, time, pickle, importlib.util
import multiprocessing as mp
import numpy as np

# ── Optional GPU via PyTorch for main NNUE ────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

# ── Kaggle environment registration ───────────────────────────────────────────
import kaggle_environments

_ENV_DIR = os.environ.get(
    "ORBIT_WARS_ENV_DIR",
    os.path.join(os.path.dirname(__file__), "orbit_wars_env")
)


def _register_env():
    spec_mod = importlib.util.spec_from_file_location(
        "orbit_wars_env", os.path.join(_ENV_DIR, "orbit_wars.py"))
    ow_env = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(ow_env)
    with open(os.path.join(_ENV_DIR, "orbit_wars.json")) as f:
        _spec = json.load(f)
    kaggle_environments.register("orbit_wars", {
        "specification": _spec,
        "interpreter":   ow_env.interpreter,
        "renderer":      ow_env.renderer,
        "html_renderer": ow_env.html_renderer,
    })


_register_env()
from kaggle_environments import make   # noqa: E402 (must come after register)


# ── Load the agent module ──────────────────────────────────────────────────────
def _load_agent_module(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "orbit_wars.py")
    spec = importlib.util.spec_from_file_location("orbit_wars", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
#  GPU-ACCELERATED MAIN NNUE WRAPPER  (optional)
# ══════════════════════════════════════════════════════════════════════════════
class TorchMainNNUE(nn.Module if _TORCH else object):
    """
    Drop-in GPU wrapper around the main NNUE.
    When PyTorch is unavailable the class degrades gracefully and all
    methods delegate to the NumPy NNUE.
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
        self.fc2 = nn.Linear(l1,     l2)
        self.fc3 = nn.Linear(l2,  out_dim)
        self._sync_from_numpy()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)
        self._opt = torch.optim.SGD(self.parameters(), lr=numpy_nnue.lr)

    def _sync_from_numpy(self):
        if not _TORCH:
            return
        self.fc1.weight.data = torch.tensor(self._np.W1.T)
        self.fc1.bias.data   = torch.tensor(self._np.b1)
        self.fc2.weight.data = torch.tensor(self._np.W2.T)
        self.fc2.bias.data   = torch.tensor(self._np.b2)
        self.fc3.weight.data = torch.tensor(self._np.W3.T)
        self.fc3.bias.data   = torch.tensor(self._np.b3)

    def _sync_to_numpy(self):
        if not _TORCH:
            return
        self._np.W1 = self.fc1.weight.data.cpu().numpy().T.astype(np.float32)
        self._np.b1 = self.fc1.bias.data.cpu().numpy().astype(np.float32)
        self._np.W2 = self.fc2.weight.data.cpu().numpy().T.astype(np.float32)
        self._np.b2 = self.fc2.bias.data.cpu().numpy().astype(np.float32)
        self._np.W3 = self.fc3.weight.data.cpu().numpy().T.astype(np.float32)
        self._np.b3 = self.fc3.bias.data.cpu().numpy().astype(np.float32)

    def forward(self, x):   # torch.Tensor → torch.Tensor
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    def train_batch(self, feats: np.ndarray, targets: np.ndarray):
        """
        feats:   (N, 340) float32
        targets: (N,)     float32
        """
        if not _TORCH:
            # Fallback: train numpy NNUE sample by sample
            for feat, tgt in zip(feats, targets):
                self._np.forward(feat)
                grad = np.array([self._np.forward_scalar(feat) - tgt],
                                 dtype=np.float32)
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
        """Return (N,) float32 predictions, always on CPU."""
        if not _TORCH:
            return self._np.forward_batch(feats)
        with torch.no_grad():
            X   = torch.tensor(feats, dtype=torch.float32, device=self._device)
            out = self.forward(X).cpu().numpy().ravel()
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  GAME RESET HELPER
# ══════════════════════════════════════════════════════════════════════════════
def reset_game_state(ow):
    ow._current_step = 0
    ow._move_log.clear()
    ow._history.clear()
    ow.get_fleet_cache().reset()


# ══════════════════════════════════════════════════════════════════════════════
#  HOP-CHAIN SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
def compute_hop_signal(target_pid, launch_step, player,
                       history, move_log, total_steps,
                       gamma: float = 0.88) -> float:
    visited: set = set()

    def hop_score(pid, from_step, depth: int = 0) -> float:
        if pid in visited:
            return 0.0
        visited.add(pid)
        spid = str(pid)

        # First step >= from_step where player owns this planet
        capture_step = None
        for t in range(from_step, total_steps):
            if history.get(str(t), {}).get(spid) == player:
                capture_step = t
                break
        if capture_step is None:
            return 0.0

        # ── per-timestep discounted ownership signal ──────────────────────
        # Each owned/contested step is discounted by gamma^(t - capture_step)
        # so earlier stable control is worth more than late-game holdings.
        direct = 0.0
        for t in range(capture_step, total_steps):
            owner = history.get(str(t), {}).get(spid, -1)
            if owner == player:
                step_val = 1.0
            elif owner != -1:
                step_val = -1.0
            else:
                step_val = 0.0

            # Discount within the planet's own ownership window
            time_decay = gamma ** (t - capture_step)
            direct += step_val * time_decay

        # Normalise to (-1, 1) range the same way the original did
        direct /= (total_steps + 1e-9)

        # ── recurse into child hops ───────────────────────────────────────
        # Each hop level is additionally discounted by gamma^depth so a
        # chain  A → B → C contributes gamma^0, gamma^1, gamma^2 weight.
        hop_total = 0.0
        for rec in move_log:
            if rec["step"] < capture_step:
                continue
            if rec["player"] != player:
                continue
            if rec.get("from_planet_id") != pid:
                continue
            child_tid = rec.get("target_planet_id")
            if child_tid is not None:
                child_signal = hop_score(child_tid, rec["step"], depth + 1)
                # Hop-level discount: each link in the chain costs one gamma
                hop_total += (gamma ** (depth + 1)) * child_signal

        return direct + hop_total

    raw = hop_score(target_pid, launch_step, depth=0)
    return float(np.clip(raw, -1.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  MINOR NNUE TRAINING  (hop-chain, CPU)
# ══════════════════════════════════════════════════════════════════════════════
def train_minor_nnues(ow, move_log, history, player, total_steps):
    nnues = {"planet": ow.planet_nnue, "ship": ow.ship_nnue, "step": ow.step_nnue}
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
            feat = np.array(dec["feat"], dtype=np.float32)
            nnue.forward(feat)
            pred      = nnue.forward_scalar(feat)
            loss_grad = np.array([pred - signal], dtype=np.float32)
            nnue.backward(loss_grad)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NNUE TRAINING  (event-weighted, batched, optional GPU)
# ══════════════════════════════════════════════════════════════════════════════
def collect_main_samples(ow, move_log, player):
    """Return (feats, targets) arrays for main NNUE training."""
    feats, targets = [], []
    fc = ow.get_fleet_cache()
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
    return np.array(feats, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_main_nnue(gpu_nnue: TorchMainNNUE, ow,
                    move_log, player, batch_size=64):
    feats, targets = collect_main_samples(ow, move_log, player)
    if feats is None:
        return
    # Mini-batch SGD
    idx = np.random.permutation(len(feats))
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        gpu_nnue.train_batch(feats[batch_idx], targets[batch_idx])


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER  (runs in subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def _worker(worker_id: int, seeds: list, weights_path: str,
            result_queue: mp.Queue):
    """
    Run a batch of games with the given weight file.
    Push (move_log, history, n_steps, winner) for each game into result_queue.
    """
    ow = _load_agent_module()
    ow.load_weights(weights_path)

    for seed in seeds:
        reset_game_state(ow)

        env = make("orbit_wars", configuration={"seed": seed}, debug=False)

        def make_agent(pid):
            def _a(obs, config=None):
                obs_copy = dict(obs)
                obs_copy["player"] = pid
                return ow.agent(obs_copy, config)
            return _a

        agents = [make_agent(i) for i in range(4)]
        env.run(agents)
        final   = env.steps[-1]
        rewards = [final[i].reward for i in range(4)]
        winner  = int(np.argmax(rewards))
        n_steps = len(env.steps)

        result_queue.put({
            "seed":     seed,
            "move_log": copy.deepcopy(ow.get_move_log()),
            "history":  copy.deepcopy(ow.get_history()),
            "fc_dump":  ow.get_fleet_cache().dump(),
            "n_steps":  n_steps,
            "winner":   winner,
            "rewards":  rewards,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  TOURNAMENT  (compare two weight files)
# ══════════════════════════════════════════════════════════════════════════════
def run_tournament(current_path: str, best_path: str,
                   n_games: int = 20) -> float:
    """Return win-rate of *current* vs *best* over n_games."""
    wins = 0
    for g in range(n_games):
        ow_cur  = _load_agent_module()
        ow_best = _load_agent_module()
        ow_cur.load_weights(current_path)
        ow_best.load_weights(best_path)

        reset_game_state(ow_cur)
        reset_game_state(ow_best)

        env = make("orbit_wars", configuration={"seed": 10000 + g}, debug=False)

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

        # current = players 0,2  |  best = players 1,3
        agents = [make_cur(0), make_best(1), make_cur(2), make_best(3)]
        env.run(agents)
        final   = env.steps[-1]
        rewards = [final[i].reward for i in range(4)]
        cur_score  = rewards[0] + rewards[2]
        best_score = rewards[1] + rewards[3]
        if cur_score > best_score:
            wins += 1

    return wins / n_games


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
N_WORKERS          = 6
TOURNAMENT_EVERY   = 24  # games between tournaments
TOURNAMENT_GAMES   = 1   # games per tournament
WIN_THRESHOLD      = 0.5 # must exceed to promote
BEST_WEIGHTS       = "nnue_weights_best.pkl"
CURRENT_WEIGHTS    = "nnue_weights.pkl"


def train(n_games: int = 120, save_every: int = 12, verbose: bool = True):
    print(f"Starting parallel self-play: {n_games} games, {N_WORKERS} workers")
    if _TORCH:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        print(f"  GPU main NNUE: {device}")
    else:
        print("  GPU not available — main NNUE trains on CPU")

    # Load NNUEs into main process for training
    ow_main = _load_agent_module()
    ow_main.load_weights(CURRENT_WEIGHTS)
    gpu_nnue = TorchMainNNUE(ow_main.main_nnue)

    # Ensure best-weights file exists
    # if not os.path.exists(BEST_WEIGHTS):
    #     ow_main.save_weights(BEST_WEIGHTS)

    games_done = 0

    while games_done < n_games:
        # ── Distribute seeds across workers ─────────────────────────────────
        batch_size  = min(N_WORKERS, n_games - games_done)
        seeds       = list(range(games_done, games_done + batch_size))
        per_worker  = math.ceil(batch_size / N_WORKERS)
        worker_seed_batches = [
            seeds[i * per_worker:(i + 1) * per_worker]
            for i in range(N_WORKERS)
            if seeds[i * per_worker:(i + 1) * per_worker]
        ]

        result_q: mp.Queue = mp.Queue()
        processes = []
        for wid, wseeds in enumerate(worker_seed_batches):
            p = mp.Process(
                target=_worker,
                args=(wid, wseeds, CURRENT_WEIGHTS, result_q),
                daemon=True,
            )
            p.start()
            processes.append(p)

        # ── Collect results ──────────────────────────────────────────────────
        results = []
        for _ in range(batch_size):
            results.append(result_q.get())
        for p in processes:
            p.join(timeout=5)

        # ── Train on collected results ───────────────────────────────────────
        for res in results:
            move_log = res["move_log"]
            history  = res["history"]
            n_steps  = res["n_steps"]

            # Restore fleet-cache event data into ow_main for scoring
            ow_main.get_fleet_cache().load(res["fc_dump"])

            for player in range(4):
                train_minor_nnues(ow_main, move_log, history, player, n_steps)
                train_main_nnue(gpu_nnue, ow_main, move_log, player)

            games_done += 1
            f=open('games_done.txt','w')
            f.write(str(games_done))
            f.close()
            if verbose:
                print(f"  Game {games_done:4d} | seed={res['seed']} "
                      f"| winner=P{res['winner']} "
                      f"| steps={res['n_steps']}")

        # ── Sync GPU weights back to numpy NNUE ─────────────────────────────
        if _TORCH:
            gpu_nnue._sync_to_numpy()

        # ── Periodic weight save ─────────────────────────────────────────────
        if games_done % save_every == 0:
            ow_main.save_weights(CURRENT_WEIGHTS)
            print(f"  → Saved weights at game {games_done}")

        # ── Checkpoint tournament ────────────────────────────────────────────
        # if games_done % TOURNAMENT_EVERY == 0:
        #     ow_main.save_weights(CURRENT_WEIGHTS)
        #     print(f"\n  → Tournament at game {games_done} ...", end=" ", flush=True)
        #     win_rate = run_tournament(CURRENT_WEIGHTS, BEST_WEIGHTS,
        #                               n_games=TOURNAMENT_GAMES)
        #     print(f"win-rate={win_rate:.2f}", end="  ")
        #     if win_rate >= WIN_THRESHOLD:
        #         ow_main.save_weights(BEST_WEIGHTS)
        #         print("PROMOTED ✓")
        #         f=open('promotion_status.txt','w')
        #         if win_rate>0.5:
        #             f.write("PROMOTED ✓")
        #         else:
        #             f.write('NOT PROMOTED')
        #         f.close()
                
        #     else:
        #         # Revert: load best weights back into training context
        #         ow_main.load_weights(BEST_WEIGHTS)
        #         if _TORCH:
        #             gpu_nnue._sync_from_numpy()
        #         print("kept previous best")
        #     print()

    # Final save
    ow_main.save_weights(CURRENT_WEIGHTS)
    # if not os.path.exists(BEST_WEIGHTS):
    #     ow_main.save_weights(BEST_WEIGHTS)
    # else:
    #     wr = run_tournament(CURRENT_WEIGHTS, BEST_WEIGHTS,
    #                          n_games=TOURNAMENT_GAMES)
    #     if wr >= WIN_THRESHOLD:
    #         ow_main.save_weights(BEST_WEIGHTS)
    #         print("Final weights promoted to best.")

    print("\nTraining complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    train(n_games=n, verbose=True)
