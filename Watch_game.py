"""
watch_game.py — Run one game and save replay as HTML.
Open the output HTML file in any browser to watch the replay.

Usage: python watch_game.py [seed]
"""
import kaggle_environments

import json, importlib.util, kaggle_environments

# Load the orbit_wars module
def _load_env_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_env_dir = r"P:\Kaggle Bots\ORBIT WARS\orbit_wars_env"
_ow_env  = _load_env_module(f"{_env_dir}\\orbit_wars.py", "orbit_wars_env")

with open(f"{_env_dir}\\orbit_wars.json") as f:
    _spec = json.load(f)

kaggle_environments.register("orbit_wars", {
    "specification":  _spec,
    "interpreter":    _ow_env.interpreter,
    "renderer":       _ow_env.renderer,
    "html_renderer":  _ow_env.html_renderer,
})
import sys, os, importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Load your bot ──
AGENT_PATH = os.path.join(os.path.dirname(__file__), "Orbit_wars.py")
ow = load_module(AGENT_PATH, "orbit_wars")

# def reset_game_state():
#     ow._current_step     = 0
#     ow._pending_launches = {}
#     ow._events           = []
#     ow._done_events      = []
#     ow._fleet_cache      = {}
#     ow._history          = {}
#     ow.reset_move_log()
#     ow.reset_history()
#     for f in [ow.FLEET_CACHE, ow.EVENTS_FILE]:
#         if os.path.exists(f): os.remove(f)

def run_and_save(seed=None, out_file="replay.html"):
    from kaggle_environments import make

    #reset_game_state()
    cfg = {"seed": seed} if seed is not None else {}
    env = make("orbit_wars", configuration=cfg, debug=False)

    def make_agent(pid):
        def _agent(obs, config=None):
            obs_copy = dict(obs)
            obs_copy["player"] = pid
            return ow.agent(obs_copy, config)
        return _agent

    agents = [make_agent(i) for i in range(4)]

    print(f"Running game (seed={seed})...")
    env.run(agents)

    final   = env.steps[-1]
    rewards = [final[i].reward for i in range(4)]
    winner  = int(max(range(4), key=lambda i: rewards[i]))
    print(f"Done! Steps={len(env.steps)} | Rewards={[round(r,1) for r in rewards]} | Winner=P{winner}")

    # Save HTML replay
    html = env.render(mode="html")
    with open(out_file, "w") as f:
        f.write(html)
    print(f"\nReplay saved to: {os.path.abspath(out_file)}")
    print("Open that file in your browser to watch!")

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_and_save(seed=seed)