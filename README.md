# Orbit Wars Bot

Orbit Wars Bot is a hybrid AI agent developed for the Kaggle Orbit Wars competition. It combines specialized neural networks with classical search algorithms to make strategic decisions under a strict real-time budget.

The bot uses four Neural Network Unified Evaluators (NNUEs): three lightweight networks for candidate generation and one larger network for board evaluation. The minor NNUEs independently learn target planet selection, fleet size estimation, and launch timing for orbiting planets, allowing the search algorithm to focus on a small set of high-quality candidate moves.

Before search, every candidate move passes through a geometry-based validation pipeline that removes impossible or strategically invalid actions, including sun-crossing trajectories, missed orbital interceptions, out-of-bounds launches, and illegal fleet sizes.

Depending on the number of players, the bot dynamically selects its search algorithm. Two-player games use Alpha-Beta search, while multiplayer games use Monte Carlo Tree Search (MCTS). Both search methods rely on the Main NNUE to evaluate resulting board positions.

To improve strategic awareness, the bot maintains an internal fleet cache that infers enemy fleet destinations, tracks friendly launches, estimates fleet pressure, and records fleet events used during training.

Training is performed entirely through parallel self-play. Six worker processes generate games simultaneously, producing training data that is used to optimize the minor NNUEs through a custom hop-chain reward signal and the Main NNUE through event-weighted board evaluations. GPU acceleration is optionally used for batched training of the Main NNUE.

### Workflow

## Workflow

```text
                  Game Observation
                          │
                          ▼
                Parse Planets & Fleets
                          │
                          ▼
               Predict Future Positions
                          │
                          ▼
      Generate Candidate Actions (Using 3 Minor NNUEs)
                          │
                          ▼
              Filter Invalid/Unsafe Moves
            (Missed intercepts, bad targets)
                          │
                          ▼                          
      Alpha-Beta Search Engine(2 Player) / MCTS (4 Player)
                          │
                          ▼
               NNUE Position Evaluation
                          │
                          ▼
                   Best Move Selection
                          │
                          ▼
                     Submit Action
                          │
                          ▼
                   Repeat Next Tick
```

### Key Features

* Hybrid neural-search architecture.
* Four specialized NNUEs with distinct responsibilities.
* Alpha-Beta and MCTS hybrid search.
* Enemy fleet target inference.
* Geometry-aware move validation.
* Orbit interception prediction.
* Fleet pressure estimation.
* Parallel self-play training.
* Custom hop-chain reward propagation.
* Optional GPU-accelerated training.
* Designed for real-time competitive gameplay under a sub-second move budget.
