# eval

Everything after training. Run from the repo root.

| script | what it does | output |
|---|---|---|
| `benchmark.py` | all replanning strategies x all push sizes, 300 episodes each (~2 h on 2 cores, resumable) | `results/benchmark.json`, `results/episodes/` |
| `detection.py` | how many pushes each detector notices at 1% false alarms (~5 min) | `results/detection.json` |
| `eval.ipynb` | all plots + the paired-difference table, in the order below | `figs/`, `results/tables.md` |

The notebook goes through the questions in the order I looked at them:

| | question | figure |
|---|---|---|
| 1 | are the demos ok, does the policy train | `01_demos_and_training.png` |
| 2 | without pushes, how often should the policy replan (which baseline is fair) | `02_horizon_no_pushes.png` |
| 3 | how much do pushes hurt | `03_pushes_hurt.png` |
| 4 | does the world model notice the pushes | `04_detection.png` |
| 5 | does replanning on surprise help, more than replanning at random / on a simple jump detector | `05_triggers.png`, `06_success_vs_compute.png`, `results/tables.md` |
| 6 | why the effect is not bigger | `07_after_push.png` |
| 7 | the observation history bug | `08_history_fix.png` |
| 8 | one episode | `09_episode.png` |
