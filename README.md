# COMP-1945-Demo
# Energy Management System — GUI v8 Documentation

**File:** `energy_management_gui.py`  
**Last modified:** 2026-04-28  
**Lines:** 1110  
**Run:** `python energy_management_gui.py`

---

## Overview

A single-file Python desktop application for energy ESG compliance monitoring and optimization. It wraps a v2 ML/RL engine in a dark-themed Tkinter GUI with four pages:

```
Startup → Inputs + Thresholds → Summary → Optimize
```

The app trains four models on first run (~15 min), caches them to `models/`, and loads instantly on subsequent runs.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | CSV loading |
| `numpy` | Numerical ops |
| `scikit-learn` | MLR, MLP, StandardScaler |
| `joblib` | Model serialization |
| `torch` | RNN/LSTM |
| `gymnasium` | RL environment |
| `stable-baselines3[extra]` | PPO agent |
| `tkinter` | GUI (bundled with Python) |

Install missing deps:
```bash
pip install pandas numpy scikit-learn joblib torch gymnasium stable-baselines3[extra]
```

---

## File Layout

```
<script directory>/
├── energy_management_gui.py   ← this file
├── 1945dataset.csv            ← required, must be in same folder
└── models/                    ← auto-created on first run
    ├── mlr_model.joblib
    ├── mlp_model.joblib
    ├── rnn_model.pt
    ├── rnn_scaler.joblib
    ├── scaler.joblib
    ├── rl_agent.zip
    └── meta.pkl
```

---

## ML Models

### MLR — Multi-output Linear Regression
- **Library:** `sklearn.linear_model.LinearRegression`
- **Inputs:** All feature columns (scaled)
- **Outputs:** CO₂ emission (30d), energy cost (30d), energy efficiency
- **Saved to:** `models/mlr_model.joblib`

### MLP — Fault Classifier
- **Library:** `sklearn.neural_network.MLPClassifier`
- **Architecture:** 64 → 32 hidden layers
- **Inputs:** All feature columns (scaled)
- **Output:** Binary fault label (0 = normal, 1 = fault) + probability
- **Saved to:** `models/mlp_model.joblib`

### RNN/LSTM — Sequence Regression
- **Library:** PyTorch `nn.LSTM`
- **Architecture:** 2-layer LSTM (hidden=64) → FC(64→32→3)
- **Inputs:** All feature columns (scaled), treated as a sequence
- **Outputs:** CO₂ emission, energy cost, energy efficiency
- **Training:** 30 epochs, batch=64, Adam lr=1e-3
- **Saved to:** `models/rnn_model.pt` (weights only) + `models/rnn_scaler.joblib` (target scaler)
- **Note:** `weights_only=True` required for PyTorch 2.6+

### PPO — RL Optimizer
- **Library:** `stable_baselines3.PPO`
- **Policy:** MlpPolicy
- **Observation space:** Full scaled feature vector
- **Action space:** Deltas for controllable features only (bounded to [-1, 1])
- **Training:** 20,000 timesteps, n_steps=256, batch=64, epochs=5, lr=3e-4
- **Saved to:** `models/rl_agent.zip`

---

## Column Detection (Automatic, Keyword-Based)

The app auto-detects target and feature columns from the CSV header using keyword matching.

| Role | Keywords |
|---|---|
| CO₂ target | `co2`, `carbon`, `emission`, `co₂` |
| Cost target | `cost`, `bill`, `energy_cost`, `electricity_cost`, `price` |
| Efficiency target | `efficiency`, `eff`, `energy_eff`, `power_eff` |
| Fault target | `fault`, `failure`, `error`, `anomaly`, `defect`, `alarm` |
| Excluded entirely | `id`, `timestamp`, `date`, `time`, `index`, `unnamed` |
| Read-only (not in RL action space) | `date`, `time`, `day`, `month`, `year`, `hour`, `week`, `temperature`, `temp`, `ambient`, `weather`, `humidity`, `outdoor`, `season`, `solar_irradiance`, `irradiance`, `efficiency`, `eff`, `cost`, `bill`, `price`, `co2`, `carbon`, `emission`, `fault`, `failure`, `error`, `demand`, `load_demand`, `grid_frequency`, `frequency` |

**Controllable columns** = feature columns that do NOT match any read-only keyword. These are the variables the RL agent can adjust.

When multiple columns match the same target keyword, the one with the highest standard deviation is selected (avoids picking normalised proxy columns).

---

## ESG Thresholds

User-defined at runtime on the Inputs page. Defaults:

| Threshold | Default | Direction |
|---|---|---|
| Max CO₂ (30d) | 400.0 | must be ≤ |
| Max Energy Cost (30d) | 1000.0 | must be ≤ |
| Min Energy Efficiency | 0.5 | must be ≥ |

These thresholds are passed to both the Summary page (violation detection) and the RL reward function (optimization target).

---

## Variable Constraints (Per-Variable Min/Max)

On the **Optimize page**, each controllable variable has two editable fields:

| Field | Default | Meaning |
|---|---|---|
| **Min** | `0` | Lower bound the RL agent cannot push the variable below |
| **Max** | *(empty = ∞)* | Upper bound the RL agent cannot push the variable above |

### How constraints are applied

1. **During RL environment steps** (`EnergyEnv.step`): after each action delta is applied, the value is clipped to `[lo, hi]` using `np.clip`.
2. **After RL rollout** (`RLOptimizer.suggest`): a final clip pass enforces bounds on all controllable indices. Variables with no explicit bound default to a floor of `0`.
3. **Swap guard:** if the user enters Min > Max, the values are silently swapped before use.

### Code location

```
EnergyEnv.__init__   line 285  — var_bounds stored as {feature_index: (lo, hi)}
EnergyEnv.step       line 303  — np.clip applied per step
RLOptimizer.suggest  line 353  — final clip after rollout
OptimizePage._run    line 890  — reads UI fields, builds var_bounds dict, passes to RLOptimizer
```

### Example

To constrain `solar_panel_output` between 50 and 200:
- Enter `50` in the Min field
- Enter `200` in the Max field
- Leave the checkbox checked
- Click **▶ Run Optimizer**

---

## GUI Pages

### 1. Startup Page (`StartupPage`)

- Displays app title, progress bar, and status messages
- Runs a background thread (`_run`) that loads or trains all models
- Shows a **"Go to Inputs →"** button once models are ready (progress = 100%)
- Calls `App.on_ready(data, sup, rnn, rl)` when done, which navigates to Inputs

### 2. Inputs + Thresholds Page (`InputPage`)

- **Left panel:** Scrollable list of all feature columns with entry fields
  - Green `[controllable]` tag = RL can adjust this variable
  - Grey `[read-only]` tag = excluded from RL action space
- **Right panel:** ESG threshold entry fields (Max CO₂, Max Cost, Min Efficiency)
- **⟳ Random Sample** button: fills all fields from a random row in the test set
- **Run Prediction →** button: reads all values, validates, stores in `App.current_inputs` and `App.thresholds`, navigates to Summary

### 3. Summary Page (`SummaryPage`)

- Runs `SupervisedEngine.predict_state` (MLR + MLP) and `RNNEngine.predict` on the current inputs
- Displays a side-by-side table: Metric | MLR value | RNN value | Status
- ESG status panel on the right: green "✓ ESG COMPLIANT" or red "⚠ ESG VIOLATION"
- Lists active thresholds and any violations
- If violations exist: **Run RL Optimizer →** button appears

### 4. Optimize Page (`OptimizePage`)

- **Left panel:** Checkbox list of all controllable variables
  - Each row has **Min** and **Max** entry fields for per-variable constraints
  - **All / None** buttons to bulk-select
  - **▶ Run Optimizer** button
- **Right panel:** Results after optimization
  - Suggested adjustments table: Variable | Before | After | Δ
  - Projected outcome table: Metric | Before | After MLR | After RNN | Status
  - Final ESG compliance verdict

---

## RL Reward Function

```
reward = 0
if CO₂ > max_co2:   reward -= 5 × (CO₂ - max_co2) / max_co2
if eff < min_eff:    reward -= 5 × (min_eff - eff) / min_eff
if cost > max_cost:  reward -= 2 × (cost - max_cost) / max_cost
reward -= 0.1 × ‖action‖₂          # penalise large adjustments
if CO₂ ≤ max_co2 and eff ≥ min_eff: reward += 2   # compliance bonus
```

---

## Caching Logic

```python
models_cached() → True if all 6 model files exist
```

On first run: trains MLR → MLP → RNN → PPO, saves all files.  
On subsequent runs: loads from `models/` instantly.

To force retrain: delete the `models/` directory.

The RL agent is also retrained automatically on the Optimize page if the saved agent's action space dimension doesn't match the currently selected variables.

---

## Architecture Notes

### Threading
All model training and loading runs in a `daemon=True` background thread. The UI polls a `queue.Queue` every 120ms (startup) or 150ms (optimizer) via `self.after()`.

### PyTorch Save/Load (PyTorch 2.6+ fix)
The RNN checkpoint stores only `state_dict` + `n_features`. The `StandardScaler` for targets is saved separately via joblib. `torch.load` is called with `weights_only=True` to avoid the default-changed warning/error in PyTorch 2.6+.

### macOS Compatibility
- `IS_MAC = platform.system() == "Darwin"` — used to select font family (`Helvetica` vs `Segoe UI`)
- Mouse wheel scroll uses `e.delta` directly on macOS, divided by 120 on Windows/Linux
- `ttk.Style.theme_use("clam")` is wrapped in a try/except to avoid macOS rendering issues

### Scrollable Frames
`_scrollable(parent)` returns `(canvas, inner_frame)`. The inner frame expands to canvas width. Scroll events are bound to the canvas only (not `bind_all`) to avoid event leakage.

---

## Color Palette

| Name | Hex | Usage |
|---|---|---|
| BG | `#0f1117` | Window background |
| SURFACE | `#1a1d27` | Nav bar, entry fields |
| CARD | `#22263a` | Panel backgrounds |
| BORDER | `#2e3250` | Separators |
| ACCENT | `#4f8ef7` | RNN values, highlights |
| ACCENT2 | `#7c5cbf` | Random Sample button |
| GREEN | `#3ecf8e` | OK status, controllable tag |
| RED | `#f05252` | Violations, faults |
| YELLOW | `#f5a623` | Warnings, section headers |
| TEXT | `#e8eaf0` | Primary text |
| SUBTEXT | `#8b90a8` | Secondary text |
| WHITE | `#ffffff` | Button text, after-values |

---

## Known Limitations

- The RL agent optimizes for ESG compliance, not for a globally optimal solution. Results depend on training timesteps (`RL_TIMESTEPS = 1_000_000`). Increase this constant for better optimization at the cost of longer training time.
- MLR can produce negative predictions for physically bounded quantities (e.g. efficiency). The Optimize page falls back to the RNN value for compliance checking when MLR output is negative.
- The app requires `1945dataset.csv` in the same directory. It will show an error dialog and exit if the file is missing.
- Model cache is not invalidated when the dataset changes. Delete `models/` manually after replacing the CSV.
