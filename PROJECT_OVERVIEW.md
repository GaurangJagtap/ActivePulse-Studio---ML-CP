# ActivePulse: Data Labeling & Active Learning Platform for Reducing Annotation Cost
### Master Technical Specification & Architecture Overview

---

## 1. Executive Summary & Academic Project Framing

- **Project Title**: Data Labeling & Active Learning Platform for Reducing Annotation Cost
- **Platform Name**: ActivePulse Studio
- **Problem Statement (PS)**: In real-world machine learning deployments, raw unlabeled data is cheap and abundant, but expert human annotation is prohibitively expensive ($15–$50/hour), slow, and bottlenecked by human bandwidth. Crucially, standard random data labeling wastes over 70% of annotator effort on easy, obvious, or near-identical samples that contribute virtually zero new learning signal to the model.
- **Objective**: Design, implement, and benchmark an end-to-end Active Learning Platform combining PyTorch, skorch, and modAL/Label Studio that algorithmically prioritizes the most informative, uncertain, and diverse samples for human annotation—closing the feedback loop between model uncertainty, human labeling, and continuous retraining.
- **Tech Stack**: Python 3.11+, PyTorch, skorch, modAL-python, scikit-learn, SQLite / SQLAlchemy, FastAPI, React, TypeScript, Vite, Tailwind CSS, Recharts, Plotly.js, pytest.
- **Expected Outcome**:
  1. Achieve target model accuracy with **30% to 50% fewer human-labeled samples** compared to standard uniform random selection.
  2. Prove higher Area Under the Learning Curve (ALC) and lower Expected Calibration Error (ECE) across multi-seed evaluations with 95% Confidence Intervals.
  3. Guarantee complete zero data leakage between the fixed holdout evaluation test set and the active sampling pool.
  4. Deliver a production-quality, explainable platform with 7 active query algorithms, 2D latent embedding projection, and an interactive annotation cockpit.
- **Current Milestone**: **50% Milestone Complete** (Phases 1 & 2 fully operational, verified with 18/18 passing tests, multi-seed benchmarked, and live in the ActivePulse Studio UI).

---

## 2. Theoretical Motivation: Why Active Learning?

### The Data Annotation Dilemma
In traditional supervised learning (Passive Learning), the dataset $\mathcal{D}$ is labeled uniformly at random:
$$\mathcal{D}_{\text{labeled}} \sim p(x)$$
However, real-world data distributions have extreme density variations: thousands of samples cluster tightly around the same prototypical patterns. Labeling 10,000 images of clear daylight highway driving yields almost no new gradient signal to an autonomous vehicle model.

### The Active Learning Solution
Active Learning inverts the control flow by allowing the machine learning model itself to query the most informative samples from an unlabeled pool $\mathcal{U}$:
$$x^* = \arg\max_{x \in \mathcal{U}} \text{Utility}(x; \theta)$$
Where $\theta$ represents the current model parameters. By querying only instances where the model's epistemic uncertainty is high or where data space coverage is sparse, the sample complexity drops exponentially, reaching target accuracy with a fraction of the manual labels.

---

## 3. High-Level System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               ORCHESTRATOR & ROUND LOOP                │
                    │   Round loop · Stopping rules · State DB · Resumability│
                    └───────────┬────────────────┬────────────────┬──────────┘
                                │                │                │
        ┌───────────────────────▼──┐   ┌─────────▼────────┐   ┌───▼────────────────┐
        │       DATA LAYER         │   │   MODEL LAYER    │   │   LABELING LAYER   │
        │ - Unlabeled pool (70%)   │   │ - PyTorch CNN    │   │ - Simulated Oracle │
        │ - Labeled set (growing)  │──▶│ - skorch wrapper │   │   (noise injection)│
        │ - Fixed test set (20%)   │   │ - Temp. Scaler   │   │ - Label Studio SDK │
        │ - Validation set (10%)   │   │ - MC-Dropout/BALD│   │   (human in loop)  │
        │ - SQLite / PostgreSQL DB │   │ - 128-d Embeds   │   │ - Cohen's Kappa QA │
        └──────────────────────────┘   └─────────┬────────┘   └───▲────────────────┘
                                                 │ probs, features│ labels
                                       ┌─────────▼────────────┐   │
                                       │     QUERY ENGINE     │───┘  Batch of K
                                       │ - Random (baseline)  │      most valuable
                                       │ - Least Conf / Margin│      samples
                                       │ - Shannon Entropy    │
                                       │ - BALD (MC-Dropout)  │
                                       │ - Core-Set (k-Center)│
                                       │ - BADGE (Gradients)  │
                                       │ - Hybrid (Unc + Div) │
                                       └─────────┬────────────┘
                                                 │
                                 ┌───────────────▼────────────────┐
                                 │     EVALUATION & TRACKING      │
                                 │ - Holdout Test Accuracy & F1   │
                                 │ - Expected Calibration Error   │
                                 │ - Area Under Curve (ALC)       │
                                 │ - % Label Reduction vs Random  │
                                 │ - Multi-seed 95% CI curves     │
                                 └───────────────┬────────────────┘
                                                 │
                                 ┌───────────────▼────────────────┐
                                 │       ACTIVEPULSE STUDIO       │
                                 │ - Mission Control & 2D PCA Map │
                                 │ - Annotation & Query Studio    │
                                 │ - Strategy Benchmark Arena     │
                                 └────────────────────────────────┘
```

---

## 4. The Data Layer & Zero Data Leakage Guarantee

### The 4-Way Partition Scheme
Before Round 0 begins, the raw dataset is deterministically partitioned using a fixed random seed into 4 disjoint sets:
1. **Fixed Holdout Test Set (20%)**: Stored in SQLite with `split='test'`. It is permanently locked and strictly forbidden from entering the training set or query engine. All reported accuracies are 100% out-of-sample.
2. **Validation Calibration Set (10%)**: Stored with `split='val'`. Used exclusively to calibrate model logits via Temperature Scaling so uncertainty rankings are truthful.
3. **Unlabeled Pool (70%)**: Stored with `split='pool', status='unlabeled'`. This is the candidate pool scored by active learning algorithms.
4. **Labeled Training Set**: Starts with a small cold-start batch (e.g. 30 samples). Each round, newly labeled instances transition to `status='labeled'` and the model retrains.

### SQLite Database State Machine
Implemented in [`src/data/db.py`](file:///d:/SEM-5/ML%20CP/src/data/db.py) via SQLAlchemy:
- **`samples` Table**: Tracks `sample_id`, `split`, `status` (`unlabeled` $\rightarrow$ `queued` $\rightarrow$ `labeled`), `label`, `true_label`, `round_added`, `annotator`, `created_at`, `updated_at`.
- **`rounds` Table**: Tracks `round_num`, `strategy`, `n_labeled`, `n_pool`, `test_acc`, `test_f1`, `ece`, `temperature`, `timestamp`.
- **Resumability**: If a run is interrupted, the platform queries completed rounds and resumes from the exact state.

---

## 5. The Model Layer & Probability Calibration

### PyTorch Architectures
1. **Image Vision (`CIFARConvNet`)**:
   - 4-layer Convolutional Neural Network with BatchNorm, ReLU, MaxPool, and spatial dropout ($p=0.3$).
   - Computes forward passes for classification and provides a `forward_features()` method exposing a 128-dimensional bottleneck embedding.
2. **Text Classification (`TextMLP`)**:
   - Dense MLP with BatchNorm and dropout processing dense text embeddings for 4-class classification.
3. **Transfer Learning (`ResNet18TransferNet`)**:
   - Pretrained ResNet-18 backbone with custom classification head and dropout.

### The `skorch` modAL Bridge
modAL expects estimators following scikit-learn conventions (`fit(X, y)`, `predict_proba(X)`). PyTorch models natively do not adhere to this. Our [`ActiveLearningClassifier`](file:///d:/SEM-5/ML%20CP/src/models/wrapper.py) wraps PyTorch through `skorch.NeuralNetClassifier`, adding:
- `extract_embeddings(X)`: Batch extraction of 128-d feature representations.
- `mc_dropout_predict(X, n_passes=T)`: Stochastic forward passes with dropout active to sample $T$ posterior probability vectors for BALD.
- `calibrate(X_val, y_val)`: Fits temperature parameter $T$ on validation logits.
- `warm_start` control: Resets weights each round by default (`warm_start: false`) to prevent catastrophic forgetting and parameter drift.

### Temperature Scaling Calibration
Modern deep neural networks produce overconfident probabilities due to unregularized cross-entropy optimization. An overconfident model distorts uncertainty rankings.
Following Guo et al. (2017), we learn a single scalar $T > 0$ on validation logits $z_i$:
$$P(Y = c \mid x_i) = \frac{\exp(z_{i, c} / T)}{\sum_{j=1}^C \exp(z_{i, j} / T)}$$
$T$ is optimized using L-BFGS to minimize negative log-likelihood on validation logits.
We measure calibration quality using **Expected Calibration Error (ECE)** across $M=15$ confidence bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 6. The Query Engine: Mathematical Formulations of All 7 Strategies

### 1. Uniform Random Sampling (Baseline Control)
Picks $K$ samples uniformly at random:
$$x \sim \text{Uniform}(\mathcal{U})$$

### 2. Least Confidence Sampling
Selects samples whose most likely class has the lowest confidence:
$$\phi_{\text{LC}}(x) = 1 - \max_{c} P(y=c \mid x)$$

### 3. Margin Sampling
Selects samples where the gap between the top-1 and top-2 predicted classes is smallest (samples nearest the decision boundary):
$$\phi_{\text{Margin}}(x) = P(y_{(1)} \mid x) - P(y_{(2)} \mid x)$$
The query engine minimizes this margin.

### 4. Shannon Predictive Entropy
Selects samples with the highest information entropy across all classes:
$$\phi_{\text{Entropy}}(x) = -\sum_{c=1}^C P(y=c \mid x) \log_2 P(y=c \mid x)$$

### 5. BALD (Bayesian Active Learning by Disagreement)
Measures mutual information between model predictions and model parameters $\omega$ using Monte Carlo Dropout:
$$\mathbb{I}[y, \omega \mid x, \mathcal{D}] = \mathbb{H}\left[ \frac{1}{T}\sum_{t=1}^T p(y \mid x, \hat{\omega}_t) \right] - \frac{1}{T}\sum_{t=1}^T \mathbb{H}[p(y \mid x, \hat{\omega}_t)]$$
High BALD score indicates high epistemic uncertainty (different sub-networks strongly disagree).

### 6. Core-Set Selection (k-Center Greedy)
Maximizes geometric coverage over the bottleneck feature embedding space $\mathcal{Z}$:
$$x^* = \arg\max_{i \in \mathcal{U}} \min_{j \in \mathcal{L} \cup \mathcal{S}} \|z_i - z_j\|_2$$
Where $\mathcal{L}$ is the labeled set and $\mathcal{S}$ is the batch currently being constructed. Prevents blind spots in unexplored data regions.

### 7. BADGE (Batch Active Learning by Diverse Gradient Embeddings)
Ash et al. (ICLR 2020). Constructs hypothetical gradient embeddings with respect to the output layer:
$$g_{x, j} = (P(y=j \mid x) - \mathbb{I}[j = \hat{y}]) \cdot z(x)$$
The magnitude $\|g_x\|$ captures uncertainty; the direction captures sample diversity. BADGE selects $K$ samples using K-Means++ initialization in gradient embedding space.

### 8. Hybrid Strategy (State-of-the-Art Batch Selection)
Solves the fundamental flaw of uncertainty sampling in batch mode (selecting $K$ near-identical ambiguous points):
1. **Uncertainty Filter**: Identifies top-$3K$ samples with highest predictive entropy.
2. **Diversity Clustering**: Runs K-Means clustering on the bottleneck embeddings of those $3K$ candidates into $K$ clusters.
3. **Medoid Selection**: Queries the sample closest to each cluster center.

---

## 7. The ActivePulse Studio Web Interface (`frontend/` + `backend/`)

The platform now uses a React + TypeScript + FastAPI architecture (replacing the legacy Streamlit prototype):

1. **Overview Dashboard**: Holdout accuracy, Macro-F1, labeled vs. unlabeled counts, budget utilization, cost savings, learning curve, pipeline tracker, and recent run activity.
2. **Dataset Explorer**: CIFAR-10 and AG News overview, deterministic/DB split breakdown, class distribution chart, and sample previews.
3. **Active Learning Lab**: Full experiment configuration (dataset, model, strategy, mode, budget, seed), live polling progress, interactive Plotly.js 2D PCA projection, Recharts learning curve, and per-sample uncertainty diagnostics.
4. **Annotation Studio**: Dedicated blind human-in-the-loop labeling workspace (ground truth hidden prior to submission), model suggestions/probabilities, round retraining trigger, annotation audit log, and Phase 3 Label Studio webhook readiness.
5. **Benchmark Arena**: Multi-seed strategy comparison vs. Random baseline, ALC, Label Reduction %, ECE, interactive comparison charts, and CSV export.
6. **Experiment History**: Centralized run ledger, side-by-side experiment comparison, detail inspector, and CSV export.

---

## 8. Evaluation & Benchmark Mathematics

### Area Under the Learning Curve (ALC)
Computed via normalized trapezoidal numerical integration:
$$\text{ALC} = \frac{1}{N_{\text{max}} - N_{\text{min}}} \int_{N_{\text{min}}}^{N_{\text{max}}} \text{Acc}(n) \, dn$$
Higher ALC indicates faster learning with fewer labels.

### Labels to Target Accuracy
Interpolates the minimum number of human labels required to reach a specific accuracy threshold $\tau$:
$$N^* = N_i + \frac{\tau - \text{Acc}_i}{\text{Acc}_{i+1} - \text{Acc}_i} (N_{i+1} - N_i)$$

### Percentage Label Reduction vs Random Baseline
$$\text{Reduction \%} = \frac{N^*_{\text{random}} - N^*_{\text{strategy}}}{N^*_{\text{random}}} \times 100\%$$

---

## 9. Complete Roadmap: 50% Milestone vs Remaining 50%

```
[CURRENT STATUS: 50% COMPLETE - VERIFIED & OPERATIONAL]
Core algorithmic engine, PyTorch models, 7 query strategies, benchmark runner, 18 automated tests, and UI cockpit.

                    ▼ REMAINING 50% IMPLEMENTATION ROADMAP ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Human-in-the-Loop with Label Studio (~20%)                         │
│ • Label Studio SDK: Connect via Python API to automatically spin up labeling│
│   projects and push query batches.                                          │
│ • Model Pre-annotations: Automatically insert model predictions into Label  │
│   Studio tasks as draft labels (speeds up human annotators by 2-3x).        │
│ • Webhook Ingestion: A FastAPI webhook receiver listening for finished      │
│   annotations (`ANNOTATION_CREATED`) to trigger the next retraining round.  │
│ • Quality Control: Double-annotate 10%–15% of samples and compute Cohen's   │
│   Kappa agreement scores to flag unreliable human annotators.               │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: Production Microservices & Experiment Tracking (~15%)              │
│ • FastAPI REST API: Endpoints for `/rounds/start`, `/status`, and           │
│   `/webhook/label-studio`.                                                  │
│ • MLflow Server: Enterprise experiment tracking recording parameters,      │
│   learning curves, and PyTorch model artifacts per seed.                    │
│ • Docker Compose: Multi-container stack bundling PostgreSQL, Label Studio,  │
│   MLflow, FastAPI, and Streamlit.                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 5: Rigorous 5-Seed Benchmarking & Final Documentation (~15%)          │
│ • Run full 5-seed benchmark comparing all 7 strategies against random.      │
│ • Generate final publication-grade CSV and Markdown label savings report.   │
│ • Complete production README and deployment documentation.                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Repository File Directory Guide

```
al-platform/
├── configs/                            # Experiment configurations
│   ├── cifar10_simulation.yaml         # CIFAR-10 configuration settings
│   └── agnews_simulation.yaml          # AG News configuration settings
├── src/
│   ├── data/                           # Data Layer
│   │   ├── base.py                     # DatasetAdapter abstract class
│   │   ├── db.py                       # SQLAlchemy models & DatabaseManager
│   │   ├── pool.py                     # PoolManager (splits, pool, labeled sets)
│   │   ├── cifar10.py                  # CIFAR-10 image adapter
│   │   └── agnews.py                   # AG News text adapter
│   ├── models/                         # Model Layer
│   │   ├── cnn.py                      # CIFARConvNet & ResNet18TransferNet
│   │   ├── text_net.py                 # TextMLP representation network
│   │   ├── calibration.py              # TemperatureScaler & compute_ece
│   │   └── wrapper.py                  # ActiveLearningClassifier (skorch + modAL)
│   ├── strategies/                     # Query Engine
│   │   ├── base.py                     # BaseStrategy & modAL adapter
│   │   ├── random.py                   # Random uniform sampling baseline
│   │   ├── uncertainty.py              # Least Confidence, Margin, Entropy
│   │   ├── bald.py                     # BALD (MC-Dropout Mutual Information)
│   │   ├── coreset.py                  # Core-Set (k-Center Greedy on embeddings)
│   │   ├── badge.py                    # BADGE (Gradient Embeddings + K-Means++)
│   │   ├── hybrid.py                   # Hybrid (Top-N Uncertainty + K-Means)
│   │   ├── cold_start.py               # Cold-Start K-Means Medoids
│   │   └── registry.py                 # Strategy Registry
│   ├── labeling/                       # Labeling Layer
│   │   ├── base.py                     # BaseLabeler abstract class
│   │   └── oracle.py                   # SimulatedOracle with noise injection
│   ├── loop/                           # Active Learning Loop
│   │   ├── state.py                    # LoopConfig dataclass
│   │   ├── stopping.py                 # StoppingRule (acc, budget, plateau, rounds)
│   │   └── orchestrator.py             # ActiveLearningOrchestrator
│   └── eval/                           # Evaluation & Benchmark
│       ├── metrics.py                  # Accuracy, Macro-F1, ECE, NLL
│       ├── learning_curve.py           # LearningCurveTracker
│       └── savings.py                  # ALC, labels-to-target, % label reduction
├── backend/
│   └── api/
│       ├── main.py                     # FastAPI application entrypoint & CORS
│       ├── config.py                   # Environment-driven API configuration
│       ├── routes/                     # REST endpoints (health, datasets, experiments, annotations, benchmarks)
│       ├── schemas/                    # Pydantic request/response contracts
│       └── services/                   # Bridges FastAPI to src/ ML & SQLite modules
├── frontend/
│   ├── package.json                    # React, TypeScript, Vite, Tailwind, Recharts, Plotly.js
│   └── src/
│       ├── charts/                     # Recharts curves/bars & Plotly 2D PCA projection
│       ├── components/                 # Reusable UI & shared metric/status components
│       ├── layouts/                    # Persistent dark navy sidebar & top bar
│       ├── pages/                      # 6 workspaces (Overview, Datasets, Lab, Annotation, Benchmarks, History)
│       ├── services/api.ts             # Typed FastAPI REST client
│       └── types/api.ts                # TypeScript interfaces
├── scripts/
│   ├── run_simulation.py               # CLI single-strategy simulation runner
│   └── run_benchmark.py                # Multi-strategy, multi-seed benchmark runner
├── tests/
│   ├── test_phase1.py                  # Data splits, state machine, model tests
│   ├── test_strategies.py              # Strategy selection, modAL adapters, savings tests
│   └── test_api.py                     # FastAPI endpoint & HITL workflow integration tests
├── results/                            # Benchmark artifacts
│   ├── benchmark_savings.csv           # Benchmark summary table (CSV)
│   ├── benchmark_savings.md            # Benchmark summary table (Markdown)
│   └── learning_curves.png             # Learning curves chart with 95% CI bands
├── README.md                           # Installation, startup & architecture guide
├── PROJECT_OVERVIEW.md                 # Master Technical Documentation
├── PRESENTATION_CHEAT_SHEET.md         # Viva defense script & glossary
└── MILESTONE_50_AND_REMAINING_ROADMAP.md # 50% milestone guide & 100% roadmap
```

---

## 11. How to Execute All Commands

### 1. Run the Complete Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/
```

### 2. Run the Multi-Seed Benchmark Runner
```powershell
.\.venv\Scripts\python scripts/run_benchmark.py --strategies random entropy hybrid --seeds 42 43 --rounds 3 --batch-size 15 --epochs 3 --synthetic --max-samples 500 --output-dir results
```

### 3. Launch ActivePulse Studio (Backend + Frontend)
```powershell
# Terminal 1: FastAPI Backend
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: React Frontend
cd frontend
npm run dev
```
Access the UI at **`http://localhost:5173`** and FastAPI Swagger docs at **`http://127.0.0.1:8000/docs`**.
