# AlphaAgents

AlphaAgents is an experiment harness for comparing single-agent and multi-agent equity research workflows. It freezes all market data before inference, runs
the same universe through several workflow and risk configurations, and writes portfolio metrics and decision-quality measures to reproducible artifacts.

The implementation accompanies a bachelor thesis based on Zhao et al., *AlphaAgents: Large Language Model Based Multi-Agents for Equity Portfolio Constructions*.

## Setup
Python 3.12 is recommended.
```bash
git clone https://github.com/Ranelkin/AlphaAgents.git
cd AlphaAgents
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Create a `.env` file with the credentials used by your selected model provider.
SEC filing retrieval also requires `EMAIL`, which is passed to EDGAR as the
request identity.

## Running an experiment
First freeze the inputs for a universe and evaluation window:


```bash
python run_experiment.py freeze \
  --provider together \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --universe AAPL MSFT NVDA \
  --benchmark-universe AAPL MSFT NVDA
```

The command prints the manifest path. Reuse that manifest for every comparison:
```bash
python run_experiment.py run \
  --manifest experiments/output/<manifest_hash>/manifest.json \
  --provider together \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --workflow-mode debate \
  --risk-profile risk_neutral \
  --tooling-mode paper_like \
  --run-label baseline
```

To run a matrix of configurations:

```bash
python run_experiment_suite.py \
  --manifest experiments/output/<manifest_hash>/manifest.json \
  --suite-config experiments/suites/paper_reproduction.json \
  --run-label paper_reproduction
```

Results are stored below `experiments/output/<manifest_hash>/`. Each run contains the full per-stock trace, portfolio metrics, runtime metadata, and a compact leaderboard row.


