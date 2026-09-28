#!/bin/sh
# Every LLM-agent number, table and figure in the paper, from the committed answer tables.
# No model is needed: llm/tables/ holds the answers of qwen2.5:7b and llama3.1:8b (six option
# orders, both scenarios), qwen2.5:14b and qwen2.5:1.5b, and the scheduler only ever sees what it paid for.
# About 3-4 hours on one core. Per-run CSVs land in paper/data/, figures in figures/out/llm_*,
# tables in paper/generated/.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
D=paper/data
mkdir -p "$D"
SEEDS="0 1 2 3 4 5 6 7 8 9"

# the reference as prompted (options in the listed order): the conference paper's main tables
$PY -m bench.llm_agents --agents 300 1000 2000 --seeds $SEEDS \
    --csv $D/station_7b_fixed.csv > $D/station_7b_fixed.log
$PY -m bench.llm_agents --scenario museum --seconds 900 --agents 300 1000 2000 --seeds $SEEDS \
    --rehearse 0.2 --max-wait 0 5 15 --csv $D/museum_7b_fixed.csv > $D/museum_7b_fixed.log
$PY -m bench.llm_agents --surrogate marginal --agents 1000 2000 --seeds $SEEDS \
    --policies parity parity_nocap cascade view_lod round_robin --csv $D/station_marginal_fixed.csv > $D/station_marginal_fixed.log
# a second model family: llama3.1:8b (all six option orders tabulated)
$PY -m bench.llm_agents --model llama3.1:8b --agents 300 1000 2000 --seeds $SEEDS \
    --csv $D/station_llama.csv > $D/station_llama.log
$PY -m bench.llm_agents --model llama3.1:8b --order-free --agents 300 1000 2000 --seeds $SEEDS \
    --csv $D/station_llama_of.csv > $D/station_llama_of.log
$PY -m bench.llm_agents --model llama3.1:8b --scenario museum --seconds 900 --agents 300 1000 2000 --seeds $SEEDS \
    --rehearse 0.2 --max-wait 5 --csv $D/museum_llama.csv > $D/museum_llama.log
# station, 7B, order-free reference: PARITY, its ablation and the baselines, 10 seeds
$PY -m bench.llm_agents --order-free --agents 300 1000 2000 --seeds $SEEDS \
    --csv $D/station_7b.csv > $D/station_7b.log
# museum evacuation, 7B, order-free: plus rehearsal and bounded waits
$PY -m bench.llm_agents --scenario museum --order-free --seconds 900 --agents 300 1000 2000 --seeds $SEEDS \
    --rehearse 0.2 --max-wait 0 5 15 --csv $D/museum_7b.csv > $D/museum_7b.log
# generality: a second reference model (fixed option order; its six orders are not tabulated)
$PY -m bench.llm_agents --model qwen2.5:14b --agents 300 1000 2000 --seeds $SEEDS \
    --csv $D/station_14b.csv > $D/station_14b.log
# a weak surrogate (per-persona average), where the ledger binds
$PY -m bench.llm_agents --order-free --surrogate marginal --agents 1000 2000 --seeds $SEEDS \
    --policies parity parity_nocap cascade view_lod round_robin --csv $D/station_marginal.csv > $D/station_marginal.log
# one setting at a time: cap, refit interval, off-screen salience, call budget, rehearsal share, wait bound
$PY -m bench.llm_sensitivity --out $D/sensitivity.csv > $D/sensitivity.log
# what the scheduler itself costs
$PY -m bench.llm_overhead > $D/overhead.log

$PY -m figures.llm_plots
