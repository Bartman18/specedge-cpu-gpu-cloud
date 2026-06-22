# 3-tier cascade na klastrze (M₀ CPU → M₁ GPU → M₂ chmura/gRPC)

Odpowiednik harnessu `wcss/` (repro SpecEdge), ale dla **naszej** kaskady 3-poziomowej.
Zbiera **te same metryki** co SpecEdge, na tych samych benchmarkach z `data/` (domyślnie
`specbench`), bo emituje JSONL w identycznym schemacie (`client_0.jsonl` + `server.jsonl`),
który czyta istniejące `src/metric/specedge.py`.

## Mapowanie na tiery (jeden węzeł, 2× GPU)

| Tier | Gdzie | Proces |
|---|---|---|
| M₂ target (chmura) | `cuda:0` | `cloud_server.py` (gRPC, port 8000) → pisze `server.jsonl` |
| M₁ verifier | `cuda:1` | wewnątrz `bench_cloud_pipeline.py` |
| M₀ drafter | CPU | wewnątrz `bench_cloud_pipeline.py` |

Klient (M₀+M₁) liczony jako „edge", M₂ jako „server" — tak jak w metrykach SpecEdge.

## Setup venv (WCSS — ważne)

Nie używaj `uv` na tym klastrze: jego standalone Python jest zbudowany pod login
node i **segfaultuje przy każdym natywnym imporcie** (numpy/torch/grpc) na węzłach
GPU. Buduj venv **modułowym Pythonem klastra** (działa na wszystkich węzłach).

Komenda `module` **nie działa na login node** — venv buduj **w zadaniu
interaktywnym** (tam `module` i `pip` mają dostęp):

```bash
# 1) zadanie interaktywne na węźle GPU
srun --pty --account=hpc-madeyski-1742229651 -p lem-gpu-short -N1 -c8 \
     --mem=64gb --gres=gpu:hopper:1 --time=01:00:00 /bin/bash

# 2) w środku joba:
cd ~/specedge            # katalog projektu (cokolwiek to jest na Twoim koncie)
module load Python/3.13.1-GCC-14.2.0
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -c "import numpy, torch; print('OK', torch.__version__, torch.cuda.is_available())"
exit
```

`cloud3.sbatch` robi `source /usr/local/sbin/modules.sh` i ładuje ten sam moduł
(`WCSS_PYTHON_MODULE`, domyślnie `Python/3.13.1-GCC-14.2.0`) przed startem joba,
więc venv znajdzie bazowy Python. Dla `make … local` załaduj moduł sam.

## Uruchomienie

**Lokalnie / interaktywnie** (1 węzeł, 2 GPU):
```bash
# startuje serwer M2, czeka, odpala benchmark, sprząta
CLOUD3_TARGET_MODEL=Qwen/Qwen3-14B CLOUD3_DATASET=specbench bash script/cloud3.sh
```

**SLURM (LEM/H100):**
```bash
sbatch wcss/slurm/cloud3.sbatch
# inny target:
sbatch --export=ALL,CLOUD3_TARGET_MODEL=Qwen/Qwen3-32B wcss/slurm/cloud3.sbatch
```

Zmienne (`script/cloud3.sh`): `CLOUD3_TARGET_MODEL`, `CLOUD3_VERIFY_MODEL`, `CLOUD3_DRAFT_MODEL`,
`CLOUD3_DATASET`, `CLOUD3_SAMPLE_REQ_CNT`, `CLOUD3_MAX_NEW_TOKENS`, `CLOUD3_GAMMA1`,
`CLOUD3_RESULT_PATH`, `CLOUD3_EXP_NAME`, `CLOUD3_SERVER_DEVICE`, `CLOUD3_VERIFY_DEVICE`.

## Metryki

```bash
PYTHONPATH=src python src/metric/specedge.py -d <result_path>/<exp_name> -s overall --gpu H100_94
# subsety specbench: multi_turn | translation | summarization | question_answering |
#                    mathematical_reasoning | retrieval
```

Daje: throughput (tok/s), ITL (ms/tok), accepted tokens/rundę, czas i koszt serwera/edge,
cost-efficiency — te same definicje co dla SpecEdge.

## ⚠️ Uwagi (do weryfikacji na sprzęcie)

1. **Niezwalidowane end-to-end** — kod nie był uruchomiony z modelami/GPU. Najpierw bramka
   losslessness (wyjście 3-tier == greedy M₂, temp=0), patrz `plan-chmura-m2.md §6`.
2. **`wcss/lib/collect_results.py` ma rozjechane mapowanie pozycyjne** względem aktualnego
   `metric/specedge.py --plain` (32 pola; `parts[14]` to latencja serwera, a nie accepted-mean;
   `parts[16]` to overall-prefill, a nie ITL). Dotyczy też repro SpecEdge. Do czasu naprawy
   agreguj **tabelą** `metric/specedge.py` (działa), nie automatem collect.
3. **v1 = M₂ bezstanowy** (re-prefill co rundę) → `server.jsonl` ma `prefill` ustawiane z flagi
   klienta tylko jako etykieta podziału latencji; throughput/koszt liczone z czasu ściany (uczciwe).
4. **Koszt edge** liczony stawką RTX 4090 (z modelu kosztu SpecEdge) — porównuj speedupy, nie
   koszt bezwzględny (ta sama uwaga co `wcss/README.md` pkt 4).
```
