# Reprodukcja SpecEdge na WCSS (LEM / H100)

Skrypty do odtworzenia wyników z artykułu (fazy 0–3) na klastrze LEM.

## Wymagania

- Projekt w `~/specedge` z działającym `.venv` (patrz **Setup venv** poniżej)
- Usługa SLURM: `hpc-madeyski-1742229651`
- Modele HuggingFace w cache Lustre (domyślnie `.../specedge_models`)
- **Brak SSH** — edge uruchamiany lokalnie (`wcss/lib/client_local.py`)

## Setup venv (jednorazowo na ui.wcss.pl)

```bash
cd ~/specedge
uv sync
make -C wcss check    # powinno wypisać: OK: WCSS_PYTHON=.../specedge/.venv/bin/python3
```

Jeśli `uv` nie jest dostępne, użyj modułów WCSS (`module avail Python`) lub wskaż interpreter ręcznie:
`export WCSS_PYTHON=/ścieżka/do/python3`

## Szybki start

```bash
cd ~/specedge

# Makefile (zalecane)
make -C wcss help
make -C wcss check
make -C wcss setup
make -C wcss all           # ★ wszystkie fazy (0, 1-14b, 2, 3, 1-32b) → SLURM
make -C wcss collect       # po zakończeniu jobów

# Pojedyncze fazy:
make -C wcss phase0
make -C wcss core-14b      # fazy 0 + 1-14b
make -C wcss all-14b       # fazy 0–3 (14B)

# Lub bezpośrednio:
chmod +x wcss/reproduce_wcss.sh wcss/lib/run_in_job.sh

# 0) Smoke test (4 zapytania)
./wcss/reproduce_wcss.sh phase 0

# 1) Tabela 1 — modele 14B (4 joby: 2× server-only + 2× specedge)
./wcss/reproduce_wcss.sh phase 1-14b

# 1b) Tabela 1 — Qwen3-32B (po sukcesie 14B)
./wcss/reproduce_wcss.sh phase 1-32b

# 2) Batch size (server-only BS 1/2/4; specedge BS 1)
./wcss/reproduce_wcss.sh phase 2

# 3) Ablation komponentów
./wcss/reproduce_wcss.sh phase 3

# Zbierz metryki
./wcss/reproduce_wcss.sh collect
```

## Wyniki

| Ścieżka | Zawartość |
|---|---|
| `/lustre/pd03/.../specedge_repro/results/<run_id>/` | JSONL, config, logi joba |
| `/lustre/pd03/.../specedge_repro/summaries/summary.csv` | Wszystkie metryki |
| `/lustre/pd03/.../specedge_repro/summaries/speedup_comparison.csv` | Speedupy vs server-only |

## Zużycie GPU (szacunek)

| Faza | Joby | GPU×h (szacunek) |
|---|---|---|
| 0 | 1 | ~1 |
| 1-14b | 4 | ~16–24 |
| 1-32b | 2 | ~12–20 |
| 2 | 4 | ~12–16 |
| 3 | 3 | ~6–9 |
| **Razem** | **14** | **~47–70 GPU×h** |

Rekomendowany budżet: **60 GPU×h** (zapas na kolejkę i pierwsze próby).

## Ograniczenia WCSS

1. **Edge = drugi/trzeci H100** na tym samym węźle (nie RTX 4090 z artykułu).
2. **RTT ≈ 0 ms** (localhost) zamiast 14 ms WAN — wpływa głównie na ITL, nie na throughput servera.
3. **SpecEdge BS>1**: węzeł ma 4 GPU; BS=2 wymaga 5 GPU wg artykułu — faza 2 uruchamia specedge tylko dla BS=1.
4. Koszt edge w metrykach specedge nadal liczony stawką RTX 4090 (model z kodu) — porównuj speedupy, nie bezwzględny cost efficiency.

## Zmienne środowiskowe

```bash
export WCSS_CACHE_PREFILL=true    # domyślnie; patrz wyjaśnienie w reproduce_wcss.sh
export WCSS_PARTITION=lem-gpu-normal  # dłuższe joby (32B)
```

## Kolejność faz

```
phase 0 → phase 1-14b → collect (wstępna analiza) → phase 2 → phase 3 → phase 1-32b → collect
```
