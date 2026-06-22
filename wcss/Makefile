# SpecEdge reproduction on WCSS — uruchamianie faz przez make.
#
# Użycie:
#   make -C wcss help
#   make -C wcss all          # wszystkie testy reprodukcyjne (submit SLURM)
#   make -C wcss collect      # metryki (po zakończeniu jobów)

REPRO := ./reproduce_wcss.sh

.PHONY: help setup check status collect finish \
	phase0 phase1-14b phase1-32b phase2 phase3 \
	list list-all list-phase0 list-phase1-14b list-phase1-32b list-phase2 list-phase3 \
	core-14b all-14b all

help:
	@echo "SpecEdge WCSS reproduction — cele make:"
	@echo ""
	@echo "  all            ★ wszystkie fazy reprodukcyjne (0, 1-14b, 2, 3, 1-32b)"
	@echo "  collect        agreguj metryki (uruchom po zakończeniu jobów SLURM)"
	@echo "  finish         collect (alias)"
	@echo ""
	@echo "  setup          chmod +x skryptów (jednorazowo)"
	@echo "  check          sprawdź Python/venv przed submitowaniem jobów"
	@echo "  status         kolejka SLURM (squeue)"
	@echo ""
	@echo "  phase0         smoke test (1 job)"
	@echo "  phase1-14b     Tabela 1 — Qwen3-14B (4 joby)"
	@echo "  phase1-32b     Tabela 1 — Qwen3-32B (2 joby)"
	@echo "  phase2         batch-size sweep (4 joby)"
	@echo "  phase3         ablation komponentów (3 joby)"
	@echo ""
	@echo "  core-14b       phase0 + phase1-14b"
	@echo "  all-14b        phase0 + phase1-14b + phase2 + phase3"
	@echo ""
	@echo "  list           podgląd wszystkich eksperymentów (bez submit)"
	@echo "  list-<faza>    podgląd jednej fazy"
	@echo ""
	@echo "Zmienne środowiskowe: WCSS_CACHE_PREFILL, WCSS_PARTITION, WCSS_SPECEDGE_ROOT"
	@echo "Szczegóły: wcss/README.md"

setup:
	chmod +x $(REPRO) lib/run_in_job.sh

check: setup
	@bash -c 'source lib/common.sh && wcss_check_project && echo "OK: WCSS_PYTHON=$$WCSS_PYTHON (collect metryk: lokalnie po rsync, polars nie działa na login node)"'

# Wszystkie testy reprodukcyjne — submit 14 jobów SLURM (fazy 0–3 + 32B)
all: setup
	$(REPRO) phase all

status:
	$(REPRO) status

collect finish: setup
	$(REPRO) collect

phase0: setup
	$(REPRO) phase 0

phase1-14b: setup
	$(REPRO) phase 1-14b

phase1-32b: setup
	$(REPRO) phase 1-32b

phase2: setup
	$(REPRO) phase 2

phase3: setup
	$(REPRO) phase 3

list list-all: setup
	$(REPRO) list all

list-phase0: setup
	$(REPRO) list 0

list-phase1-14b: setup
	$(REPRO) list 1-14b

list-phase1-32b: setup
	$(REPRO) list 1-32b

list-phase2: setup
	$(REPRO) list 2

list-phase3: setup
	$(REPRO) list 3

core-14b: phase0 phase1-14b

all-14b: phase0 phase1-14b phase2 phase3
