# ── rtknnc Makefile ───────────────────────────────────────────────────────
# Targets:
#   make install      instala o pacote em modo editável
#   make test         roda os testes com pytest
#   make coverage     testa com relatório de cobertura
#   make lib          compila a biblioteca C compartilhada (librtknnc.so)
#   make rtknnc       compila o executável standalone linkado à lib
#   make clean        remove artefatos de build e compilação
#   make lint         verifica o código Python com ruff
#   make format       formata o código Python com ruff
#   make clib-deps    gera grafo/relatórios de dependências dos headers C
#   make all          install + lib + rtknnc + test

PYTHON     := python
PIP := $(PYTHON) -m pip
# PIP        := pip
PYTEST  := $(PYTHON) -m pytest
# PYTEST     := pytest
RUFF       := ruff

# Diretórios
SRC_DIR    := src/rtknnc
CLIB_DIR   := $(SRC_DIR)/_clib
INC_DIR    := $(CLIB_DIR)/include
TEST_DIR   := tests
CLIB_DEPS_OUT := build/clib-deps

# ── Biblioteca C compartilhada ────────────────────────────────────────────────
LIB        := $(CLIB_DIR)/librtknnc.so
# CC         := gcc
CFLAGS     := -O2 -Wall -fPIC -shared -I$(INC_DIR) $(DIRECTIVES)
C_SRCS := $(CLIB_DIR)/hyperquadtree.c \
          $(CLIB_DIR)/k-nn.c \
          $(CLIB_DIR)/k-nn-topology.c \
		  $(CLIB_DIR)/io-stuff.c \
          $(CLIB_DIR)/set-stuff.c \
          $(CLIB_DIR)/z-norm.c \
          $(CLIB_DIR)/showresults.c \
          $(CLIB_DIR)/show-graphs.c

.PHONY: all install test coverage lib rtknnc clean lint format clib-deps help

# ── targets ───────────────────────────────────────────────────────────────────

all: install lib rtknnc test

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) $(TEST_DIR) -v

coverage:
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Compila librtknnc.so a partir dos módulos C (sem main)
lib: $(LIB)

$(LIB): $(C_SRCS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "✓ Built $(LIB)"

# ── Executável standalone ─────────────────────────────────────────────────────
EXE        := rtknnc
CFLAGS_EXE := -O2 -Wall -I$(INC_DIR) $(DIRECTIVES)

EXE_SRCS   := $(CLIB_DIR)/main.c \
              $(CLIB_DIR)/io-stuff.c \
              $(CLIB_DIR)/showresults.c \
              $(CLIB_DIR)/show-graphs.c

rtknnc: $(LIB) $(EXE_SRCS)
	$(CC) $(CFLAGS_EXE) -o $(EXE) $(EXE_SRCS) \
	    -L$(CLIB_DIR) -lcannon -Wl,-rpath,$(CLIB_DIR) -lm
	@echo "✓ Built $(EXE)"

clib-deps:
	@mkdir -p $(CLIB_DEPS_OUT)
	$(PYTHON) scripts/clib_deps.py --root $(CLIB_DIR) --out $(CLIB_DEPS_OUT)
	@echo "✓ Dependências em $(CLIB_DEPS_OUT)/"

clean:
	@echo "Cleaning build artefacts..."
	rm -f $(LIB) $(EXE)
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete
	find . -type f -name "*.pyo"     -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"   -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
	@echo "✓ Clean"

lint:
	$(RUFF) check $(SRC_DIR) $(TEST_DIR)

format:
	$(RUFF) format $(SRC_DIR) $(TEST_DIR)

help:
	@echo ""
	@echo "rtknnc — available targets:"
	@echo "  make install      instala o pacote em modo editável (pip install -e)"
	@echo "  make test         roda todos os testes com pytest"
	@echo "  make coverage     testes + relatório de cobertura"
	@echo "  make lib          compila librtknnc.so a partir dos fontes C"
	@echo "  make rtknnc       compila o executável linkado à librtknnc.so"
	@echo "  make clean        remove artefatos de build e .pyc"
	@echo "  make clib-deps    gera grafo de dependências dos headers C"
	@echo "  make lint         verifica estilo com ruff"
	@echo "  make format       formata código com ruff"
	@echo "  make all          install + lib + rtknnc + test"
	@echo ""
