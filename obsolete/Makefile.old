# ── cannon-knn Makefile ───────────────────────────────────────────────────────
# Targets:
#   make install      instala o pacote em modo editável
#   make test         roda os testes com pytest
#   make coverage     testa com relatório de cobertura
#   make lib          compila a biblioteca C compartilhada (libcannon.so)
#   make clean        remove artefatos de build e compilação
#   make lint         verifica o código Python com ruff
#   make format       formata o código Python com ruff
#   make clib-deps    gera grafo/relatórios de dependências dos headers C
#   make all          install + lib + test

PYTHON     := python
PIP        := pip
PYTEST     := pytest
RUFF       := ruff

# Diretórios
SRC_DIR    := src/cannon_knn
CLIB_DIR   := $(SRC_DIR)/_clib
INC_DIR    := $(CLIB_DIR)/include
TEST_DIR   := tests
CLIB_ROOT  := $(CLIB_DIR)
CLIB_DEPS_OUT := build/clib-deps

# Biblioteca C
LIB        := $(CLIB_DIR)/libcannon.so
CC         := gcc
CFLAGS     := -O2 -Wall -fPIC -shared -I$(INC_DIR)
C_SRCS     := $(CLIB_DIR)/hyperquadtree.c \
              $(CLIB_DIR)/k-nn.c \
              $(CLIB_DIR)/k-nn-topology.c \
              $(CLIB_DIR)/set-stuff.c \
              $(CLIB_DIR)/z-norm.c


.PHONY: all install test coverage lib clean lint format help clib-deps clib-deps-strict clib-deps-clean

# ── targets ───────────────────────────────────────────────────────────────────

all: install lib test

clib-deps:
	@$(PYTHON) scripts/clib_deps.py --root $(CLIB_ROOT) --out $(CLIB_DEPS_OUT)

clib-deps-strict:
	@$(PYTHON) scripts/clib_deps.py --root $(CLIB_ROOT) --out $(CLIB_DEPS_OUT) \
		--fail-on-cycles --fail-on-redundant

clib-deps-clean:
	@rm -rf $(CLIB_DEPS_OUT)

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) $(TEST_DIR) -v

coverage:
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

lib: $(LIB)

$(LIB): $(C_SRCS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "✓ Built $(LIB)"

clean:
	@echo "Cleaning build artefacts..."
	rm -f $(LIB)
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete
	find . -type f -name "*.pyo"     -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	
BIN        := cannon-knn
BIN_SRC    := $(CLIB_DIR)/main.c
CEXEFLAGS  := -O2 -Wall -I$(INC_DIR)
LDLIBS     := -lm

$(BIN): $(BIN_SRC)
	$(CC) $(CEXEFLAGS) -o $@ $< $(LDLIBS)
	@echo "✓ Built $(BIN)"

run-cannon-knn: $(BIN)
	./$(BIN) $(ARGS)
