# Delegate native compilation and package copying to one build definition.
PYTHON ?= python
PYTEST_ARGS ?= tests/ -v

.PHONY: all install test clean

all:
	$(MAKE) -C native

# native/Makefile also copies the binaries into metalq/lib.
install: all

test: install
	$(PYTHON) -m pytest $(PYTEST_ARGS)

clean:
	$(MAKE) -C native clean
