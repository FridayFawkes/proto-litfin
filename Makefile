VENV=.venv
DEV_PYTHON=$(VENV)/bin/python

# Only creave virtual environment if it does not exist already
$(VENV) create-venv:
	@if [ ! -d "$(VENV)" ]; then python3 -m venv $(VENV); fi

# Run build frontend
build: build-clean
	pyproject-build

# Clean build artifacts
build-clean:
	@rm -rf dist build src/*.egg-info

# Install package locally
install-package: $(VENV)
	$(DEV_PYTHON) -m pip install .

clean-all: build-clean
	@rm -rf $(VENV)
	@make $(VENV)
