#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = financial_market_crash_anomaly_detection_and_investment_strategy
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

# Define the path to the CSV file
FEATURES_CSV = data/processed/features.csv

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 financial_market_crash_anomaly_detection_and_investment_strategy
	isort --check --diff --profile black financial_market_crash_anomaly_detection_and_investment_strategy
	black --check --config pyproject.toml financial_market_crash_anomaly_detection_and_investment_strategy

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml financial_market_crash_anomaly_detection_and_investment_strategy




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) financial_market_crash_anomaly_detection_and_investment_strategy/dataset.py

## Process raw data and generate features
.PHONY: process_data
process_data: requirements
	@echo "Processing data using $(FEATURES_CSV)"
	$(PYTHON_INTERPRETER) financial_market_crash_anomaly_detection_and_investment_strategy/dataset.py --input $(FEATURES_CSV)

## Train anomaly detection model
.PHONY: train_model
train_model: requirements process_data
	$(PYTHON_INTERPRETER) financial_market_crash_anomaly_detection_and_investment_strategy/modeling/train.py

## Generate plots from models
.PHONY: plot_models
plot_models: requirements
	$(PYTHON_INTERPRETER) financial_market_crash_anomaly_detection_and_investment_strategy/plots.py --output-path reports/figures


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
