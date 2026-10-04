.PHONY: install train evaluate test predict app clean help

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install     - Install project dependencies"
	@echo "  make train       - Run complete training pipeline and CV"
	@echo "  make evaluate    - Run model evaluation and report generation"
	@echo "  make test        - Run test suite with pytest"
	@echo "  make predict     - Generate predictions (specify INPUT and OUTPUT)"
	@echo "  make app         - Launch interactive Streamlit dashboard"
	@echo "  make clean       - Remove cached files and bytecode"

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

test:
	$(PYTHON) -m pytest tests/ -v

predict:
	$(PYTHON) -m src.predict --input $(or $(INPUT),ai4i2020.csv) --output $(or $(OUTPUT),predictions/predictions.csv)

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
