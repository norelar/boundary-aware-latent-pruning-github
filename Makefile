.PHONY: install test compile lint clean

install:
	python -m pip install -e ".[all]"

test:
	python -m unittest discover -s tests -v

compile:
	python -m compileall -q src tests

lint:
	ruff check src tests

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -r {} +
