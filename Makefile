.PHONY: demo test sweep serve chaos clean

demo: ; python3 -m kosh generate --batches 120 && python3 -m kosh run --data data/demo
test: ; python3 -m pytest tests/ -q
sweep: ; python3 -m kosh sweep --seeds 12 --batches 120
serve: ; python3 -m kosh serve --data data/demo --port 8000
chaos: ; python3 -m kosh chaos --data data/demo
clean: ; rm -rf data __pycache__ .pytest_cache **/__pycache__
