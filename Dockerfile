# Single stage is enough: there is nothing to compile and no dependency to
# install. The image is the interpreter plus this repo.
FROM python:3.12-slim

WORKDIR /app
COPY kosh/ ./kosh/
COPY web/ ./web/
COPY tests/ ./tests/
COPY pyproject.toml README.md Makefile ./

# Generate a dataset at build time so `docker run` shows something immediately
# rather than an empty dashboard.
RUN python -m kosh generate --batches 120 --out data/demo

EXPOSE 8000
CMD ["python", "-m", "kosh", "serve", "--data", "data/demo", "--port", "8000"]
