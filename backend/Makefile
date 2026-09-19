.PHONY: run test test-fast init-db install seed docker-up

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -v

test-fast:
	pytest -v -m "not integration"

init-db:
	python -c "from app.db import engine, Base; Base.metadata.create_all(bind=engine)"

install:
	pip install -r requirements.txt

seed:
	python scripts/seed_demo_data.py

docker-up:
	docker compose up --build -d
