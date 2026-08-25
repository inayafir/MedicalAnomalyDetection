.PHONY: run test init-db install

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -v

init-db:
	python -c "from app.db import engine, Base; Base.metadata.create_all(bind=engine)"

install:
	pip install -r requirements.txt
