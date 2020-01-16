
build:
	docker build -t lajd94/elasticsearch-keyword-extractor:latest .

push:
	docker push lajd94/elasticsearch-keyword-extractor:latest

test-service-up:
	docker-compose -f docker-compose/docker-compose.yml build \
	&& docker-compose -f docker-compose/docker-compose.yml up -d

test-service-down:
	docker-compose -f docker-compose/docker-compose.yml down

test:
	pip install -r requirements.txt \
	&& pytest tests.py
