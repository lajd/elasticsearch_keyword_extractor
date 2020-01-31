IMAGE_NAME=lajd94/elasticsearch-keyword-extractor:latest

build:
	docker build -t $(IMAGE_NAME) .

push:
	docker push $(IMAGE_NAME)

test-service-up:
	docker-compose -f docker-compose/docker-compose.yml build \
	&& docker-compose -f docker-compose/docker-compose.yml up -d

test-service-down:
	docker-compose -f docker-compose/docker-compose.yml down

test:
	pip install -r requirements.txt \
	&& pytest tests.py
