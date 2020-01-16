FROM python:3.8-alpine

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

COPY extraction /extraction
COPY main.py /main.py

ENTRYPOINT ["python", "main.py"]
