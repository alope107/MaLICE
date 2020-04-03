FROM python:3.7-buster

COPY complex/requirements.txt /

RUN pip install -r /requirements.txt

RUN pip install pytest

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app/src

COPY complex /app
WORKDIR /app

CMD ["pytest"]
