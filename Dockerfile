FROM python:3.7-buster

COPY requirements.txt /

RUN pip install -r /requirements.txt

RUN pip install pytest

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app/src

COPY . /app
WORKDIR /app

CMD ["pytest"]

# TODO: Restore old Dockerfile and find a new home for this one
