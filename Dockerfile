FROM python:3.7-buster

COPY requirements.txt /

RUN pip install -r /requirements.txt

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app/src

COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "src/malice/runner.py"]
CMD ["data/dev.csv", "--pop_size", "5", "--pop_iter", "1", "--evo_max_iter", "10", "--least_squares_max_iter", "1", "--thread_count", "1", "--bootstraps", "2", "--deterministic"]
