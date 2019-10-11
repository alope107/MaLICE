FROM python:3.7-slim-buster

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "src/malice.py"]
CMD ["data/dev.csv", "--pop_size", "5", "--pop_iter", "1", "--evo_max_iter", "10", "--least_squares_max_iter", "1", "--thread_count", "1"]
