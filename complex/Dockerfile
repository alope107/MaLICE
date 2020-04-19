FROM python:3.7-buster

COPY requirements.txt /

RUN pip install -r /requirements.txt

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app/src

COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "src/run_and_upload.py"]

#../data/dev.csv 
CMD ["data/minidev.csv",                \
     "--phase1_islands", "2",          \ 
     "--phase1_generations", "10",     \ 
     "--phase1_evo_rounds", "1",       \ 
     "--phase2_islands", "2",          \ 
     "--phase2_generations", "10",     \ 
     "--phase2_evo_rounds", "1",       \ 
     "--least_squares_max_iter", "10", \ 
     "--mcmc_walks", "10",             \ 
     "--mcmc_steps", "20",             \ 
     "--s3_prefix", "malice/test"]

# CMD ["data/dev.csv",                    \
#      "--pop_size", "5",                 \
#      "--pop_iter", "1",                 \
#      "--evo_max_iter", "10",            \
#      "--least_squares_max_iter", "1",   \
#      "--thread_count", "1",             \
#      "--bootstraps", "2",               \
#      "--s3_prefix", "malice/test",      \
#      "--output_dir", "default",         \
#      "--deterministic"]

