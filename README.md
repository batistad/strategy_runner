# install
pip install -e .[yahoo]

# run crossover (uses config.yaml)
strategyrunner run --config config.yaml

# backfill a specific date (signals use next-bar logic)
strategyrunner run --config config.yaml --asof 2025-09-12 --dry