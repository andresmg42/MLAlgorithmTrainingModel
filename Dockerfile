FROM jupyter/datascience-notebook:latest

RUN pip install ray yfinance pandas_ta PyPortfolioOpt PyPortfolioOpt ray pandas_datareader

CMD bash -c "ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 && start-notebook.sh"