FROM jupyter/datascience-notebook:latest

RUN pip install ray yfinance pandas_ta PyPortfolioOpt  pandas_datareader requests "fastapi[standard]" "ray[data,train,tune,serve]"

CMD bash -c "ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265  && start-notebook.sh"