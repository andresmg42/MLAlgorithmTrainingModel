from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
import copy
from sklearn.cluster import KMeans
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import ray
import time
import argparse
from symbols_list_distribution import symbolsList
warnings.filterwarnings('ignore')



class RollingOLSRegression:

  def __init__(self,symbols_list,start_date,end_date):
    self.model=None;
    self.symbols_list=symbols_list;
    self.start_date=start_date;
    self.end_date=end_date;


  def load_data(self):
    data= yf.download(tickers=self.symbols_list,
                 start=self.start_date,
                 end=self.end_date,auto_adjust=False)
    return data

  #paralelizable
  def calculate_tecnical_indicators(self,data):
    df=copy.deepcopy(data)
    df=df.stack()
    df.index.names=['date','ticker']
    df.columns=df.columns.str.lower()


    # calculate Garman-Klass Volatility, rsi,bollinger bands, atr,macd,dollar volume
    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

    df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

    df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])

    df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])

    df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

    def compute_atr(stock_data):
        atr = pandas_ta.atr(high=stock_data['high'],
                            low=stock_data['low'],
                            close=stock_data['close'],
                            length=14)
        return atr.sub(atr.mean()).div(atr.std())

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    def compute_macd(close):
        macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
        return macd.sub(macd.mean()).div(macd.std())

    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

    df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

    return df

  def aggregate_to_monthly_level(self,df):

    # To reduce training time and experiment with features and strategies,
    # we convert the business-daily data to month-end frequency.


    last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                            'high', 'low', 'close']]

    data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                    df.unstack()[last_cols].resample('M').last().stack('ticker')],
                    axis=1)).dropna()

    # Calculate 5-year rolling average of dollar volume for each stocks before filtering.

    data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

    data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

    data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

    return data



  # parlelizable
  def Calculate_Montly_Returns(self,data):

    # To capture time series dynamics that reflect, for example,
    # momentum patterns, we compute historical returns using the method
    # .pct_change(lag), that is, returns over various monthly periods as identified by lags.

    def calculate_returns(df):

      outlier_cutoff = 0.005

      lags = [1, 2, 3, 6, 9, 12]

      for lag in lags:

          df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                      upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
      return df


    data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

    return data

  #paralizable
  def download_fama_f_and_calculate_rolling_f_b(self,data):
      factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

      factor_data.index = factor_data.index.to_timestamp()

      factor_data = factor_data.resample('M').last().div(100)

      factor_data.index.name = 'date'

      factor_data = factor_data.join(data['return_1m']).sort_index()

      # filter out stocks whit less than 10 months of data

      observations = factor_data.groupby(level=1).size()

      valid_stocks = observations[observations >= 10]

      factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

      #calculate rolling factor betas

      betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

      return betas

  def join_the_rolling_factors_data(self,betas,data):
      factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

      data = (data.join(betas.groupby('ticker').shift()))

      data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

      data = data.drop('adj close', axis=1)

      data = data.dropna()

      return data

  def apply_pre_defined_centroids(self):

    target_rsi_values = [30, 45, 55, 70]

    initial_centroids = np.zeros((len(target_rsi_values), 18))

    initial_centroids[:, 6] = target_rsi_values

    return initial_centroids

  def k_means_clustering(self,data,initial_centroids):
    def get_clusters(df):
      df['cluster'] = KMeans(n_clusters=4,
                            random_state=0,
                            init=initial_centroids).fit(df).labels_
      return df

    data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

    return data

  def portfolio_based__on_cluster(self,data):
    filtered_df = data[data['cluster']==3].copy()

    filtered_df = filtered_df.reset_index(level=1)

    filtered_df.index = filtered_df.index+pd.DateOffset(1)

    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

    dates = filtered_df.index.get_level_values('date').unique().tolist()

    fixed_dates = {}

    for d in dates:

        fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

    return fixed_dates


  def optimize_weights(prices, lower_bound=0):

    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)

    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)

    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')

    weights = ef.max_sharpe()

    return ef.clean_weights()


  def download_fresh_daily_prices(self,data):
    stocks = data.index.get_level_values('ticker').unique().tolist()

    new_df = yf.download(tickers=stocks,
                        start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                        end=data.index.get_level_values('date').unique()[-1],auto_adjust=False)

    return new_df
  #paralelizable
  def calculate_each_day_portfolio_return(self,new_df,fixed_dates):

    returns_dataframe = np.log(new_df['Adj Close']).diff()
    
    portfolio_df = pd.DataFrame()
    
    for start_date in fixed_dates.keys():
    
        try:
            end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            cols = fixed_dates[start_date]
            optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    
            optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
    
            success = False
            try:
                weights = optimize_weights(prices=optimization_df,
                                       lower_bound=round(1/(len(optimization_df.columns)*2),3))
                weights = pd.DataFrame(weights, index=optimization_df.columns, columns=[0])
                success = True
            except:
                print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
    
            if success==False:
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                         index=optimization_df.columns.tolist(),
                                         columns=[0])
    
            temp_df = returns_dataframe[start_date:end_date][cols]
    
            # Create portfolio returns by calculating weighted average
            portfolio_returns = []
            for date in temp_df.index:
                daily_return = 0
                for ticker in temp_df.columns:
                    if pd.notna(temp_df.loc[date, ticker]) and ticker in weights.index:
                        daily_return += temp_df.loc[date, ticker] * weights.loc[ticker, 0]
                portfolio_returns.append(daily_return)
    
            temp_portfolio_df = pd.DataFrame(portfolio_returns,
                                           index=temp_df.index,
                                           columns=['Strategy Return'])
    
            portfolio_df = pd.concat([portfolio_df, temp_portfolio_df], axis=0)
    
        except Exception as e:
            print(f"Error for {start_date}: {e}")
    
    portfolio_df = portfolio_df.drop_duplicates()
    
    return portfolio_df
    
    
  def train_pipeline(self):
    
    df=self.load_data()
    
    df_indicators = self.calculate_tecnical_indicators(df)
    
    df_aggregate= self.aggregate_to_monthly_level(df_indicators)
    
    df_montly_returns = self.Calculate_Montly_Returns(df_aggregate)
    
    betas = self.download_fama_f_and_calculate_rolling_f_b(df_montly_returns)
    
    join_data=self.join_the_rolling_factors_data(betas,df_montly_returns)
    
    centroids=self.apply_pre_defined_centroids()
    
    clusters=self.k_means_clustering(join_data,centroids)
    
    fixed_dates = self.portfolio_based__on_cluster(clusters)
    
    fresh_daily_prices=self.download_fresh_daily_prices(clusters)
    
    portfolio_returns=self.calculate_each_day_portfolio_return(fresh_daily_prices,fixed_dates)


    return portfolio_returns

    
        

if __name__=='__main__':

    
    parser=argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--index','-i',type=str,required=True, help='available stock index (s&p500,downjones)')
    parser.add_argument('--start_date','-s',type=str,required=True, help='start date to train')
    parser.add_argument('--end_date','-e',type=str,required=True, help='end date to train')
    
    args=parser.parse_args()

    symbols_list=symbolsList(args.index)

    rolling=RollingOLSRegression(symbols_list,args.start_date,args.end_date)
 
    results=rolling.train_pipeline()

    print(results.isna().sum().sum())

    
    
    


      
