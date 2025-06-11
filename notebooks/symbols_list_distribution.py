
import pandas as pd


def sp500():
    
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    
    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
    
    symbols_list = sp500['Symbol'].unique().tolist()
    
    symbols_list=list(filter(lambda x: x not in ['SOLV', 'GEV', 'SW', 'VLTO'],symbols_list))

    return symbols_list

def down_jones():

    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG',
        'MA', 'UNH', 'DIS', 'HD', 'VZ', 'NVDA', 'PYPL', 'BAC', 'ADBE', 'CMCSA',
        'NFLX', 'NKE', 'KO', 'PEP', 'MRK', 'PFE', 'XOM', 'CVX', 'WMT', 'T'
        ]
    
def symbolsList(index):
    match index:
        case 's&p500': return sp500()

        case 'downjones': return down_jones() 
            
