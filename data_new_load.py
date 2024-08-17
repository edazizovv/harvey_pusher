#
import json
import time
import datetime
import urllib3

#
import numpy
import pandas

import ccxt

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange

#


#
exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({# 'apiKey': 'YOUR_API_KEY',
                           # 'secret': 'YOUR_SECRET',
                           # 'timeout': 30000,
                           # 'enableRateLimit': True,
                           })

api_key = ''

ts = TimeSeries(key=api_key, output_format='pandas')
cc = CryptoCurrencies(key=api_key, output_format='pandas')
fe = ForeignExchange(key=api_key, output_format='pandas')

symbols = ['BTC', 'ETH', 'LTC', 'DOGE', 'XRP']
equities = ['FDN', 'VPU']
ex_symbols = ['EUR', 'JPY']
drops = ['1b. open', '2b. high', '3b. low', '4b. close', '6']


def get_fx(from_symbol, to_symbol, interval='60min', fun='FX_INTRADAY'):
    http = urllib3.PoolManager()
    r = http.request('GET',
                     'https://www.alphavantage.co/query?function={0}&from_symbol={1}&to_symbol={2}&interval=60min&apikey={3}'.format(fun,
                         from_symbol, to_symbol, api_key))
    pp = numpy.array([x for x in json.loads(r.data)['Time Series FX (60min)'].values()])
    tt = list(json.loads(r.data)['Time Series FX (60min)'].keys())
    tt = pandas.to_datetime(tt)
    ppp = [[x[y] for y in x] for x in pp]
    cols = list(pp[0].keys())
    da = pandas.DataFrame(data=ppp, columns=cols, index=tt)
    return da


def get_eq(symbol, interval='60min', fun='TIME_SERIES_INTRADAY'):
    http = urllib3.PoolManager()
    r = http.request('GET',
                     'https://www.alphavantage.co/query?function={0}&symbol={1}&interval=60min&apikey={2}'.format(fun,
                         symbol, api_key))
    pp = numpy.array([x for x in json.loads(r.data)['Time Series (60min)'].values()])
    tt = list(json.loads(r.data)['Time Series (60min)'].keys())
    tt = pandas.to_datetime(tt)
    ppp = [[x[y] for y in x] for x in pp]
    cols = list(pp[0].keys())
    da = pandas.DataFrame(data=ppp, columns=cols, index=tt)
    return da


def rename(x):
    if '1a' in x or '1.' in x:
        return 'Open'
    if '2a' in x or '2.' in x:
        return 'High'
    if '3a' in x or '3.' in x:
        return 'Low'
    if '4a' in x or '4.' in x:
        return 'Close'
    if '5' in x:
        return 'Volume'


def get_quotes(hour):
    data_ = []
    """
    for symbol in equities:
        try:
            data__ = ts.get_daily(symbol=symbol)[0]
        except Exception as e:
            print("\n\tAlphaVantage Exception [Equities]\n")
            print(e)
        data__ = data__.rename(
            columns={column: '{0}-{1}_{2}'.format(symbol, 'USD', rename(column)) for column in data__.columns.values})
        data_.append(data__)
        time.sleep(20)
    """
    for symbol in equities:
        try:
            # data__ = ts.get_daily(symbol=symbol)[0]
            data__ = get_eq(symbol=symbol)
        except Exception as e:
            print("\n\tAlphaVantage Exception [Equities]\n")
            print(e)
        data__ = data__.rename(
            columns={column: '{0}-{1}_{2}'.format(symbol, 'USD', rename(column)) for column in data__.columns.values})
        data_.append(data__)
        time.sleep(20)
    """
    for symbol in symbols:
        try:
            data__ = cc.get_digital_currency_daily(symbol=symbol, market='USD')[0]
        except Exception as e:
            print("\n\tAlphaVantage Exception [Crypto]\n")
            print(e)
        data__ = data__.drop(columns=[x for x in data__.columns.values if any([y in x for y in drops])])
        data__ = data__.rename(
            columns={column: '{0}-{1}_{2}'.format(symbol, 'USD', rename(column)) for column in data__.columns.values})
        data_.append(data__)
        time.sleep(20)
    """

    for symbol in symbols:
        try:
            # exchange.fetch_ticker(symbol)
            print('{0}/USDT'.format(symbol))
            result = exchange.fetch_ohlcv('{0}/USDT'.format(symbol), '1h')
            columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data__ = pandas.DataFrame(data=result, columns=columns)
            data__['date'] = data__['date'].apply(func=lambda x: datetime.datetime.utcfromtimestamp(x / 1000.))
            data__['date'] = pandas.to_datetime(data__['date'])
            data__ = data__.set_index('date')
        except Exception as e:
            print("\n\tAlphaVantage Exception [Crypto]\n")
            print(e)
        data__ = data__.rename(
            columns={column: '{0}-{1}_{2}'.format(symbol, 'USD', column) for column in data__.columns.values})
        data_.append(data__)
        time.sleep(20)

    """
    for symbol in ex_symbols:
        try:
            data__ = fe.get_currency_exchange_daily(from_symbol=symbol, to_symbol='USD')[0]
        except Exception as e:
            print("\n\tAlphaVantage Exception [Exchange]\n")
            print(e)
        data__ = data__.rename(
            columns={column: '{0}-{1}_{2}'.format(symbol, 'USD', rename(column)) for column in data__.columns.values})
        data_.append(data__)
        time.sleep(20)
    """

    for symbol in ex_symbols:
        try:
            # data__ = fe.get_currency_exchange_daily(from_symbol=symbol, to_symbol='USD')[0]
            data__ = get_fx(from_symbol=symbol, to_symbol='USD')
        except Exception as e:
            print("\n\tAlphaVantage Exception [Exchange]\n")
            print(e)
        data__ = data__.rename(
            columns={column: '{0}-{1}_{2}'.format(symbol, 'USD', rename(column)) for column in data__.columns.values})
        data_.append(data__)
        time.sleep(20)

    _data = pandas.concat(data_, axis=1)

    _data = _data.fillna(method='ffill')
    _data = _data.sort_index(ascending=True)
    _data = _data.dropna()
    _data = _data[_data.index.hour == hour]
    _data = _data.astype(dtype=numpy.float64)

    return _data



# datetime.datetime.utcfromtimestamp(huh[0][0] / 1000.)
