#
import time
import json
import urllib3

#
import numpy
import pandas
import schedule

from m_utils.transform import lag_it

#
from data_new_load import get_quotes
from reporter import Controls
from poster import vk_post, tg_post


#
def call_all():

    # prepare data

    _data = get_quotes()

    data = numpy.log(_data.pct_change() + 1)
    data = lag_it(data, n_lags=1, exactly=False)
    data = data.dropna()
    prediction_ms = [x for x in data.columns.values if 'LAG0' not in x]
    prediction_xs = data[prediction_ms]

    # api call

    targets = ['BTC-USD', 'DOGE-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD']
    predicted_ = []
    horizon = 96

    for target in targets:

        rsg = []
        for j in range(horizon):
            rgg = {'target': target}
            for col in prediction_ms:
                rgg[col] = data[col].values[-j-1]
            rsg.append(rgg)

        encoded_body = json.dumps(rsg)

        http = urllib3.PoolManager()

        r = http.request('POST', 'http://185.149.243.162:8000/predict/m1',
                         headers={'Content-Type': 'application/json'},
                         body=encoded_body)

        pp = numpy.array([x for x in json.loads(r.data)['predictions'].values()]).reshape(-1, 1)
        predicted_.append(pp)

    predicted = pandas.DataFrame(index=data.index.values[-horizon:], data=numpy.concatenate(predicted_, axis=1), columns=targets)

    # revert logpcts

    cls = ['{0}_Close'.format(x) for x in targets]
    rany = predicted.index.values + pandas.Timedelta(days=-1)
    base = _data.query("@rany.min() <= index and index <= @rany.max()")
    dw = numpy.exp(predicted.values + numpy.log(base[cls].values))
    predicted_rev = pandas.DataFrame(index=predicted.index.values, columns=predicted.columns.values,
                                     data=dw)
    predicted_compare = pandas.concat((base[cls], predicted_rev), axis=1)

    statuses = {}
    predoooo = {}
    for target in targets:

        contra = Controls()
        contra.check(X=data[prediction_ms].values[-horizon-1:-1],
                     Y_true=predicted_compare['{0}_Close'.format(target)].dropna().values,
                     Y_hat=predicted_compare[target].dropna().values)
        statuses[target] = {'cc': contra,
                            'target': contra.target,
                            'errors': contra.errors,
                            'warns': contra.warns}

        predoooo[target] = predicted_compare[target].values[-1]

    eld = predicted_compare[[target + '_Close' for target in targets]].values[-2]
    eldy = {targets[j]: eld[j] for j in range(len(targets))}
    
    co = {k: {r: statuses[k][r] for r in statuses[k].keys() if r != 'cc'} for k in statuses.keys()}
    save_data = pandas.DataFrame(columns=(targets + ['{0}_Close'.format(target) for target in targets]), index=predicted_compare.index)
    
    for target in targets:
    
        save_data[target] = predicted_compare[target].copy()
        save_data['{0}_Close'.format(target)] = predicted_compare['{0}_Close'.format(target)].copy()
    
    save_data = save_data.dropna()
    
    save_dir = 'C:/Theta/J/crypted_tablo/data/'
    
    with open(save_dir + 'co.json', 'w') as f:
        json.dump(co, f)
    save_data.to_csv(save_dir + 'data.csv')
    
    # post to VK and TG

    # vk_post(statuses, predoooo, eldy)
    # tg_post(statuses, predoooo, eldy)


# schedule.every().day.at("01:00").do(call_all)
# schedule.every().hour.do(call_all)
"""
while True:
    schedule.run_pending()
    time.sleep(60)
"""

call_all()
