#
import json
import numpy
import pandas
from scipy import stats

import sqlalchemy
from matplotlib import pyplot

from scipy.stats import kstest, shapiro
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
#


#
def get_worst_flag(x, name):
    _targets = list(x.keys())
    _flags = [x[t][name] for t in _targets]
    if 'red' in _flags:
        return 'red'
    elif 'yellow' in _flags:
        return 'yellow'
    else:
        return 'green'


def select_result(mre):
    _models = list(mre.keys())
    _statuses = [mre[m]['statuses'] for m in _models]
    _targets = [get_worst_flag(x, 'target') for x in _statuses]
    _errors = [get_worst_flag(x, 'errors') for x in _statuses]

    if numpy.unique(_targets).shape[0] != 1:
        if 'green' in _targets:
            result_ix = _targets.index('green')
        elif 'yellow' in _targets:
            result_ix = _targets.index('yellow')
        else:
            result_ix = _targets.index('red')
    elif numpy.unique(_errors).shape[0] != 1:
        if 'green' in _errors:
            result_ix = _errors.index('green')
        elif 'yellow' in _errors:
            result_ix = _errors.index('yellow')
        else:
            result_ix = _errors.index('red')

    else:
        result_ix = 0

    return _models[result_ix]


class Controls:
    def __init__(self):

        self.smape = None
        self.smape__red = 0.050
        self.smape__green = 0.025

        self.normal = None
        self.normal__red = 0.01
        self.normal__green = 0.05
        self.autocorrelated = None
        self.autocorrelated__red = 0.10
        self.autocorrelated__green = 0.05
        self.stationary = None
        self.stationary__red = 0.05
        self.stationary__green = 0.01
        self.homoscedastic = None
        self.homoscedastic__red = 0.01
        self.homoscedastic__green = 0.05
        self.zero_mean = None
        self.zero_mean__red = 0.01
        self.zero_mean__green = 0.05

        self.ill_posed = None
        self.ill_posed__red = 10
        self.ill_posed__green = 5

    @property
    def smape__flag(self):
        if self.smape > self.smape__red:
            return 'red'
        elif self.smape < self.smape__green:
            return 'green'
        else:
            return 'yellow'

    @property
    def normal__flag(self):
        if self.normal < self.normal__red:
            return 'red'
        elif self.normal > self.normal__green:
            return 'green'
        else:
            return 'yellow'

    @property
    def autocorrelated__flag(self):
        if self.autocorrelated > self.autocorrelated__red:
            return 'red'
        elif self.autocorrelated < self.autocorrelated__green:
            return 'green'
        else:
            return 'yellow'

    @property
    def stationary__flag(self):
        if self.stationary > self.stationary__red:
            return 'red'
        elif self.stationary < self.stationary__green:
            return 'green'
        else:
            return 'yellow'

    @property
    def homoscedastic__flag(self):
        if self.homoscedastic < self.homoscedastic__red:
            return 'red'
        elif self.homoscedastic > self.homoscedastic__green:
            return 'green'
        else:
            return 'yellow'

    @property
    def zero_mean__flag(self):
        if self.zero_mean < self.zero_mean__red:
            return 'red'
        elif self.zero_mean > self.zero_mean__green:
            return 'green'
        else:
            return 'yellow'

    @property
    def ill_posed__flag(self):
        if self.ill_posed > self.ill_posed__red:
            return 'red'
        elif self.ill_posed < self.ill_posed__green:
            return 'green'
        else:
            return 'yellow'

    @property
    def warns(self):
        return self.ill_posed__flag

    @property
    def target(self):
        return self.smape__flag

    @property
    def errors(self):
        alls = [self.normal__flag, self.autocorrelated__flag, self.stationary__flag, self.homoscedastic__flag, self.zero_mean__flag]
        if 'red' in alls:
            return 'red'
        elif 'yellow' in alls:
            return 'yellow'
        else:
            return 'green'

    def check(self, X, Y_true, Y_hat):

        errors = Y_true - Y_hat

        self.smape = SMAPE(y_true=Y_true, y_hat=Y_hat)

        self.normal = stats.shapiro(errors)[1]
        self.autocorrelated = numpy.abs(acf(errors, fft=True)[1:]).max()
        self.stationary = adfuller(errors)[1]
        self.homoscedastic = het_breuschpagan(resid=errors, exog_het=X)[1]
        self.zero_mean = stats.ttest_1samp(a=errors, popmean=0.0)[1]

        self.ill_posed = numpy.max([variance_inflation_factor(exog=X, exog_idx=j) for j in range(X.shape[1])])


def _RMSE(y_true, y_hat):
    value = (((y_true - y_hat) ** 2).sum() / y_true.shape[0]) ** 0.5
    return value


def _MAE(y_true, y_hat):
    value = ((numpy.abs(y_true - y_hat)).sum() / y_true.shape[0])
    return value


def _MAPE(y_true, y_hat):
    value = (numpy.abs(y_true - y_hat) / numpy.abs(y_true)).sum() / y_true.shape[0]
    return value


def _SMAPE(y_true, y_hat):
    value = (numpy.abs(y_true - y_hat) / ((y_true + y_hat) / 2)).sum() / y_true.shape[0]
    return value


def _LCH(y_true, y_hat):
    value = numpy.log(numpy.cosh(y_true - y_hat)).sum()
    return value


def RMSE(y_true, y_hat):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_hat.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_hat[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return _RMSE(y_true=y_true_, y_hat=y_pred_)


def MAE(y_true, y_hat):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_hat.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_hat[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return _MAE(y_true=y_true_, y_hat=y_pred_)


def MAPE(y_true, y_hat):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_hat.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_hat[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return _MAPE(y_true=y_true_, y_hat=y_pred_)


def SMAPE(y_true, y_hat):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_hat.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_hat[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return _SMAPE(y_true=y_true_, y_hat=y_pred_)


def LCH(y_true, y_hat):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_hat.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_hat[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return _LCH(y_true=y_true_, y_hat=y_pred_)


def KS(y, dist):
    Z = numpy.concatenate([y.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_ = y[nan_mask]
    if y_.shape[0] == 0:
        return numpy.nan
    else:
        if y_.shape[0] != y.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y.shape[0] - y_.shape[0]))
        return kstest(y_, dist)[1]


def KS2(y_true, y_hat):
    Z = numpy.concatenate([y_true.reshape(-1, 1), y_hat.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_true_, y_pred_ = y_true[nan_mask], y_hat[nan_mask]
    if y_true_.shape[0] == 0:
        return numpy.nan
    else:
        if y_true_.shape[0] != y_true.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y_true.shape[0] - y_true_.shape[0]))
        return kstest(y_true_, y_pred_)[1]


def SW(y):
    Z = numpy.concatenate([y.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_ = y[nan_mask]
    if y_.shape[0] == 0:
        return numpy.nan
    else:
        if y_.shape[0] != y.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y.shape[0] - y_.shape[0]))
        return shapiro(y_)[1]


def ADF(y, regression='nc'):
    Z = numpy.concatenate([y.reshape(-1, 1)], axis=1)
    Z = numpy.array(Z, dtype=numpy.float32)
    Z[Z == numpy.inf] = numpy.nan
    Z[Z == -numpy.inf] = numpy.nan
    nan_mask = ~pandas.isna(Z).any(axis=1)
    y_ = y[nan_mask]
    if y_.shape[0] == 0:
        return numpy.nan
    else:
        if y_.shape[0] != y.shape[0]:
            print('MAE: the sample contains NaNs, they were dropped\tN of dropped NaNs: {0}'.format(y.shape[0] - y_.shape[0]))
        return adfuller(x=y_, regression=regression)[1]


def quality_table(y_true_train, y_hat_train, y_true_test, y_hat_test):

    rmse_train, rmse_test = RMSE(y_true=y_true_train, y_hat=y_hat_train), RMSE(y_true=y_true_test, y_hat=y_hat_test)
    mae_train, mae_test = MAE(y_true=y_true_train, y_hat=y_hat_train), MAE(y_true=y_true_test, y_hat=y_hat_test)
    mape_train, mape_test = MAPE(y_true=y_true_train, y_hat=y_hat_train), MAPE(y_true=y_true_test, y_hat=y_hat_test)
    smape_train, smape_test = SMAPE(y_true=y_true_train, y_hat=y_hat_train), SMAPE(y_true=y_true_test, y_hat=y_hat_test)
    logcosh_train, logcosh_test = LCH(y_true=y_true_train, y_hat=y_hat_train), LCH(y_true=y_true_test, y_hat=y_hat_test)

    ks_train, ks_test = KS2(y_true_train, y_hat_train), KS2(y_true_test, y_hat_test)
    adf_train, adf_test = ADF(y=(y_true_train - y_hat_train), regression='nc'), ADF(y=(y_true_test - y_hat_test), regression='nc')
    normal_train, normal_test = SW(y_true_train - y_hat_train), SW(y_true_test - y_hat_test)
    laplace_train, laplace_test = KS((y_true_train - y_hat_train), 'laplace'), KS((y_true_test - y_hat_test), 'laplace')

    indices = ['train', 'test']
    columns = ['rmse', 'mae', 'mape', 'smape', 'logcosh',
               'KS_distance', 'ADF_on_errors', 'SHAPIRO_errors', 'KS_LAPLACE_errors']

    values = [[rmse_train, mae_train, mape_train, smape_train, logcosh_train,
               ks_train, adf_train, normal_train, laplace_train],
              [rmse_test, mae_test, mape_test, smape_test, logcosh_test,
               ks_test, adf_test, normal_test, laplace_test]]

    table = pandas.DataFrame(columns=columns, index=indices, data=values)
    return table


def quality_plots(y_true_train, y_hat_train, y_true_test, y_hat_test):

    fig, ax = pyplot.subplots(5, 2)

    # true series vs hat series
    ax[0, 0].plot(range(y_true_train.shape[0]), y_true_train, 'black', range(y_hat_train.shape[0]), y_hat_train, 'navy')
    ax[0, 1].plot(range(y_true_test.shape[0]), y_true_test, 'black', range(y_hat_test.shape[0]), y_hat_test, 'navy')

    # values hist
    ax[1, 0].hist(y_true_train, bins=100, density=True, color='black')
    ax[1, 0].hist(y_hat_train, bins=100, density=True, color='navy')
    ax[1, 1].hist(y_true_test, bins=100, density=True, color='black')
    ax[1, 1].hist(y_hat_test, bins=100, density=True, color='navy')

    # errors series
    ax[2, 0].plot(range(y_true_train.shape[0]), y_true_train - y_hat_train, 'orange')
    ax[2, 1].plot(range(y_true_test.shape[0]), y_true_test - y_hat_test, 'orange')

    # errors hist
    ax[3, 0].hist(y_true_train - y_hat_train, bins=100, density=True, color='blue')
    ax[3, 1].hist(y_true_test - y_hat_test, bins=100, density=True, color='orange')

    # true values QQ vs hat series QQ
    qs = numpy.linspace(start=0, stop=1, endpoint=False)[1:]
    y_true_train_q, y_hat_train_q = numpy.quantile(a=y_true_train, q=qs), numpy.quantile(a=y_hat_train, q=qs)
    y_true_test_q, y_hat_test_q = numpy.quantile(a=y_true_test, q=qs), numpy.quantile(a=y_hat_test, q=qs)
    ax[4, 0].plot(y_true_train_q, y_hat_train_q, color='navy', marker='o')
    ax[4, 0].plot(y_true_train_q, y_true_train_q, color='black')
    ax[4, 1].plot(y_true_test_q, y_hat_test_q, color='navy', marker='o')
    ax[4, 1].plot(y_true_test_q, y_true_test_q, color='black')


def get_connection():

    with open('C:/Users/MainUser/Desktop/users.json') as f:
        users = json.load(f)

    user, password = users['researcher']['user'], users['researcher']['password']

    with open('C:/Users/MainUser/Desktop/servers.json') as f:
        servers = json.load(f)

    host, port = servers['GOLA']['host'], servers['GOLA']['port']

    dbname = 'experiments'

    connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(user, password, host, port, dbname)
    engine = sqlalchemy.create_engine(connection_string)
    connection = engine.connect()

    return connection


def reported_n_muted(report_measures, report_values):

    connection = get_connection()

    report_measures.to_sql(name='report_measures', con=connection, if_exists='append', index=False)
    report_values.to_sql(name='report_values', con=connection, if_exists='append', index=False)


def get_all_stuff():

    connection = get_connection()

    report_measures = pandas.read_sql(sql="SELECT * FROM report_measures;", con=connection)
    report_values = pandas.read_sql(sql="SELECT * FROM report_values;", con=connection)

    return report_measures, report_values


