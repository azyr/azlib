import re
import numpy as np
import pytz
import pandas as pd
import tzlocal
from datetime import datetime


def get_localtz():
    """Return the local timezone."""
    return pytz.timezone(tzlocal.get_localzone().zone)


# Deprecated, use pd.Timestamp instead
# def local_to_utc(dt, is_dst=False):
#     l_tz = get_localtz()
#     dt = l_tz.normalize(l_tz.localize(dt, is_dst))
#     return dt.astimezone(pytz.utc)


# Deprecated, use pd.Timestamp instead
# def utc_to_local(dt):
#     dt = pytz.utc.normalize(pytz.utc.localize(dt))
#     return dt.astimezone(get_localtz())

def utcnow():
    """Return UTC time now as pd.Timestamp."""
    return pd.Timestamp(datetime.utcnow(), tz='UTC')

def estnow():
    """Return US/Eastern time now as pd.Timestamp."""
    return pd.Timestamp(datetime.utcnow(), tz='UTC').tz_convert("US/Eastern")


def sma(x, period):
    """Calculate simple moving average. Return np.ndarray.

    Arguments:
    x       -- input vector
    period  -- sma length
    """
    weigths = np.repeat(1.0, period) / period
    # TODO: figure out what np.convolve actually does (mathematical education)
    ma = np.convolve(x, weigths, 'valid')
    # append some nans to make the result match the length of x
    return np.hstack([np.repeat(np.nan, period - 1), ma])


def feq(a, b, e=1e-8):
    """Test floats for equality.

    Arguments:
    a -- first float
    b -- second float
    e -- maximum allowed difference
    """
    if abs(a-b) < e:
        return True
    else:
        return False

def feqd(a, b, e=1e-14):
    """Test floats for equality.

    This is a little bit more accurate and general than feq() but possibly slower.

    Arguments:
    a -- first float
    b -- second float
    e -- maximum allowed 'error'
    """
    return abs(1 - a / b) < e
    
def chunks(l, n):
    """Iterator to return n-sized chunks from l.

    Arguments:
    l -- input vector
    n -- size of a chunk
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

def globber(s, choices):
    """Imitate the globber behaviour of bash. Return list of matching choices.

    Arguments:
    s       -- string to match against choices
    choices -- all available choices
    """
    candidates = []
    s_re = s
    # escape some characters to get the correct behaviour
    s_re = s_re.replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')
    s_re = s_re.replace('^', '\\^').replace('$', '\\$').replace('|', '\\|').replace('\\', '\\\\')
    s_re = s_re.replace('.', '\\.')
    s_re = s_re.replace("*", ".*").replace("?", ".")
    # word boundaries
    s_re = "^" + s_re + "$"
    for x in choices:
        if re.search(s_re, x):
            candidates.append(x)
    return candidates

def periods_per_year(p):
    """Return average number periods per year.

    Two overloads:

    periods_per_year(series):
    Return number of periods in a year for the series.

    periods_per_year(p):
    Return number of periods in a year for period p (pandas syntax).
    """
    if type(p) == str:
        return len(pd.date_range('2000-01-01', '2010-01-01', freq=p)) / 10.0
    elif type(p) == pd.Series:
        if p.index.freq:
            return periods_per_year(p.index.freq.name)
        nyears = (p.index[-1] - p.index[0]).days / 365.25
        return len(p) / nyears
    else:
        raise Exception("p has to be either str or pd.Series")

def get_periodicity(series, p):
    """Return number of periods on average in a series of a given length p.

    Arguments:
    series -- pd.Series to analyze
    p      -- str representing the period length, use pandas notation
    """

    days_per_year = 365.25
    nyears = period = (series.index[-1] - series.index[0]).days / days_per_year
    return nyears * periods_per_year(p)

def annual_volatility(changes, period=-1):
    """Calculate annualized volatility (standard deviation)

    Arguments:
    changes -- asset returns
    period          -- periodicity of the asset returns (optional, not neccessary
                       nor recommended when changes is pd.Series)
    """
    if type(changes) == pd.Series and period == -1: 
        period = periods_per_year(changes)
    if period == -1:
        raise Exception("Period parameter required when changes is not pd.Series")
    return np.std(changes) * np.sqrt(period)

def sharpe_ratio(changes, rfrate_yearly, period=-1):
    """Calculate Sharpe ratio.

    Arguments:
    changes         -- asset returns
    rfrate_yearly   -- risk-free rate (yearly), for example 1.04
    period          -- periodicity of the asset returns (optional, not neccessary
                       nor recommended when changes is pd.Series)
    """
    if type(changes) == pd.Series and period == -1: 
        period = periods_per_year(changes)
    if period == -1:
        raise Exception("Period parameter required when changes is not pd.Series")
    rfrate_per_period = rfrate_yearly ** (1/period) - 1
    excess_returns = changes - rfrate_per_period
    std_excess_return = np.std(excess_returns)
    avg_excess_return = np.mean(excess_returns)
    dailysharpe = avg_excess_return / std_excess_return
    annualized = dailysharpe * np.sqrt(period)
    return annualized

def sharpe_ratio_gacr(changes, rfrate_yearly, period=-1):
    """Calculate Sharpe ratio (GACR method)

    Calculate Sharpe-ratio using daily GACR minus daily risk-free rate as a nominator
    instead of average daily excess return. Normally gives more sensible results than
    the standard method (especially on daily returns).

    Arguments:
    changes         -- asset returns
    rfrate_yearly   -- risk-free rate (yearly), for example 1.04
    period          -- periodicity of the asset returns (optional, not neccessary
                       nor recommended when changes is pd.Series)
    """
    if type(changes) == pd.Series and period == -1: 
        period = periods_per_year(changes)
    if period == -1:
        raise Exception("Period parameter required when changes is not pd.Series")
    endprice = np.cumprod(changes + 1)[-1]
    g = gacr(1, endprice, len(changes)) - 1
    rfrate_per_period = rfrate_yearly ** (1/period) - 1
    excess_returns = changes - rfrate_per_period
    std_excess_return = np.std(excess_returns)
    dailysharpe = (g - rfrate_per_period) / std_excess_return
    annualized = dailysharpe * np.sqrt(period)
    return annualized


def sortino_ratio(changes, trate_yearly, period=-1):
    """Calculate Sortino ratio.

    Arguments:
    changes         -- asset returns
    rfrate_yearly   -- target rate (yearly), for example 1.04
    period          -- periodicity of the asset returns (optional, not neccessary
                       nor recommended when changes is pd.Series)
    """
    if type(changes) == pd.Series and period == -1: 
        period = periods_per_year(changes)
    if period == -1:
        raise Exception("Period parameter required when changes is not pd.Series")
    trate_per_period = trate_yearly ** (1/period) - 1
    excess_returns = changes - trate_per_period
    avg_excess_return = np.mean(excess_returns)
    minus_exc_returns_squared = excess_returns[excess_returns < 0] ** 2
    downside_risk = np.sqrt(np.sum(minus_exc_returns_squared) / len(excess_returns))
    dailysortino = avg_excess_return / downside_risk
    annualized = dailysortino * np.sqrt(period)
    return annualized


def sortino_ratio_gacr(changes, trate_yearly, period=-1):
    """Calculate Sortino ratio (GACR method)

    Calculate Sortino-ratio using daily GACR minus daily risk-free rate as a nominator
    instead of average daily excess return. Normally gives more sensible results than
    the standard method (especially on daily returns).

    Arguments:
    changes         -- asset returns
    rfrate_yearly   -- risk-free rate (yearly), for example 1.04
    period          -- periodicity of the asset returns (optional, not neccessary
                       nor recommended when changes is pd.Series)
    """
    if type(changes) == pd.Series and period == -1: 
        period = periods_per_year(changes)
    if period == -1:
        raise Exception("Period parameter required when changes is not pd.Series")
    endprice = np.cumprod(changes + 1)[-1]
    g = gacr(1, endprice, len(changes)) - 1
    trate_per_period = trate_yearly ** (1/period) - 1
    excess_returns = changes - trate_per_period
    minus_exc_returns_squared = excess_returns[excess_returns < 0] ** 2
    downside_risk = np.sqrt(np.sum(minus_exc_returns_squared) / len(excess_returns))
    dailysortino = (g - trate_per_period) / downside_risk
    annualized = dailysortino * np.sqrt(period)
    return annualized

def gacr(*args):
    """Return GACR.

    Two overloads:

    gacr(series, p):
    series -- pd.Series for input (containing asset price or cumulative returns)
    p      -- periodicity to use for calculation (pandas notation)

    gacr(start, end, period):
    start  -- starting balance/price
    end    -- ending balance/price
    period -- how many periods to use for calculation
    """
    if len(args) == 2:
        series = args[0]
        p = args[1]
        if type(p) == int or type(p) == float:
            period = p
        else:
            period = get_periodicity(series, p)
        return np.e**(np.log(series[-1]/series[0])/period)
    if len(args) == 3:
        start = args[0]; end = args[1]; period = args[2]
        return np.e**(np.log(end/start)/period)
    else:
        raise Exception("Invalid number of arguments: {}".format(len(args)))


def decorate_str(input_s, method='underline', char='=', newln=True):
    """Decorate input_s str.

    Arguments:
    input_s   -- str to decorate
    method    -- decoration method, options: 'underline', 'over_under', 'surround'
    char      -- char to use for decoration
    newln     -- put newline to the end of the output
    """
    s = ""
    if method == "underline":
        s += input_s + '\n'
        s += len(input_s) * char
        if newln:
            s += '\n'
        return s
    if method == "over_under":
        s += len(input_s) * char + '\n'
        s += input_s + '\n'
        s += len(input_s) * char
        if newln:
            s += '\n'
        return s
    if method == "surround":
        s += (len(input_s) + 6) * char + '\n'
        s += char * 2 + ' ' + input_s + ' ' + char * 2 + '\n'
        s += (len(input_s) + 6) * char
        if newln:
            s += '\n'
        return s
    raise Exception("Unknown method: {}".format(method))




