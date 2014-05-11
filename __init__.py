import numpy as np

def sma(x, period):
    weigths = np.repeat(1.0, period) / period
    # TODO: figure out what np.convolve actually does (mathematical education)
    ma = np.convolve(x, weigths, 'valid')
    # append some nans to make the result match the length of x
    return np.hstack([np.repeat(np.nan, period - 1), ma])

def feq(a, b, e = 1e-8):
    if abs(a-b) < e:
        return True
    else:
        return False

# more accurate than feq (but possibly slower)
def feqd(a, b, e = 1e-14):
    return abs(1 - a / b) < e
    

def dump(obj):
   for attr in dir(obj):
       if not (len(attr) > 4 and attr[0:2] == '__' and attr[-2:] == '__') and hasattr( obj, attr ):
           print("{}: {}".format(attr, getattr(obj, attr)))
