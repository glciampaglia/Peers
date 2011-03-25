import numpy as np
from numpy.testing import assert_array_equal
from rand import *

def test_binsearch():
    x = np.random.rand(10)
    x.sort()
    vals = np.linspace(x.min(),x.max(),100, endpoint=1)
    result = np.asarray([ binsearch(v, x) for v in vals ])
    expected = np.searchsorted(x, vals, 'right')
    print result[-1], expected[-1]
    assert_array_equal(result, expected)

# TODO <Fri Nov 26 10:15:21 CET 2010> should test randwpmf, possibly with a
# Chi-squared test and also checking that RMS goes to zero with N -> infty
