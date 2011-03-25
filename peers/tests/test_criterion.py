from criterion import *
import numpy as np
from numpy.testing import assert_approx_equal
from nose import SkipTest
from warnings import warn

def R_factory(name, vector_type='float'):
    '''
    Parameters
    ----------
    name - string
        The full R function name, including the package name. It follows the
        rpy2 convention to replace '.' with '_' in the function name.
    vector_type - string
        R Vector type used for packing args. Either 'float' or 'int'.
    '''
    try:
        from rpy2.robjects.packages import importr
        from rpy2.robjects import FloatVector, IntVector, r
        from rpy2.rinterface import RRuntimeError
    except ImportError:
        warn('''module rpy2 is missing. 
    Please install it from: http://rpy.sourceforge.net/rpy2.html''',
                category=UserWarning)
        raise SkipTest
    
    if vector_type not in ['float', 'int']:
        raise ValueError('unknown vector_type: %s' % vector_type)
    elif vector_type == 'float':
        vector = FloatVector
    else:
        vector = IntVector
    
    idx = name.find('.')
    package_name, func_name = name[:idx], name[idx+1:]
    try:
        package = importr(package_name)
    except RRuntimeError:
        raise ImportError('rpy2: cannot import R package %s' % package_name)
    rfunc = getattr(package, func_name)
    
    def wrapper(*args, **kwargs):
        args = map(vector, map(list, args))
        return rfunc(*args, **kwargs)
    wrapper.func_name = func_name.replace('.','_')
    wrapper.func_doc = 'Wrapped version of %s:\n\n' % name + rfunc.r_repr()
    return wrapper

def test_cvmt():
    x = np.random.randn(100)
    y = np.random.randn(100)
    cvmts_test = R_factory('CvM2SL2Test.cvmts_test')
    assert_approx_equal(cvmts_test(x,y)[0], cvmt(x,y))

def test_auc():
    x = np.arange(1,100, dtype=float)
    y = x + 1.
    assert_approx_equal(auc(x,y),1.)
    assert_approx_equal(auc(x,x),0.)
    assert_approx_equal(c_auc(x,y),1.)
    assert_approx_equal(c_auc(x,x),0.)

def test_auc_cython():
    x = np.random.randn(100)
    y = np.random.randn(100)
    assert_approx_equal(c_auc(x,y), auc(x,y))

# TODO <Thu Nov 25 23:12:10 CET 2010> must find an implementation of the
# Chi-squared test statistic for two samples
def test_chisq2sam():
    raise SkipTest 
    x = np.random.randn(100)
    y = np.random.randn(100)
    r = np.min([x,y]), np.max([x,y])
    xh,_ = np.histogram(x, bins=10, range=r)
    yh,_ = np.histogram(y, bins=10, range=r)
    ch, pval = chisq_2sam(xh, yh)
    r_chisq_test = R_factory('stats.chisq_test')    # <-- XXX not the right one!
    ch1, pval1 = r_chisq_test(xh, yh)               # <-- XXX not the right one!
    assert_approx_equal(ch, ch1)
    assert_approx_equal(pval, pval1)

# This test fails occasionally due to poor precision of the R function
def test_adk():
    r_adk = R_factory('adk.adk_test')
    x = np.random.randn(100)
    y = np.random.randn(100)
    out = dict(r_adk(x,y).iteritems())
    expected = out['adk'][0]
    # adk::adk_test return values has only 5 significant digits
    assert_approx_equal(adk(x,y, std=True), expected, significant=5)
    assert_approx_equal(c_adk([x,y], 1), expected, significant=5)
    assert_approx_equal(adk(x,y), c_adk([x,y], 0))
    assert_approx_equal(adk(x,y, std=True), c_adk([x,y], 1))

def test_adk_stability():
    x = np.random.randn(2,20000) # Large N
    y = np.random.randn(1000,10) # Large k
    assert_approx_equal(adk(*x), c_adk(x, 0))
    assert_approx_equal(adk(*y), c_adk(y, 0))
