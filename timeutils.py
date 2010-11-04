# coding=utf-8
# file: timeutils.py
# vim:ts=8:sw=4:sts=4

CONV_UNITS = [
        ('year', 365),
        ('week', 7),
        ('day', 1),
        ('hour', 24**-1),
        ('minute', 3600**-1),
        ('second', 86400**-1),
]

def si_str(x, abbrv=False):
    '''
    Returns the time representation in the International System of Units

    Parameters
    ----------
    x - a time length (in days)
    abbrv - if True, prints '1d' instead of '1 day', etc.
    '''
    global CONV_UNITS
    res = []
    for u,r in CONV_UNITS:
        q = x/r
        if q >= 1:
            if abbrv:
                res.append('%d%s' % (q,u[0]))
            else:
                res.append('%d %s' % (q,u+'s' if q > 1 else u))
        x = x % r
    return ', '.join(res)
