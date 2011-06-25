''' 
additional actions for argparse's parsers. Do not import this module
directly, use peers.utils instead.
'''

import os.path
from argparse import Action, _AppendAction, ArgumentError

__all__ = ['NNAction', 'AppendRangeAction', 'AppendTupleAction', 'AppendMaxAction']

class NNAction(Action):
    def __call__(self, parser, ns, value, option_string=None):
        if value < 0:
            raise ArgumentError(self, 'negative values not allowed: %g' % value)
        setattr(ns, self.dest, value)

class AppendRangeAction(_AppendAction):
    ''' Check a < b, nargs must be equal to 2 '''
    def __init__(self, *args, **kwargs):
        if kwargs['nargs'] != 2:
            raise ValueError('nargs must be exactly 2')
        super(AppendRangeAction, self).__init__(*args, **kwargs)
    def __call__(self, parser, ns, values, option_string=None):
        a, b = values
        if a > b:
            raise ArgumentError(self, 'not a range: %g, %g' % (a, b))
        super(AppendRangeAction, self).__call__(parser, namespace, values, 
                option_string)

class AppendTupleAction(_AppendAction):
    ''' like Append but inserts tuples '''
    def __call__(self, parser, ns, values, option_string=None):
        super(AppendTupleAction, self).__call__(parser, ns, tuple(values),
                option_string)

class AppendMaxAction(_AppendAction):
    '''
    Append to an argument up to a specified maximum number of times.

    Requires `maxlen' to be passed to ArgumentParser.add_argument or ValueError
    will be raised
    '''
    # let's do an explicit declaration of parameters since we add a new one
    def __init__(self, 
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 maxlen=None):
        if maxlen > 0:
            self.maxlen = int(maxlen)
        else:
            raise ValueError('maxlen be positive: %s' % maxlen)
        super(AppendMaxAction, self).__init__(option_strings=option_strings,
            dest=dest, nargs=nargs, const=const, default=default, type=type,
            choices=choices, required=required, help=help, metavar=metavar)
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        if len(items) < self.maxlen:
            super(AppendMaxAction, self).__call__(parser, namespace, values, 
                    option_string)
        else:
            raise ArgumentError(self, 'this option cannot be specified more than %d'
                    ' times' % self.maxlen)
