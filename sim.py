#!/usr/bin/env python
#coding=utf-8

# Uber command script. TODO Checklist:
# 1. Should be used like manage.py in a Django project. 
# 2. Should user parse_known_args and get the name of the script requested.
# 3. Should import make_parser and main from the actual script requested, parse
#    the remaining args obtained at point 2) and pass the resulting namespace to
#    main (but what happens if main also accepts another argument? Make sure all
#    scripts have uniform calling conventions)
# 4. The main function should handle exceptions as it prefers
# 5. Points 1-4) should be implemented in a single function (called 'main' as
#    well) The module should export that function. 
# 6. The distutils script should auto-generate an executable script that imports
#    from this script the main function of point 5) and runs it. This
#    auto-generated script is the one the user calls.
# 7. Things to take care of: the Python PATH, since we are importing modules.

# TODO <Thu Dec  2 12:10:09 CET 2010> find a better name than sim.py 

from argparse import ArgumentParser

def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
            'command',
            choices=['lhd', 'grid', 'jobs', 'cluster',])

if __name__ == '__main__':
    parser = make_parser()
    ns, args = parser.parse_known_args()

