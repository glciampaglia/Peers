''' fast Cython implementation '''

# will import only __all__
from peers import *
from _cpeers import simulate 
import sys

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        ns = Arguments(ns) # will check argument values here
        print >> sys.stderr, ns
        if not ns.dry_run:
            if ns.profile:
                import pstats, cProfile
                fn = ns.profile_file or __file__ + ".prof"
                cProfile.runctx('simulate(ns)', globals(), locals(), fn)
                stats = pstats.Stats(fn)
                stats.strip_dirs().sort_stats("time").print_stats()
            else:
                prng, users, pages = simulate(ns)
    except:
        ty, val, tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            name = ty.__name__
            print >> sys.stderr, '\n%s: %s\n' % (name, val)
    finally:
        if ns.info_file:
            ns.info_file.close()
