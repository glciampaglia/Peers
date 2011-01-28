import sys
from subprocess import Popen, PIPE

def ttysize():
    '''
    Returns the size of the terminal by calling stty size, or None if the
    system call raise error from the OS.
    '''
    p = Popen('stty size'.split(), stdout=PIPE, stderr=PIPE)
    try:
        return map(int, p.communicate()[0].split())
    except OSError:
        pass


