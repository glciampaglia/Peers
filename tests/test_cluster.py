import subprocess
import tempfile
import time
from cluster import *

def test_execpipe():
    o,e = execpipe('echo ciao, mondo!')
    assert o == 'ciao, mondo!\n'

def test_execpipe_2():
    o,e = execpipe('echo ciao, mondo! | cat')
    assert o == 'ciao, mondo!\n'

def test_execpipe_stderr():
    script='''if __name__ == '__main__':
    import sys
    print >> sys.stderr, 'ciao, mondo!' '''
    try:
        f = tempfile.NamedTemporaryFile('w', suffix='.py')
        f.write(script)
        f.flush()
        o,e = execpipe('python %s' % f.name)
        assert e == 'ciao, mondo!\n'
    finally:
        f.close()

def TestCluster():
    @classmethod
    def setUpAll(cls):
        cls.ipcluster = subprocess.Popen('ipcluster local -n 2', shell=True)
        time.sleep(1)
        cls.parser = make_parser()
    @classmethod
    def tearDown(cls):
        cls.ipcluster.send_signal(2)
    # send touch commands to the cluster
    def setUp(self):
        self.cluster = subprocess.Popen('python cluster.py'.split(), stdin=-1)
    def tearDown(self):
        self.cluster.send_signal(2)
        for i in [1,2]:
            fn = '/tmp/%d.txt' % i
            if os.exists(fn):
                os.remove(fn)
    def test_cluster(self):
        self.cluster.communicate('touch /tmp/1.txt\ntouch /tmp/2.txt')
        assert os.exists('/tmp/1.txt')
        assert os.exists('/tmp/2.txt')
    def test_cluster_pipe(self):
        cmds = [ 'echo %(num)d | cat > /tmp/%(num)d.txt' % dict(num=i) for i in
                xrange(2) ]
        self.cluster.communicate('\n'.join(cmds))
        assert os.exists('/tmp/1.txt')
        assert os.exists('/tmp/2.txt')
