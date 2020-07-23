from glob import glob
import unittest

if __name__ == '__main__':
    # First we collect all our tests into a single TestSuite object.
    all_tests = unittest.TestSuite()

    testfiles = glob('test_*.py')
    all_test_mods = []
    for file in testfiles:
        module = file[:-3]
        mod = __import__(module)
        try:
            all_tests.addTest(mod.suite)
        except AttributeError:
            print("Could not load tests automatically from {0}.py. To test that functionality, run those tests manually as python {0}.py.".format(mod.__name__))
        
    unittest.TextTestRunner(verbosity=2).run(all_tests)
