import unittest

if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTests(test_loader.discover('tests'))

    runner = unittest.TextTestRunner()
    runner.run(test_suite)
