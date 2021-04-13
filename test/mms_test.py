import unittest

# https://docs.google.com/document/d/1xGgBpFzCcwiRUU_l9_Y_q0nTeIcCG1-GvXWnBZhfcGs/edit


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_something2(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
