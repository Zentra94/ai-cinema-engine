import unittest
from packages.video_manager.core.scrapper import Scrapper


class TestScrapper(unittest.TestCase):
    scrapper = Scrapper()

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
