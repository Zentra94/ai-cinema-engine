import unittest
from packages.video_manager.core.translator import translate_multi_languages


class TestTranslator(unittest.TestCase):
    def test_translate_multi_languages(self):
        test_inputs = {"0": {"inputs": "hello, how are you?",
                             "expected": {"es": "¿Hola cómo estás?"}}}

        for _, item in test_inputs.items():
            raw_text = item["inputs"]
            exp = item["expected"]
            # TODO: not working - API response error
            res = translate_multi_languages(raw_text=raw_text)
            self.assertEqual(exp, res)


if __name__ == '__main__':
    unittest.main()
