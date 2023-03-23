import translators as ts


def translate_multi_languages(raw_text, input_language: str = "en",
                              output_languages=None):
    """

    Args:
        raw_text ():
        input_language ():
        output_languages ():

    Returns:

    """
    if output_languages is None:
        output_languages = ["es"]

    multi_languages = {}

    for l in output_languages:
        multi_languages[l] = ts.google(raw_text, from_language=input_language,
                                       to_language=l)

    return multi_languages
