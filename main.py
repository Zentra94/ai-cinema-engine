import pathlib
import sys

from packages.video_manager.core.youtube import YoutubeManager
from packages.video_manager.core.screenwriter import ScreenWriter
from configs import (OPENAI_API_KEY, REPLICATE_API_KEY,
                     PICOVOICE_API_KEY, CLIENT_SECRETS, GCP_SA_KEY,
                     PATH_DATA_MOVIES_MUSIC, PATH_DATA_MOVIES)

from typing import Union, List

DEFAULT_MUSIC_PATH = PATH_DATA_MOVIES_MUSIC / "calm-chill-beautiful.mp3"


# TODO: add trends scrapper to automatize the prompt title input

def main(prompt: str,
         music_path: Union[pathlib.Path, str] = DEFAULT_MUSIC_PATH,
         target_languages: List[str] = None,
         cover_path: Union[pathlib.Path, str] = None) -> None:
    """Executes an end-to-end process of creating and uploading a video.

    Args:
        prompt (str): The title or prompt of the video. Example: "create a unique title
            of a youtube video of 3 curios facts about pandas bears".
        music_path (pathlib.Path or str, optional): The path to the audio file to be
            used as background music. Defaults to DEFAULT_MUSIC_PATH.
        target_languages (List[str], optional): A list of language codes in which the
            video should be captioned. Defaults to None, which means no captions will be
            generated.
        cover_path (Union[pathlib.Path, str], optional): The path to the image file to
            be used as the video's cover.Defaults to None, which means no custom cover
            image will be used.
    """

    screen_writer = ScreenWriter(
        base_path=PATH_DATA_MOVIES,
        gcp_sa_key=GCP_SA_KEY,
        open_ai_key=OPENAI_API_KEY,
        replicate_api_key=REPLICATE_API_KEY)

    screen_writer.fit(title_prompt=prompt,
                      music_path=music_path)

    base_path = screen_writer.base_path

    youtube_manager = YoutubeManager(base_path=base_path,
                                     caption_access_key=PICOVOICE_API_KEY,
                                     client_secret_file=CLIENT_SECRETS)

    youtube_manager.fit_upload(target_languages=target_languages,
                               cover_path=cover_path)


if __name__ == '__main__':

    main(*sys.argv[1:])
