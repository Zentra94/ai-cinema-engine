import sys

from packages.video_manager.core.youtube import YoutubeManager
from packages.video_manager.core.screenwriter import ScreenWriter
from configs import (OPENAI_API_KEY, STABLEDIFFUSION_API_KEY,
                     PICOVOICE_API_KEY, CLIENT_SECRETS, GCP_SA_KEY,
                     PATH_DATA_MOVIES_MUSIC, PATH_DATA_MOVIES)


def main(prompt,
         music_path=PATH_DATA_MOVIES_MUSIC / "calm-chill-beautiful.mp3",
         target_languages=None,
         cover_path=None):
    screen_writer = ScreenWriter(
        base_path=PATH_DATA_MOVIES,
        gcp_sa_key=GCP_SA_KEY,
        open_ai_key=OPENAI_API_KEY,
        replicate_api_key=STABLEDIFFUSION_API_KEY)

    screen_writer.fit(title_prompt=prompt,
                      music_path=music_path)

    base_path = screen_writer.base_path

    youtube_manager = YoutubeManager(base_path=base_path,
                                     caption_access_key=PICOVOICE_API_KEY,
                                     client_secret_file=CLIENT_SECRETS)

    youtube_manager.fit_upload(target_languages=target_languages,
                               cover_path=cover_path)


if __name__ == '__main__':
    # TODO: test e2e

    print(*sys.argv[1:])

    # main(*sys.argv[1:])
