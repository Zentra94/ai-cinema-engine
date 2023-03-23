import os.path
import unittest
from configs import (PATH_DATA_MOVIES,
                     PICOVOICE_API_KEY,
                     CLIENT_SECRETS)

from packages.video_manager.core.youtube import YoutubeManager


# TODO: base_path and video_id should be create to test (due that example are in
#  .gitignore)

class TestYoutube(unittest.TestCase):
    base_path = PATH_DATA_MOVIES / "m20230323094608"
    video_id = "QRuOv-j-NRc"
    youtube_manager = YoutubeManager(base_path=base_path,
                                     caption_access_key=PICOVOICE_API_KEY,
                                     client_secret_file=CLIENT_SECRETS)

    def test_get_captions_from_audio_path(self):
        captions = self.youtube_manager.get_captions_from_audio_path()

        self.assertEqual(len(captions), 2)

    def test_upload_video(self):
        self.youtube_manager.upload_video()

        response = self.youtube_manager.response_video

        self.assertEqual(response is not None, True)

    def test_upload_captions(self):
        self.youtube_manager.upload_captions(video_id=self.video_id)

        response_dict = self.youtube_manager.response_captions

        self.assertEqual(len(response_dict) > 0, True)

    def test_upload_thumbnails(self):
        # TODO: not working - set a default cover

        cover_path = self.base_path / "paragraphs/p1_image.png"

        self.youtube_manager.upload_thumbnails(video_id=self.video_id,
                                               cover_path=cover_path),

        response = self.youtube_manager.response_thumbnails

        self.assertEqual(response is not None, True)

    def test_fit_upload(self):
        self.youtube_manager.fit_upload()

        path = self.youtube_manager.base_path / "YoutubeManager.pkl"

        self.assertEqual(os.path.isfile(path), True)


if __name__ == '__main__':
    unittest.main()
