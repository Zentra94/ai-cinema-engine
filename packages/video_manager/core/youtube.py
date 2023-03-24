import pvleopard
from packages.video_manager.core.translator import translate_multi_languages
import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request


def _fmt_response(response):
    # TODO: Change 'PLACE_HOLDER' to a image class compatible with to_json() method

    for attr_name in dir(response):
        attr = response.__getattribute__(attr_name)
        if isinstance(attr, bytes):
            response.__setattr__(attr_name, "PLACE_HOLDER")

    if hasattr(response, "to_json"):
        return response.to_json()

    return response


def _second_to_timecode(x: float) -> str:
    hour, x = divmod(x, 3600)
    minute, x = divmod(x, 60)
    second, x = divmod(x, 1)
    millisecond = int(x * 1000.)

    return '%.2d:%.2d:%.2d,%.3d' % (hour, minute, second, millisecond)


def _to_srt(
        words,
        endpoint_sec: float = 1.,
        length_limit=16) -> str:
    def _helper(end: int) -> None:
        lines.append("%d" % section)
        lines.append(
            "%s --> %s" %
            (
                _second_to_timecode(words[start].start_sec),
                _second_to_timecode(words[end].end_sec)
            )
        )
        lines.append(' '.join(x.word for x in words[start:(end + 1)]))
        lines.append('')

    lines = list()
    section = 0
    start = 0
    for k in range(1, len(words)):
        if ((words[k].start_sec - words[k - 1].end_sec) >= endpoint_sec) or \
                (length_limit is not None and (k - start) >= length_limit):
            _helper(k - 1)
            start = k
            section += 1
    _helper(len(words) - 1)

    return '\n'.join(lines)


def _create_service(base_path, client_secret_file, api_name, api_version, scopes):
    print(client_secret_file, api_name, api_version, scopes, sep='-')
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    print(scopes)

    cred = None

    pickle_file = base_path / f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, scopes)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, 'service created successfully')
        return service
    except Exception as e:
        print('Unable to connect.')
        print(e)
        return None


class YoutubeManager:
    """A class for managing the creation and uploading of YouTube videos, including
    captions & thumbnails."""

    def __init__(self,
                 base_path,
                 caption_access_key,
                 client_secret_file,
                 api_name="youtube",
                 api_version="v3",
                 scopes=None):

        """Initialize YoutubeManager.

        Args:
            base_path (pathlib.Path): The base directory for the project.
            caption_access_key (str): The access key for the captioning engine.
            client_secret_file (str): The path to the client secret file for the YouTube
                API.
            api_name (str): The name of the YouTube API.
            api_version (str): The version of the YouTube API.
            scopes (list of str): The scopes required for the YouTube API.

        Raises:
            FileNotFoundError: If the client secret file or ScreenWriter.pkl file cannot
                be found.
        """

        self.response_video = None
        self.response_captions = {}
        self.response_thumbnails = None
        self.video_id = None

        if scopes is None:
            scopes = ["https://www.googleapis.com/auth/youtube.upload",
                      "https://www.googleapis.com/auth/youtube.force-ssl",
                      "https://www.googleapis.com/auth/youtubepartner"]

        self.caption_engine = pvleopard.create(access_key=caption_access_key)
        self.original_language = "en"
        self.base_path = base_path
        with open(self.base_path / "ScreenWriter.pkl", 'rb') as f:
            self.screen_writer = pickle.load(f)

        self.youtube_service = _create_service(base_path=base_path,
                                               client_secret_file=client_secret_file,
                                               api_name=api_name,
                                               api_version=api_version,
                                               scopes=scopes)

    def get_captions_from_audio_path(self,
                                     audio_path=None,
                                     target_languages=None):

        captions = {}

        if target_languages is None:
            target_languages = ["es"]
        if audio_path is None:
            audio_path = str(self.base_path / "speech.wav")

        transcript, words = self.caption_engine.process_file(audio_path=audio_path)

        original_captions = _to_srt(words=words)

        captions[self.original_language] = original_captions

        for language in target_languages:
            new_captions = []
            for i, elemt in enumerate(original_captions.split("\n")):
                if i == 2 or (i - 2) % 4 == 0:
                    elemt_fmt = translate_multi_languages(
                        raw_text=elemt,
                        output_languages=[language])[language]
                else:
                    elemt_fmt = elemt

                new_captions.append(elemt_fmt)

            captions[language] = "\n".join(new_captions)

        return captions

    def upload_video(self):

        title = self.screen_writer.title

        media_file = MediaFileUpload(str(self.base_path / "final.mp4"))

        description = self.screen_writer.description

        tags = self.screen_writer.tags

        request_body = {
            "snippet": {
                "shorts": False,
                "title": title,
                "description": description,
                "tags": tags
            },
            "status": {
                "privecyStatus": "public",
                "selfDeclareMadeForKids": False
            }
        }

        self.response_video = _fmt_response(
            response=self.youtube_service.videos().insert(
                part="snippet,status",
                body=request_body,
                media_body=media_file).execute())

        video_id = self.response_video["id"]

        print("VideoId {} load successfully".format(video_id))

        self.video_id = video_id

    def upload_captions(self, target_languages=None,
                        audio_path=None,
                        video_id=None):

        if video_id is None:
            video_id = self.video_id

        captions = self.get_captions_from_audio_path(target_languages=target_languages,
                                                     audio_path=audio_path)

        for language, caption in captions.items():
            caption_path = self.base_path / "captions_{}.srt".format(language)

            with open(caption_path, "w") as of:
                of.write(caption)

            caption_file = MediaFileUpload(caption_path,
                                           mimetype='application/octet-stream')

            request_captions = {
                "snippet": {
                    "language": language,
                    "videoId": video_id,
                    "name": "Caption {}".format(language),
                    "isDraft": False
                }
            }

            self.response_captions[language] = _fmt_response(
                response=self.youtube_service.captions().insert(
                    part="snippet",
                    body=request_captions,
                    media_body=caption_file).execute())

    def upload_thumbnails(self, video_id=None,
                          cover_path=None):

        if video_id is None:
            video_id = self.video_id

        if cover_path is None:
            cover_path = self.base_path / "paragraphs/p0_image.png"

        media_body = MediaFileUpload(str(cover_path),
                                     mimetype="image/png")

        response = self.youtube_service.thumbnails().set(videoId=video_id,
                                                         media_body=media_body)
        self.response_thumbnails = _fmt_response(response=response)

    def fit_upload(self, target_languages=None, audio_path=None, cover_path=None):

        self.upload_video()

        self.upload_captions(target_languages=target_languages,
                             audio_path=audio_path)

        self.upload_thumbnails(cover_path=cover_path)

        path_youtube_manager = self.base_path / "YoutubeManager.pkl"

        self.youtube_service.close()

        # TODO: change this data types to one that could be pickled.

        self.youtube_service = "PLACE_HOLDER"
        self.caption_engine = "PLACE_HOLDER"

        with open(path_youtube_manager, 'wb') as of:
            pickle.dump(self, of)

        print("Youtube Manager saved at {}".format(path_youtube_manager))
