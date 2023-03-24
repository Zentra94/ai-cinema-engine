import re
from datetime import datetime
import google.cloud.texttospeech as tts
from google.oauth2 import service_account

from moviepy.editor import (AudioFileClip,
                            ImageClip,
                            VideoFileClip,
                            concatenate_videoclips,
                            CompositeAudioClip)
import openai
import replicate
import requests
from PIL import Image
from io import BytesIO
import os
from natsort import natsorted
from rake_nltk import Rake
import pickle

REPLICATE_VERSION = "f178fa7a1ae43a9a9af01b833b9d2ecf97b1bcb0acfd2dc5dd04895e042863f1"
REPLICATE_ENGINE = "stability-ai/stable-diffusion"


def _prompt_engineering(prompt,
                        fmmt_phrase="Concept art of:"):
    clean_prompt = " ".join(re.sub('[^a-zA-Z ]+', '', prompt).split())

    # TODO: Improve prompt engineering (get verb, subject, etc..)
    return "{} {}".format(fmmt_phrase, clean_prompt)


def _add_static_image_to_audio(image_path, audio_path, output_path):
    """Create and save a video file to `output_path` after combining a static image that
    is located in `image_path` with an audio file in `audio_path`."""

    # create the audio clip object
    audio_clip = AudioFileClip(str(audio_path))
    # create the image clip object
    image_clip = ImageClip(str(image_path), ismask=False)
    # use set_audio method from image clip to combine the audio with the image
    video_clip = image_clip.set_audio(audio_clip)
    # specify the duration of the new clip to be the duration of the audio clip
    video_clip.duration = audio_clip.duration
    # set the FPS to 1
    video_clip.fps = audio_clip.duration * 2
    # write the resulting video clip
    video_clip.write_videofile(str(output_path))


def _paragraphs_splitter(text, rules_kw=None, min_length=5):
    """ Splits a given text into paragraphs based on the specified rules and minimum
    paragraph length.

    Args:
        text (str): The text to be split into paragraphs.
        rules_kw (list[str], optional): A list of keywords that indicate the end of a
            paragraph (default=["\\n", ",", ".", ";"]).
        min_length (int, optional): The minimum number of words in a paragraph
            (default=5).

    Returns:
        dict: A dictionary where the keys are "p1", "p2", ..., and the values are the
            corresponding paragraphs of the input text.
    """

    if rules_kw is None:
        rules_kw = ["\n", ",", ".", ";"]
    paragraphs_dict = {}
    text_split = text.split()

    n_paragraphs = 0
    counter = 0
    new_paragraph = []
    for i, word in enumerate(text_split):
        new_paragraph.append(word)
        counter += 1
        if counter > min_length:
            for kw in rules_kw:
                if kw in word:
                    counter = 0
                    n_paragraphs += 1
                    paragraphs_dict["p{}".format(n_paragraphs)] = " ".join(
                        new_paragraph)
                    new_paragraph = []

                    break
        if i + 1 == len(text_split) and len(new_paragraph) > 0:
            n_paragraphs += 1
            paragraphs_dict["p{}".format(n_paragraphs)] = " ".join(new_paragraph)

    return paragraphs_dict


class ScreenWriter:
    """A class for generating video scripts using AI-powered text generation engines."""

    def __init__(self,
                 base_path,
                 gcp_sa_key,
                 replicate_api_key,
                 open_ai_key,
                 engine="gpt-3.5-turbo",
                 verbose=1,
                 replicate_engine=REPLICATE_ENGINE,
                 replicate_version=REPLICATE_VERSION,
                 image_height=768,
                 image_width=768,
                 timeout=60 * 60,
                 default_tags=None):
        """Initialize ScreenWriter.

        Args:
            base_path (str): The base path for the generated video script files.
            gcp_sa_key (str): The path to the Google Cloud Platform service account key
                file.
            replicate_api_key (str): The API key for the Replicate API.
            open_ai_key (str): The API key for the OpenAI API.
            engine (str, optional): The AI-powered text generation engine to use.
                Defaults to "gpt-3.5-turbo".
            verbose (int, optional): The verbosity level of the class. Defaults to 0.
            replicate_engine (str, optional): The Replicate engine to use. Defaults to
                REPLICATE_ENGINE.
            replicate_version (str, optional): The version of the Replicate engine to
                use. Defaults to REPLICATE_VERSION.
            image_height (int, optional): The height of the images used in the video
                script. Defaults to 768.
            image_width (int, optional): The width of the images used in the video
                script. Defaults to 768.
            timeout (int, optional): The maximum amount of time in seconds to wait for a
                response from the text generation engine. Defaults to 3600.
            default_tags (List[str], optional): The default tags to use for the video
                script. Defaults to None.
        """

        if default_tags is None:
            default_tags = ["artificial intelligence", "future", "machines",
                            "stable diffusion", "chatGPT", "youtubeislife",
                            "subscriber", "youtubeguru", "youtubecontent",
                            "newvideo", "subscribers", "youtubevideo",
                            "youtub", "youtuber", "youtubevideos"]

        self.tags = default_tags
        self.description = ""

        openai.api_key = open_ai_key
        openai.timeout = timeout

        self.engine = engine
        self.verbose = verbose
        self.base_path = base_path / "m{}".format(
            datetime.now().strftime("%Y%m%d%H%M%S"))
        self.replicate_client = replicate.Client(api_token=replicate_api_key)
        self.replicate_model = self.replicate_client.models.get(replicate_engine)
        self.replicate_version = self.replicate_model.versions.get(replicate_version)
        self.image_dimensions = "{}x{}".format(image_width, image_height)

        self.gcp_service_account = service_account.Credentials.from_service_account_file(
            gcp_sa_key)

        self.title = None
        self.paragraphs = {}
        self.cover = None

    def generate_text(self, prompt, **kwargs):
        """Generates a response to the given prompt using OpenAI's chatbot API.

        Args:
            prompt (str): The prompt to use for generating the response.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            str: The generated response to the prompt.

        Raises:
            OpenAIError: If there was an error with the OpenAI API.
        """

        messages = [{"role": "user",
                     "content": prompt}]

        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=messages,
            **kwargs)

        return response["choices"][0]["message"]["content"]

    def generate_title_from_prompt(self,
                                   prompt,
                                   **kwargs
                                   ):
        title = self.generate_text(prompt=prompt,
                                   **kwargs)

        if self.verbose > 0:
            print(title)

        return re.sub(r'[\n\t\r\'\"]', '', title)

    def generate_content_from_prompt(self, prompt, **kwargs):
        content = self.generate_text(prompt=prompt,
                                     **kwargs)

        if self.verbose:
            print(content)

        return content

    def generate_image_from_prompt(self, prompt, **kwargs):

        pred = self.replicate_version.predict(prompt=prompt,
                                              image_dimensions=self.image_dimensions,
                                              **kwargs)

        image = Image.open(BytesIO(requests.get(pred[0]).content))

        if self.verbose > 1:
            image.show()

        return image

    def text_to_speech(self, text, voice_name=None):
        """Synthesizes a given text into speech using Google Cloud Text-to-Speech API.

        Args:
            text (str): The input text to synthesize into speech.
            voice_name (Optional[str]): The name of the voice to use for synthesizing
                speech. If None, a default voice is used.

        Returns:
            bytes: The synthesized audio content as bytes.

        Raises:
            ValueError: If the given text is empty or too long.
            google.api_core.exceptions.InvalidArgument: If there is an error with the
                Google Cloud Text-to-Speech API.
        """

        if voice_name is None:
            voice_name = "en-US-News-N"

        language_code = "-".join(voice_name.split("-")[:2])

        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code, name=voice_name)
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

        client = tts.TextToSpeechClient(credentials=self.gcp_service_account)
        response = client.synthesize_speech(input=text_input, voice=voice_params,
                                            audio_config=audio_config)

        return response.audio_content

    def update_tags_from_content(self, content,
                                 max_tags=5,
                                 rake_nltk_var=Rake(),
                                 ):

        clean_content = " ".join(re.sub('[^a-zA-Z ]+', '', content).split())
        rake_nltk_var.extract_keywords_from_text(clean_content)
        keyword_extracted = rake_nltk_var.get_ranked_phrases()
        keyword_extracted = [k for k in set(keyword_extracted)]
        keyword_extracted = sorted(keyword_extracted, key=len)
        max_tags = min(max_tags, len(keyword_extracted))
        self.tags += keyword_extracted[:max_tags]

    def update_description_from_content(self, content,
                                        description_kwargs=None,
                                        content_to_description_prompt=None):

        if description_kwargs is None:
            description_kwargs = {"max_tokens": 64 * 2,
                                  "temperature": 0.9}

        if content_to_description_prompt is None:
            content_to_description_prompt = "Summarize the content:"

        description_prompt = "{} {}".format(content_to_description_prompt, content)

        description = self.generate_content_from_prompt(prompt=description_prompt,
                                                        **description_kwargs)

        description = """Video Generate using 100% Artificial Intelligence. \n {}
        """.format(description)

        self.description = description

    def fit(self,
            title_prompt,
            music_path,
            title_kwargs=None,
            content_kwargs=None,
            description_kwargs=None,
            image_kwargs=None,
            rules_kw=None,
            min_length=5,
            voice_name=None,
            title_to_content_prompt="Generate a Youtube script about",
            title_to_cover_prompt="""Generate a HD very engage-able Youtube cover video 
            that the title is:""",
            content_to_description_prompt=None,
            content_to_image_prompt="Generate a HD Youtube scene where:"):

        """Trains the ScreenWriter model using the provided parameters and generates a
        final video based on the script.

        Args:
            title_prompt (str): The initial prompt for generating the video title.
            music_path (str): The path to the audio file to use as background music.
            title_kwargs (Optional[dict]): The parameters to use for generating the
                video title. Defaults to None.
            content_kwargs (Optional[dict]): The parameters to use for generating the
                video content. Defaults to None.
            description_kwargs (Optional[dict]): The parameters to use for generating
                the video description. Defaults to None.
            image_kwargs (Optional[dict]): The parameters to use for generating the
                video images. Defaults to None.
            rules_kw (Optional[dict]): The parameters to use for splitting the generated
                content into paragraphs. Defaults to None.
            min_length (int): The minimum length of a paragraph. Defaults to 5.
            voice_name (Optional[str]): The name of the voice to use for the
                text-to-speech synthesis. Defaults to None.
            title_to_content_prompt (str): The prompt to use for generating the video
                content based on the title. Defaults to "Generate a Youtube script
                about".
            title_to_cover_prompt (str): The prompt to use for generating the video
                cover image based on the title. Defaults to "Generate a HD very
                engage-able Youtube cover video that the title is:".
            content_to_description_prompt (Optional[str]): The prompt to use for
                generating the video description based on the content. Defaults to None.
            content_to_image_prompt (str): The prompt to use for generating the video
                images based on the content. Defaults to "Generate a HD Youtube scene
                where:".
        """

        if content_kwargs is None:
            content_kwargs = {"max_tokens": 256 * 2,
                              "temperature": 0.9,
                              }
        if title_kwargs is None:
            title_kwargs = {"max_tokens": 32,
                            "temperature": 0.9}

        if image_kwargs is None:
            image_kwargs = {"num_inference_steps": 50}

        os.mkdir(path=self.base_path)
        paragraphs_path = self.base_path / "paragraphs"

        os.mkdir(path=paragraphs_path)

        self.title = self.generate_title_from_prompt(prompt=title_prompt,
                                                     **title_kwargs)

        content_prompt = "{} {}".format(title_to_content_prompt, self.title)

        self.paragraphs["p0"] = self.title

        cover_prompt = "{} {}".format(title_to_cover_prompt, self.title)

        self.cover = self.generate_image_from_prompt(prompt=cover_prompt,
                                                     **image_kwargs)

        # TODO: improve cover
        #  https://blog.devgenius.io/how-to-generate-youtube-thumbnails-easily-with-python-5d0a1f441f20

        content = self.generate_content_from_prompt(prompt=content_prompt,
                                                    **content_kwargs)

        self.update_description_from_content(content=content,
                                             description_kwargs=description_kwargs,
                                             content_to_description_prompt=content_to_description_prompt)

        self.update_tags_from_content(content=content)

        content = _paragraphs_splitter(text=content,
                                       rules_kw=rules_kw,
                                       min_length=min_length)

        self.paragraphs.update(content)

        # Combine speech + images

        for p, txt in self.paragraphs.items():

            prompt = _prompt_engineering(prompt=txt,
                                         fmmt_phrase=content_to_image_prompt)

            if p == "p0":
                new_image = self.cover
            else:
                new_image = self.generate_image_from_prompt(prompt=prompt,
                                                            **image_kwargs)

            new_speech = self.text_to_speech(text=txt,
                                             voice_name=voice_name)

            image_path = paragraphs_path / "{}_image.png".format(p)
            speech_path = paragraphs_path / "{}_speech.wav".format(p)

            new_image.save(image_path)

            with open(speech_path, "wb") as out:
                out.write(new_speech)

            output_path = paragraphs_path / "{}_image_speech.mp4".format(p)

            _add_static_image_to_audio(image_path=image_path,
                                       audio_path=speech_path,
                                       output_path=output_path)
            if self.verbose > 0:
                print("Paragraph {} saved in {}".format(p, output_path))

        # Merge Videos

        consolidate_list = []

        for root, dirs, files in os.walk(paragraphs_path):
            files = natsorted(files)
            for file in files:
                if file.endswith("_image_speech.mp4"):
                    file_path = os.path.join(root, file)
                    video = VideoFileClip(file_path)
                    consolidate_list.append(video)

        final_clip = concatenate_videoclips(consolidate_list)
        full_speech = final_clip.audio
        full_speech.write_audiofile(str(self.base_path / "speech.wav"), fps=44100)

        # Add Background Music
        # TODO: Add AI music engine:
        #  https://google-research.github.io/seanet/musiclm/examples/

        audio_clip = AudioFileClip(str(music_path))
        audio_clip = audio_clip.volumex(0.05)
        audio_clip = audio_clip.subclip(final_clip.start, final_clip.end)
        final_audio = CompositeAudioClip([final_clip.audio, audio_clip])
        final_clip = final_clip.set_audio(final_audio)
        final_clip.write_videofile(str(self.base_path / "final.mp4"))

        with open(self.base_path / "ScreenWriter.pkl", 'wb') as of:
            pickle.dump(self, of)
