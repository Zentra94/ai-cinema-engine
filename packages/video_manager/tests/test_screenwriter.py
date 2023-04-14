import unittest
from packages.video_manager.core.screenwriter import (ScreenWriter,
                                                      _paragraphs_splitter,
                                                      _enhance_thumbnail)
from configs import (PATH_DATA_MOVIES_MUSIC,
                     PATH_DATA_MOVIES,
                     GCP_SA_KEY,
                     REPLICATE_API_KEY,
                     OPENAI_API_KEY)
from PIL import Image


class TestScreenWriter(unittest.TestCase):
    content = """
        Hey everyone! Welcome to my channel. Today I’m going to be talking about 5 
        Surprising Facts about Panda Bears – You Won't Believe #3! 

        First, did you know that panda bears have been around for millions of years? 
        That’s right, they’ve been roaming the earth since ancient times. 
        
        Second, pandas are actually considered a type of bear, not a type of raccoon as 
        many people believe. Pandas are part of the Ursidae family, the same family 
        that includes brown, polar and black bears. 
        
        Third, pandas are actually one of the pickiest eaters in the animal kingdom, 
        and they only consume bamboo. In fact, pandas can eat up to 12 kg of bamboo 
        every day and they can do it in as little as 15 minutes! 
        
        Fourth, panda bears are actually very good climbers and they spend most of their 
        time high up in the trees, where their favorite food - bamboo - is plentiful. 
        
        And finally, here’s the one you won’t believe – pandas have six fingers! Yes, 
        that’s right, the pandas have an extra digit on each of their front paws, 
        making them great at climbing and grabbing things. 
        
        So there you have it – 5 surprising facts about panda bears. What do you think, 
        have you learned something new? Let me know in the comments below! 
        
        Thanks for watching and I’ll see you next time!
        """

    screen_writer = ScreenWriter(verbose=1,
                                 engine="gpt-3.5-turbo",
                                 base_path=PATH_DATA_MOVIES,
                                 gcp_sa_key=GCP_SA_KEY,
                                 open_ai_key=OPENAI_API_KEY,
                                 image_height=512,
                                 image_width=1024,
                                 replicate_api_key=REPLICATE_API_KEY,
                                 replicate_stability_engine="stability-ai/stable-diffusion",
                                 replicate_stability_version="f178fa7a1ae43a9a9af01b833b9d2ecf97b1bcb0acfd2dc5dd04895e042863f1",
                                 )

    def test_generate_title_from_keywords(self):
        title = self.screen_writer.generate_title_from_prompt(
            prompt="""
            create a unique title of a youtube video of 5 curios 
            facts about pandas bears""",
            max_tokens=32,
            temperature=0.9
        )

        self.assertEqual(len(title) > 0, True)

    def test_generate_content_from_prompt(self):
        title = "5 Surprising Facts About Panda Bears - You Won't Believe #3!"
        title_to_content_prompt = "Generate a Youtube plain script about",
        content_prompt = "{} {}".format(title_to_content_prompt, title)

        content = self.screen_writer.generate_content_from_prompt(
            prompt=content_prompt,
            max_tokens=256 * 2,
            temperature=0.4)

        self.assertEqual(len(content) > 0, True)

    def test_paragraphs_splitter(self):
        text = self.content

        paragraphs = _paragraphs_splitter(text=text, min_length=5)

        self.assertEqual(len(paragraphs) > 5, True)

    def test_generate_image_from_prompt(self):
        image = self.screen_writer.generate_image_from_prompt(
            prompt="5 Surprising Facts About Panda Bears - You Won't Believe #3!",
            num_inference_steps=20)

        self.assertEqual(image.height > 0, True)

    def test_text_to_speech(self):
        text = """
        Hey everyone! Welcome to my channel. Today I’m going to be talking about 5 
        Surprising Facts about Panda Bears – You Won't Believe #3!"""

        speech = self.screen_writer.text_to_speech(text=text)

        self.assertEqual(len(speech), 362254)

    def test_generate_tags_from_content(self):
        content = self.content
        old_tags = self.screen_writer.tags.copy()

        self.screen_writer.update_tags_from_content(content=content,
                                                    max_tags=15)

        tags = self.screen_writer.tags

        for t in old_tags:
            self.assertEqual(t in tags, True)

        self.assertEqual(len(tags) > len(old_tags), True)

    def test_update_description_from_content(self):
        content = self.content

        old_description = self.screen_writer.description

        self.screen_writer.update_description_from_content(content=content)

        description = self.screen_writer.description

        self.assertEqual(len(old_description) < len(description), True)

    def test_prompt_engineering(self):
        text = "5 Surprising Facts About Panda Bears - You Won't Believe #3!"
        prompt = self.screen_writer.prompt_engineering(text=text,
                                                       update_prompt_engineering_default_list=True)
        quality = "HD, dramatic lighting, detailed, realistic."

        assert prompt.endswith(quality)

    def test_enhance_thumbnail(self):
        title = "Video generated 100% with Artificial Intelligence."
        path_image = PATH_DATA_MOVIES / 'm20230413165056/paragraphs/p0_image.png'
        image = Image.open(path_image)
        new_image = _enhance_thumbnail(image=image,
                                       title=title)

        new_image.show()
        self.assertEqual(True, True)

    def test_generate_music(self):
        text = "5 Surprising Facts About Panda Bears - You Won't Believe #3!"

        path = self.screen_writer.generate_wav_file_music_from_prompt(prompt_a=text,
                                                                      path=PATH_DATA_MOVIES_MUSIC/"test_bg_music.wav",
                                                                      min_duration_sec=20)

        self.assertEqual(path, PATH_DATA_MOVIES_MUSIC/"test_bg_music.wav")

    def test_fit(self):
        title_prompt = """
            create a unique title of a youtube video of 3 curios 
            facts about pandas bears"""

        content_kwargs = {"max_tokens": 256 * 4}
        image_kwargs = {"num_inference_steps": 50}
        music_path = PATH_DATA_MOVIES_MUSIC / "calm-chill-beautiful.mp3"

        self.screen_writer.fit(title_prompt=title_prompt,
                               music_path=music_path,
                               content_kwargs=content_kwargs,
                               image_kwargs=image_kwargs,
                               min_length=5)

        paragraphs = self.screen_writer.paragraphs

        self.assertEqual(len(paragraphs) > 1, True)


if __name__ == '__main__':
    unittest.main()
