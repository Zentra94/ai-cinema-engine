# AI Cinema Engine

![Futuristic Cinema by Stable Diffusion](statics/images/logo.png)

## Project Requirements

- `python` 3.x (this project was developed using `python` 3.9.6)
- `virtualenv` installed (or any virtual environment manager ex: `conda`) 
- `pip` installed
- `git` installed

## Local Installation

### Clone the Project
```bash
git clone https://github.com/Zentra94/ai-cinema-engine.git
```

### Create virtual environment:
```bash
virtualenv venv
```
### Activate the virtual environment:
```bash
venv/Scripts/activate
```
### Packages installation with pip:
```bash
pip install -r requirements.txt
```
### For create all directories:
```bash
python configs.py
```

## 3th. Party Credentials

### Add environments variables

Please follow the instructions on each respective link to obtain the corresponding API 
keys for the services (all of them are free but have usage limits):

- [OpenAI](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) (chatGPT)
- [Replicate](https://replicate.com/docs/get-started/python) (stable-diffusion) 
- [Picovoice](https://github.com/Picovoice/picovoice/tree/master/sdk/python) (captions)

Add the API at the `.env.example` file as follows, and rename as `.env` as follows:
```text
# .env
OPENAI_API_KEY=<OPENAI_API_KEY>
STABLEDIFFUSION_API_KEY=<STABLEDIFFUSION_API_KEY>
PICOVOICE_API_KEY=<PICOVOICE_API_KEY>
```
Please follow the instructions on each respective link to obtain the JSON-formatted 
keys and save them to their respective locations:

- [Google Cloud Platform Service Account](https://cloud.google.com/iam/docs/service-accounts-create) (Speech-to-Text) &rarr;  `keys/client_secrets.json`
- [Client Secrets](https://developers.google.com/youtube/v3/quickstart/python) (Youtube Data API v3) &rarr;  `keys/GCP_sa_key.json`


## Usage

You can have a E2E execution (create & upload a new video) running the `python main.py "TITLE PROMPT"` as the next example:

```bash
python main.py "create a unique title of a youtube video of 3 curios facts about pandas bears"
```

Wait some minutes, and you will have a result like this (click the image for check the video):

[![IMAGE_ALT](statics/images/pandas_example.png)](https://www.youtube.com/watch?v=LP7kzXZt34Q)

This will include: cover page image for the Youtube miniature (1), automatic title (2) and description (3).

![pandas1](statics/images/pandas_example_1.png)

Also, will upload some ad-hoc tags.

![pandas2](statics/images/pandas_example_2.png)

And will provide some multi-language subtitles (uploaded automatically as well).

![pandas3](statics/images/pandas_example_3.png)

For more details and configurations you can check `packages/video_manges/core/screenwriter.py` & `packages/video_manges/core/youtube.py` files.



## (Optional) Add Virtualenv at Jupyter Notebook's Kernel

```bash
python -m ipykernel install --name venv --display-name "Ai-cinema-engine-venv"
```
