from pathlib import Path
import os
from dotenv import load_dotenv

PATH_MAIN = Path(os.path.dirname(__file__))
PATH_DATA = PATH_MAIN / "data"
PATH_DATA_SANDBOX = PATH_DATA / "sandbox"
PATH_DATA_MOVIES = PATH_DATA / "movies"
PATH_DATA_MOVIES_MUSIC = PATH_DATA_MOVIES / "music"
PATH_NOTEBOOKS = PATH_MAIN / "notebooks"
PATH_STATICS = PATH_MAIN / "statics"
PATH_STATICS_IMAGES = PATH_STATICS / "images"
PATH_STATICS_FONTS = PATH_STATICS / "fonts"

PATH_KEYS = PATH_MAIN / "keys"
PATH_PACKAGES = PATH_MAIN / "packages"


CLIENT_SECRETS = PATH_KEYS / "client_secrets.json"
GCP_SA_KEY = PATH_KEYS / "GCP_sa_key.json"

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
STABLEDIFFUSION_API_KEY = os.environ["STABLEDIFFUSION_API_KEY"]
PICOVOICE_API_KEY = os.environ["PICOVOICE_API_KEY"]

if __name__ == '__main__':
    print("the main path is: {}".format(PATH_MAIN))
    vars = locals().copy()
    paths = {}
    for k, v in vars.items():
        if k.startswith("PATH_"):
            path = Path(v)
            if path.is_dir():
                print("directory {} already exists".format(v))
            else:
                os.mkdir(path)
                print("directory {} created".format(v))
