import re

import fasttext
import langdetect

# TODO: move these bin files to a separate directory?
# This would require updating the install.sh script as well!
GLOTILD_MODEL_NAME = "glotlid-model.bin"
OPENILD_MODEL_NAME = "lid201-model.bin"


class FASTTEXTLIDModel:
    def __init__(self, model_bin_path="lid201-model.bin"):
        self.model = fasttext.load_model(model_bin_path)

    def predict(self, text):
        # \n breaks the fasttext prediction
        # TODO: Is it better to split the text into sentences using "\n" and do LID for each sentence?
        text = re.sub("\n", " ", text)
        label, logit = self.model.predict(text)
        label = label[0][len("__label__") :]
        language_code, script = label.split("_")
        assert len(language_code) == 3

        return f"[{language_code}:{logit[0]}]"


class LANGDETECTModel:
    def __init__(self):
        pass

    def predict(self, text):
        try:
            lang_pred = langdetect.detect_langs(text)
        except:
            # TODO: investigate this
            lang_pred = "error"
        return lang_pred
