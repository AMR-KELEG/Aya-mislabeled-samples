import re

import fasttext
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import langcodes
from lingua import LanguageDetectorBuilder

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

        return language_code, logit[0]


class LANGDETECTModel:
    def __init__(self):
        pass

    def predict(self, text):
        try:
            res = langdetect.detect_langs(text)
            # Only consider the first language.
            # This is what we are doing with FastText too.
            raw_lang, prob = res[0].lang, res[0].prob

            # Convert the ISO 639-1 language code to ISO 639-2 (3 letters)
            lang = langcodes.Language.get(raw_lang).to_alpha3()

            lang_pred = lang, prob

        except LangDetectException:
            # TODO: investigate this
            lang_pred = "error"
        return lang_pred


class LINGUAModel:
    def __init__(self) -> None:
        pass

    def predict(self, text):
        # Set up the detector in what's called the "high accuracy" mode.
        # The accuracy will supposedly be higher at the cost of speed.
        # https://github.com/pemistahl/lingua-py?tab=readme-ov-file#115-low-accuracy-mode-versus-high-accuracy-mode
        detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )

        res = detector.compute_language_confidence_values(text)

        # The name comes in uppercase, hence the `.lower()`
        label = res[0].language.iso_code_639_3.name.lower()
        proba = res[0].value

        return label, proba
