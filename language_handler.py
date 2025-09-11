# language_handler.py
from deep_translator import GoogleTranslator

class LanguageHandler:
    def __init__(self):
        from deep_translator import GoogleTranslator  # safer than googletrans
        self.translator = GoogleTranslator(source='auto', target='en')

    def translate_symptoms(self, text: str, target_lang: str = 'en') -> str:
        """
        Translate symptoms into the target language (default: English).
        """
        try:
            if not text.strip():
                return text
            translated = self.translator.translate(text)
            return translated
        except Exception as e:
            print(f"[LanguageHandler] Translation failed: {e}")
            # fallback = original text
            return text

language_handler = LanguageHandler()
