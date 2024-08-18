from googletrans import Translator
from deep_translator import GoogleTranslator
# translator = Translator(service_urls=['translate.googleapis.com'])
# output=translator.translate("I am a good boy.", dest='hi')
#output=translator.translate("I am a good boy.", dest='hi')

translator = GoogleTranslator(source='en', target='hi')
 
 
translation = translator.translate("I am good boy")
print(translation)