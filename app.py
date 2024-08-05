from flask import Flask,render_template,request,redirect

from googletrans import Translator

from deep_translator import GoogleTranslator

app = Flask(__name__)


@app.route("/",methods=["GET", "POST"])
def home():
    if request.method == "POST":
     t_sentence = request.form["sentence"]
     language = request.form['inputvalue']
    #  output = Translator(service_urls=['translate.googleapis.com']).translate(t_sentence,dest='hi')
    #  output = GoogleTranslator(source='en', target='hi').translate(t_sentence)
     translator = GoogleTranslator(source='en', target='hi')
 
 
     translation = translator.translate(t_sentence)
 
 
    else:
        return render_template("home.html")
    return render_template('home.html',output=translation,sentence=t_sentence)


if __name__ == "__main__":
    app.run(debug=True)

























# from flask import Flask,render_template,request,redirect
# # from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime
# from googletrans import Translator
# from deep_translator import GoogleTranslator
# app = Flask(__name__)




# @app.route("/",methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#      t_sentence = request.form["sentence"]
#      language = request.form['inputvalue']
#      output = GoogleTranslator(source='en', target='hi').translate(t_sentence)
#     else:
#         return render_template("home.html")
#     return render_template('home.html',output=output,sentence=t_sentence)


# # @app.route("/admin", methods=["GET", "POST"])
# # def admin_post():
# #     if request.method == 'GET':
# #      post = Contacts.query.all()    
# #     return render_template('admin.html',post=post)




# if __name__ == '__main__':
#     app.run(debug=True)

