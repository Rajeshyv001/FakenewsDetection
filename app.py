from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from random import randrange
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

app = Flask(__name__)




def decontracted(sentence):
    # specific
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence


def text_preprocessing(review_news):
    sent = review_news
    print(sent)
    sent = re.sub(r"http\S+", "", sent)            # removing urls
    sent = re.sub(r"\S+(.com)\S*", "", sent)       # removing url without http
    sent = decontracted(sent)                      # decontracting words
    sent = re.sub(r"[^\w\s]", "", sent)           # special characters or puntuations
    sent = re.sub("\S*\d\S*", "", sent).strip()   # removing numbers 
    sent = nltk.word_tokenize(sent)               # tokenizing
    words = [word.lower() for word in sent if word.lower() not in stopwords.words('english')]   # removing stopwords
    lemmatize_word = [WordNetLemmatizer().lemmatize(w) for w in words if len(w)>2]     # lemmatization
    filtered_data = ' '.join(lemmatize_word)
    print(filtered_data)
    return filtered_data
            

text_vectorizer = pickle.load(open("CountVectorizer.pkl", 'rb'))
logistic_model= pickle.load(open("Logistic_model.pkl", 'rb'))




@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=['GET','POST'])
def Newsprediction():
    if request.method == "POST":
        if request.form.get("news_submit_a"):
            retrieved_news = str(request.form['news_details'])
            print(retrieved_news)
            text_preprocessed = text_preprocessing(retrieved_news)
            predict = logistic_model.predict(text_vectorizer.transform([text_preprocessed]))[0]
            print(predict)
            return render_template("prediction.html", predicted_news = predict)
        elif request.form.get("news_submit_b"):
            data = pd.read_csv("test_data.csv.gz", compression='gzip')
            index = randrange(0, len(data)-1, 1)
            # news = jsonify({'title': data.loc[index].title, 'text': data.loc[index].text})
            news = data.loc[index].title +" "+ data.loc[index].text
            print(data.loc[index].id)
            return render_template("prediction.html", predict_news = news)
        elif request.form.get("news_submit_c"):
            return render_template("prediction.html")

    else:
        return render_template("prediction.html")

@app.route("/about", methods=['GET'])
def about(): 
    return render_template("About.html")

@app.route("/contact", methods=['GET'])
def contact(): 
    return render_template("Contact.html")

@app.route("/feedback", methods=['GET'])
def feedback():
    return render_template("feedback.html", feedback= "Thank you! We will going to consider your feedback")

if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080)