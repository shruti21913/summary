from flask import Flask,request, url_for, redirect, render_template
import nltk
import bs4 as BeautifulSoup
import urllib.request

# from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# from nltk.corpus import stopwords

# from nltk import word_tokenize

from heapq import nlargest
import urllib3

app = Flask(__name__)

# filename = 'nlp_model.pkl'
# clf = pickle.load(open(filename, 'rb'))
# cv=pickle.load(open('tranform.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("main.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':

        # article_content = ''
        text_url = request.form['text1']
        http = urllib3.PoolManager()
        r = http.request('GET', text_url)
        text = r.data
        # proxies=urllib.request.ProxyHandler({'http':None})

        # opener=urllib.request.build_opener(proxies)

        # urllib.request.install_opener(opener)

        # text=urllib.request.urlopen(url="https://www.google.com/",timeout=20)


        

        #text_url ='https://en.wikipedia.org/wiki/Machine_learning'
        # text = urllib.request.Request(text_url)
        # t=urllib.request.urlopen(text)
        article = text
        # # Parsing the URL content 
        article_parsed = BeautifulSoup.BeautifulSoup(article,'html.parser')
        # Returning <p> tags
        paragraphs = article_parsed.find_all('p')
        # To get the content within all poaragrphs loop through it
        article_content = ''
        for p in paragraphs:  
            article_content += p.text
        # for p in paragraphs:  
        #     article_content += p.text
        tokens = word_tokenize(article_content)
        # nltk.download("stopwords")
        stop_words = stopwords.words('english')

        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\n\n'  #punctuation + '\n'
        word_frequencies = {}
        for word in tokens:    
            if word.lower() not in stop_words:
                if word.lower() not in punctuation:
                    if word not in word_frequencies.keys():
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word]/max_frequency
        sent_token = sent_tokenize(article_content)
        sentence_scores = {}
        for sent in sent_token:
            sentence = sent.split(" ")
            for word in sentence:        
                if word.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.lower()]
        select_length = int(len(sent_token)*0.3)
        summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
        final_summary = [word for word in summary]
        summary = ' '.join(final_summary)

    
    return render_template('results.html',prediction = summary)


if __name__ == '__main__':
    app.run(debug=True)
