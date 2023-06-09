import nltk
import pickle
import numpy as np
import json
import random
import time

from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from keras.models import load_model



lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

# list kosong bernama word, classes, documents
words=[] 
classes = []
documents = []
# stop word/pengabaian kata
ignore_words = ['?', '!'] 
# load dataset intents
data_file = open('intents.json').read() 
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # Memisahkan setiap kalimat menjadi perkata
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # menambahkan ke list document
        documents.append((w, intent['tag']))

        # menambahkan ke list classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Melakukan case folding dengan dan mengabaikan kata dengan ignore word
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words] 
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))



model = load_model('chatbot_model.h5')


## end keras chat brain

## vars
now = time.time() # float

filename = str(now)+"_chatlog.txt" #create chatlog

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
    
## end vars

class Storage:
    old_answers=[] # storage for old answers
    
    @classmethod
    def save_storage(cls):
        with open ("storage.txt", "w") as myfile:
            for answer in Storage.old_answers:
                
                myfile.write(answer+"\n")

    @classmethod
    def load_storage(cls):
        Storage.old_answers=[]
        with open ('storage.txt', 'r') as myfile:
            lines = myfile.readlines()
            for line in lines:
                Storage.old_answers.append(line.strip())
        print (Storage.old_answers)


app = Flask(_name_)


def bot_response(userText):

    '''fake brain'''
    print ("your q was: " + userText)
    return "your q was: " + userText
   
## new funcs
def clean_up_sentence(sentence):
    """tokenize/ pisah kalimat menjadi kata"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 atau 1 untuk setiap kata dalam bag yang ada dalam kalimat

def bow(sentence, words, show_details=True):
    # tokenize/pisah kata pada pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # cetak 1 jika kata saat ini berada di posisi kosakata
                bag[i] = 1
                if show_details:
                    print ("ditemukan pada bag : %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter prediksi dibawah threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # mengurutkan berdasarkan probabilitas tertinggi
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probabilitas": str(r[1])})
    return return_list
    
    
def getResponse(ints, intents_json):
    '''membaca intents file'''
    # pseudo code
    # menganggap jawaban lama di dalam didalam old_answers
    # old_answers = ['response1','response2']

    # load old answers ke storage
    Storage.load_storage()
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    old_answers = Storage.old_answers  # [:-len(list_of_intents)]
    possible_responses = [i['responses'] for i in list_of_intents if i['tag']== tag ][0]
    history = Storage.old_answers[-len('kemungkinan response\n'):]
    print("** kemungkinan answers dan hriwayat old answers\n", possible_responses, "\nhistory\n", history)
    unused_answers = [answer for answer in possible_responses if answer not in history ] # list comprehension
    print("unused answers\n", unused_answers)
    unused_two = history[-(len(possible_responses)-1):]
    print('5 jawaban terakhir\n', unused_two)
    try:
        result = random.choice([answer for answer in possible_responses if answer not in unused_two ])
        print(result)
    except IndexError:
        print("Saya kehabisan pilihan, saya akan memilih secara acak.")
        result = random.choice(possible_responses)

    Storage.old_answers.append(result) 
    Storage.old_answers= Storage.old_answers[-20:] 
    Storage.save_storage()

  
    return result,tag

def chatbot_response(msg):
    '''fungai terpenting'''
    ints = predict_class(msg, model)
    res,tag = getResponse(ints, intents)
    return res,tag


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        # tambah ke log file
        with open(filename,'a') as myfile:
            myfile.write("user: "+ msg + "\n")
            myfile.write("bot: "+ res + "\n")


## end new funcs
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route("/get")
def get_bot_response():    
    print ("get is called")
    userText = request.args.get('msg')    
    
    res,tag = chatbot_response(userText)
    with open( "logfile.csv", "a" ) as logfile:
        logfile.write(str(now)+","+userText+","+res+","+tag+","+"\n")
    print ('Saya pilih ini : ', res,tag)
    #return res + '<p style="font-size:8pt;">tag: ' + tag + '</p>'
    return res + '<p style="font-size:8pt;">' + tag + '</p>'



if _name_ == '_main_':
  app.run()
#    app.run(debug=True)