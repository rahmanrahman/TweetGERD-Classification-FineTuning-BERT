from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pickle

import transformers
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer

import tensorflow as tf

app = Flask(__name__)
PRE_TRAINED_MODEL = None
bert_tokenizer = None
bert_load_model = None
max_length= None



@app.route("/", methods=['GET', 'POST'])
def gerd_prediction():
    if request.method == 'GET':
        return render_template("gerd_prediction.html")
    elif request.method == 'POST':
        print(request.form)
        gerd_features = request.form
        gerd_features = str(request.form)

        
        

        input_text_tokenized = bert_tokenizer.encode((gerd_features),
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors='tf')

        bert_predict = bert_load_model(input_text_tokenized)       
        bert_output = tf.nn.softmax(bert_predict[0], axis=-1) 

        sms_labels = ['Lainnya','Informasi','Promosi', 'Keluhan']

        label = tf.argmax(bert_output, axis=1)
        label = label.numpy()

        result = sms_labels[label[0]]
        return render_template('gerd_prediction.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    PRE_TRAINED_MODEL = pickle.load(open("model-development/bert1.pkl", "rb"))
    bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    bert_load_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=4)
    bert_load_model.load_weights('model-development/bert-model-b.h5')
    max_length = 65
    app.run(port=5000, debug=True)