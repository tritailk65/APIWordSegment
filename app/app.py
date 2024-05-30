from flask import Flask, request, jsonify, make_response, redirect, url_for
from datetime import datetime
import py_vncorenlp
import json
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hello!</h1>"

@app.route("/ping")
def test():
    return "pong"

print("Current working directory:", os.getcwd())
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg","pos"], save_dir="E:/Learning/Project_Data/ProjectLaw/Final/API_VNCoreNLP/vncorenlp")
@app.route('/keyphrase', methods=["POST"])
def get_keyphrase():
    content3  = request.data.decode('utf-8')
    output = rdrsegmenter.word_segment(content3)
    return make_response(
              json.dumps(output,
                  ensure_ascii=False).encode('utf-8')) 

@app.route('/postag', methods=["POST"])
def get_postag():
    stopwords_path = "E:/Learning/Project_Data/ProjectLaw/Final/API_VNCoreNLP/vietnamese.txt"
    print("Current working directory:", os.getcwd())
    stop_words = get_stopwords_list(stopwords_path)
    content  = request.data.decode('utf-8')
    json_object = json.loads(content)
    my_dict = rdrsegmenter.annotate_text(json_object["text"])
    word_pos_list = [[d['wordForm'], d['posTag']] for d in my_dict[0]]
    # append new key with (N-N)
    generate_new_key(word_pos_list)
    # remove stop word
    stop_word_remove = [pair for pair in word_pos_list if pair[0] not in stop_words]  
    # remove one-syllable word
    filter_dict = [word for word in stop_word_remove if '_' in word[0]]
    json_data = []
    for item in filter_dict:
        json_data.append({"Key":item[0],"Pos":item[1]})
    return  jsonify(json_data)
    

def get_stopwords_list(stop_file_path):
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip().replace(" ","_") for m in stopwords)
        return list(frozenset(stop_set))
    
def generate_new_key(dict):
    new_dict = []
    for i in range(len(dict)-1):
        if dict[i][1] == 'N' and dict[i+1][1] == 'N':
            new_element = [dict[i][0]+'_'+dict[i+1][0],'N']
            new_dict.append(new_element)

    for element in new_dict:
        dict.append(element)
    return dict

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)