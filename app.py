import streamlit as st;
import numpy as np;
import pickle;
from tensorflow.keras.models import load_model;
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('lstm_gru.h5');
with open('token_pickle.pickle','rb') as file:
    token=pickle.load(file)


def predict_next_word(text):
    max_seq_length=model.input_shape[1]+1;
    token_list=token.texts_to_sequences([text])[0]
    if len(token_list)>=max_seq_length:
        token_list=token_list[-(max_seq_length-1):]
    token_list=pad_sequences([token_list],maxlen=max_seq_length-1,padding='pre')
    predicted=model.predict(token_list)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in token.word_index.items():
        if index==predicted_word_index:
            return word
        
st.title("Next Word Prediction Model")
input_text=st.text_input("Enter your text here")
if st.button("Predict Next Word"):
    next_word=predict_next_word(input_text)
    st.write('Next Word is ',next_word);

 