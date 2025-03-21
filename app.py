import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


## Load teh LSTM MODEL
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("next_word_lstm.h5")
        if model is None:
            raise ValueError("Model is None after loading!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model only once
model = load_trained_model()

if model is None:
    st.stop()  # Stop execution if model fails to load

# Load teh tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

reversed_word_index = {value : key for key, value in tokenizer.word_index.items()}

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # Ensure the sequence length mathces the max_lenght
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    # for word, index in tokenizer.word_index.items():
    #     if index == predicted_word_index:
    #         return word
    return reversed_word_index.get(predicted_word_index, None)


## Streamlit app
st.title("Next Word Prediction With LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words", "To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word is not None:
        st.success(f"Next Word Predicted successfully: {next_word}")
    else:
        st.error("Oops! Prediction Failed")
    st.title(f"{next_word}")
