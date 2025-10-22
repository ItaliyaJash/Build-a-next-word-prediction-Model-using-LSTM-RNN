from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model('next_word.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

def predict_word(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predict_word = ""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predict_word = key
            break
    
    print(predict_word)  # Fixed indentation
    return predict_word  # Fixed indentation

  
# # Call the function and store the result
# result = predict_word(model, tokenizer, "The Man with the")

while True: 
    text = input("Enter your line: ")
    if text == "1":
        break
    else : 
        text = text.split(" ")
        text = text[-3:]
        predict_word(model, tokenizer, text)