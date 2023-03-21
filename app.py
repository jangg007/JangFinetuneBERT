import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("jang007/JangModel")
    return tokenizer,model

tokenizer, model = get_model()

user_input = st.text_area('Enter text to analyze')
button = st.button('Analyze')

d = {
    1: "Toxic",
    0: "Non Toxic"
}


if user_input and button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])


# pip install -r .\requirements.txt
# streamlit run .\app.py

# git init
# git add .\app.py .\requirements.txt
# git commit -m 'first commit'
# git branch -M main
# git push -u origin main