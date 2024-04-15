import os
import pickle
import string

import joblib
import nltk
import numpy as np
import pandas as pd
from huggingface_hub import HfFolder, HfApi, login
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from numpy import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as score
import openllm



tqdm.pandas()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

tokenizer = nltk.MWETokenizer(separator="")
stop_words = set(stopwords.words('english'))


def preprocess_text(text_df, fe_type):
    df_clean = text_df

    print('Removing Punctuation')
    df_clean['text'] = text_df['text'].progress_apply(remove_punctuation)

    print('Removing Stopwords')
    df_clean['text'] = text_df['text'].progress_apply(remove_stopwords)

    print('Applying Lemmatization')
    df_clean['text'] = text_df['text'].progress_apply(lemmatize_text)

    print('Applying Tokenization')
    df_clean['text_tokenized'] = text_df['text'].progress_apply(tokenize_text)

    df_clean.dropna(subset=['text_tokenized'], inplace=True)

    if fe_type == 'train':
        df_clean.to_csv("data/clean_text_tokenized_" + fe_type + ".csv", index=False)
        x_text = train_feature_extraction(df_clean)

    elif fe_type == 'test':
        x_text = test_feature_extraction(df_clean)

    return df_clean, x_text


def label_binarazation(y):
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    y_classes = label_binarizer.classes_
    y = np.argmax(y, axis=1)
    return y, y_classes


def train_evaluate_model():
    text_df = pd.read_csv("data/text.csv", encoding="ISO-8859-1")

    df, x_text = preprocess_text(text_df, fe_type='train')

    X = x_text
    y = text_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    y_test, _ = label_binarazation(y_test)
    y_train, y_classes = label_binarazation(y_train)

    classifiers = {
        "Linear SVM": LinearSVC(),
        "SGD": SGDClassifier(n_jobs=-1)
    }
    resp_lst = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        resp = evaluate_model(y_test, y_pred, name, y_classes)
        filename = 'models/' + name + '.sav'
        joblib.dump(clf, filename)
        resp_lst.append(resp)

    return resp_lst, 200


def load_model(model_name, text):
    loaded_model = joblib.load("models/" + model_name + '.sav')
    df = pd.DataFrame({'text': [text]}, index=[0])

    df, x_text = preprocess_text(df, fe_type='test')



    y_pred = loaded_model.predict(x_text)

    pred_label = mapping_labels(y_pred)
    response = generate_LLM(pred_label)
    return response, 200

def mapping_labels(prediction):
    label = ""
    if prediction[0] == 0:
        label = 'sadness'
    elif prediction[0] == 1:
        label = 'joy'
    elif prediction[0] == 2:
        label = 'love'
    elif prediction[0] == 3:
        label = 'anger'
    elif prediction[0] == 4:
        label = 'fear'
    elif prediction[0] == 5:
        label = 'surprise'

    return label

def train_feature_extraction(df):
    text_tfidf_vectorizer = TfidfVectorizer(max_features=100000)

    x_text = text_tfidf_vectorizer.fit_transform(df['text_tokenized'])

    joblib.dump(text_tfidf_vectorizer, "fe_data/tf-idf-train-vectorizer.pkl")

    with open(f"fe_data/tfidf_X_train_text.pkl", "wb") as f:
        pickle.dump(x_text, f)

    return x_text, text_tfidf_vectorizer


def test_feature_extraction(df):
    loaded_text_vector = joblib.load('fe_data/tf-idf-train-vectorizer.pkl')

    text = ' '.join(df['text_tokenized'][0])
    x_text_test = loaded_text_vector.transform([text])

    return x_text_test


def tokenize_text(text):
    tokens = tokenizer.tokenize(nltk.word_tokenize(text))
    return tokens


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def lemmatize_text(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Tokenize text into words
    words = nltk.word_tokenize(text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in words])
    return lemmatized_text


def evaluate_model(y_test, y_pred, model_name, y_classes):
    precision, recall, fscore, support = score(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    jacc_sc = jaccard_score(y_test, y_pred, average='weighted')

    print("Accuracy: ", acc)
    print("Classifier Used:", str(model_name))
    print(f'Jaccard Score: {jacc_sc:.4f}')
    print("\n")

    metric_df = pd.DataFrame(data=[precision, recall, fscore, support],
                             index=["Precision", "Recall", "F-1 score", "True Count"],
                             columns=y_classes)
    metric_df.to_csv("models/models_results/" + str(model_name) + "_metrics.xlsx")
    print(metric_df)
    resp = {"Accuracy": acc, "Jaccard Score": jacc_sc, "Classifier": str(model_name)}
    return resp


# async def prompt_llm(emotion):
#     # Initialize the LLM
#     llm = openllm.LLM('microsoft/phi-2') # Replace 'microsoft/phi-2' with your desired model
#     response = ""
#     # Craft the prompt
#     prompt = f"You are a helpful assistant. Recognize that the emotion is {emotion}. Respond appropriately."
#
#     # Generate the response
#     async for generation in llm.generate_iterator(prompt):
#         response += generation.outputs[0].text
#     return  response

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_LLM(prediction):
    # Check if token exists in cache or git credential


    login()
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it", torch_dtype=torch.bfloat16)

    # Generate text from a prompt
    prompt = "You are a helpful assistant. Recognize that the emotion is {emotion}. Respond appropriately.".format(
        emotion=prediction)
    input_ids = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**input_ids, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])