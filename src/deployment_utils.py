# global
from typing import Tuple, List
import re
import numpy as np
import pandas as pd

import tensorflow as tf
import fasttext
from tensorflow import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models.doc2vec import Doc2Vec

import transformers
from transformers import pipeline, BertTokenizer

# local
from preprocessing import Preprocessor
from utils import read_data


# read data
X_train, X_test, y_train, y_test = read_data()

# instantiate preprocessor object
preprocessor = Preprocessor()

# load models
doc2vec_model_embeddings = Doc2Vec.load(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_doc2vec_embeddings")
doc2vec_model = keras.models.load_model(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_doc2vec_model.h5")
tfidf_model = keras.models.load_model(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_tfidf_model.h5")
cnn_model = keras.models.load_model(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_cnn_model.h5")
glove_model = keras.models.load_model(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_glove_model.h5")
lstm_model = keras.models.load_model(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_lstm_model.h5")
bert_model = keras.models.load_model(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_bert_model.h5", custom_objects={"TFBertModel": transformers.TFBertModel})
fasttext_model = fasttext.load_model(
    "D:/Projects/APEX_PROJECT/Ongoing/GUXLocal/code/lawGov/models/best_fasttext_model-002.bin")
summarization_model = pipeline(
    "summarization", model="facebook/bart-large-cnn")


# TODO: Add Docstrings
def extract_case_information(case_content: str):
    content_list = case_content.split("\n")
    petitioner = re.findall(r"petitioner:(.+)", content_list[0])[0]
    respondent = re.findall(r"respondent:(.+)", content_list[1])[0]
    facts = re.findall(r"facts:(.+)", content_list[2])[0]

    return petitioner, respondent, facts


def generate_random_sample() -> Tuple[str, str, str, int]:
    random_idx = np.random.randint(low=0, high=len(X_test))

    petitioner = X_test["first_party"].iloc[random_idx]
    respondent = X_test["second_party"].iloc[random_idx]
    facts = X_test["Facts"].iloc[random_idx]
    label = y_test.iloc[random_idx][0]

    return petitioner, respondent, facts, label


def generate_highlighted_words(facts: str, petitioner_words: List[str], respondent_words: List[str]):
    rendered_text = '<div class="text-facts"> '

    for word in facts.split():
        if word in petitioner_words:
            highlight_word = ' <span class="highlight-petitioner"> ' + word + " </span> "
            rendered_text += highlight_word

        elif word in respondent_words:
            highlight_word = ' <span class="highlight-respondent"> ' + word + " </span> "
            rendered_text += highlight_word

        else:
            rendered_text += " " + word

    rendered_text += " </div>"

    return rendered_text


class VectorizerGenerator:
    def __init__(self) -> None:
        pass

    def generate_tf_idf_vectorizer(self) -> keras.preprocessing.text.Tokenizer:
        first_party_names = X_train["first_party"]
        second_party_names = X_train["second_party"]
        facts = X_train["Facts"]

        anonymized_facts = preprocessor.anonymize_data(
            first_party_names, second_party_names, facts)

        text_vectorizer, _ = preprocessor.convert_text_to_vectors_tf_idf(
            anonymized_facts)

        return text_vectorizer

    def generate_cnn_vectorizer(self) -> keras.preprocessing.text.Tokenizer:
        balanced_df = preprocessor.balance_data(X_train["Facts"], y_train)
        X_train_balanced = balanced_df["Facts"]

        text_vectorizer, _ = preprocessor.convert_text_to_vectors_cnn(
            X_train_balanced)

        return text_vectorizer

    def generate_glove_tokenizer(self) -> keras.preprocessing.text.Tokenizer:
        balanced_df = preprocessor.balance_data(X_train["Facts"], y_train)
        X_train_balanced = balanced_df["Facts"]

        glove_tokenizer, _ = preprocessor.convert_text_to_vectors_glove(
            X_train_balanced)

        return glove_tokenizer

    def generate_lstm_tokenizer(self) -> keras.preprocessing.text.Tokenizer:
        lstm_tokenizer = Tokenizer(num_words=18430)
        lstm_tokenizer.fit_on_texts(X_train)

        return lstm_tokenizer

    def generate_bert_tokenizer(self) -> transformers.BertTokenizer:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        return bert_tokenizer


class DataPreparator:
    def __init__(self) -> None:
        self.vectorizer_generator = VectorizerGenerator()

    def prepare_doc2vec(self, facts: str) -> pd.DataFrame:
        facts = pd.Series(facts)
        facts_processed = preprocessor.preprocess_data(facts)
        facts_vectors = preprocessor.convert_text_to_vectors_doc2vec(
            facts_processed, train=False, embeddings_doc2vec=doc2vec_model_embeddings)

        return facts_vectors

    def _anonymize_facts(self, first_party_name: str, second_party_name: str, facts: str) -> str:
        anonymized_facts = preprocessor._anonymize_case_facts(
            first_party_name, second_party_name, facts)

        return anonymized_facts

    def prepare_tf_idf(self, anonymized_facts: str) -> tf.Tensor:
        anonymized_facts = pd.Series(anonymized_facts)
        tf_idf_vectorizer = self.vectorizer_generator.generate_tf_idf_vectorizer()

        facts_vector = preprocessor.convert_text_to_vectors_tf_idf(
            anonymized_facts, train=False, text_vectorizer=tf_idf_vectorizer)

        return facts_vector

    def prepare_cnn(self, facts: str) -> tf.Tensor:
        facts = pd.Series(facts)

        cnn_vectorizer = self.vectorizer_generator.generate_cnn_vectorizer()

        facts_vector = preprocessor.convert_text_to_vectors_cnn(
            facts, train=False, text_vectorizer=cnn_vectorizer)

        return facts_vector

    def prepare_glove(self, facts: str) -> np.ndarray:
        facts = pd.Series(facts)

        glove_tokneizer = self.vectorizer_generator.generate_glove_tokenizer()

        facts_vector = preprocessor.convert_text_to_vectors_glove(
            facts, train=False, glove_tokenizer=glove_tokneizer)

        return facts_vector

    def prepare_lstm(self, facts: str) -> np.ndarray:
        facts = pd.Series(facts)
        lstm_tokenizer = self.vectorizer_generator.generate_lstm_tokenizer()
        facts_vector = lstm_tokenizer.texts_to_sequences(facts)
        facts_vector_padded = pad_sequences(facts_vector, 974)

        return facts_vector_padded

    def prepare_bert(self, facts: str) -> tf.Tensor:
        bert_tokenizer = self.vectorizer_generator.generate_bert_tokenizer()
        facts_vector_dict = bert_tokenizer.encode_plus(
            facts,
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )

        return facts_vector_dict["input_ids"]


class Predictor:

    def __init__(self) -> None:
        self.data_preparator = DataPreparator()

    def predict_doc2vec(self, facts: str) -> np.ndarray:
        facts_vector = self.data_preparator.prepare_doc2vec(facts)
        predictions = doc2vec_model.predict(facts_vector)

        pet_res_scores = []
        for i in predictions:
            temp = i[0]
            pet_res_scores.append(np.array([1 - temp, temp]))

        return np.array(pet_res_scores)

    def predict_tf_idf(self, anonymized_facts: str) -> np.ndarray:
        facts_vector = self.data_preparator.prepare_tf_idf(anonymized_facts)
        predictions = tfidf_model.predict(facts_vector)

        pet_res_scores = []
        for i in predictions:
            temp = i[0]
            pet_res_scores.append(np.array([1 - temp, temp]))

        return np.array(pet_res_scores)

    def predict_cnn(self, facts: str) -> np.ndarray:
        facts_vector = self.data_preparator.prepare_cnn(facts)
        predictions = cnn_model.predict(facts_vector)

        pet_res_scores = []
        for i in predictions:
            temp = i[0]
            pet_res_scores.append(np.array([1 - temp, temp]))

        return np.array(pet_res_scores)

    def predict_glove(self, facts: str) -> np.ndarray:
        facts_vector = self.data_preparator.prepare_glove(facts)
        predictions = glove_model.predict(facts_vector)

        pet_res_scores = []
        for i in predictions:
            temp = i[0]
            pet_res_scores.append(np.array([1 - temp, temp]))

        return np.array(pet_res_scores)

    def predict_lstm(self, facts: str) -> np.ndarray:
        facts_vector = self.data_preparator.prepare_lstm(facts)
        predictions = lstm_model.predict(facts_vector)

        pet_res_scores = []
        for i in predictions:
            temp = i[0]
            pet_res_scores.append(np.array([1 - temp, temp]))

        return np.array(pet_res_scores)

    def predict_bert(self, facts: str) -> np.ndarray:
        facts_vector = self.data_preparator.prepare_bert(facts)
        predictions = bert_model.predict(facts_vector)

        return predictions

    def predict_fasttext(self, facts: str) -> np.ndarray:
        prediction = fasttext_model.predict(facts)[1]
        prediction = np.array([prediction])

        pet_res_scores = []
        for i in prediction:
            temp = i[0]
            pet_res_scores.append(np.array([1 - temp, temp]))

        return np.array(pet_res_scores)

    def summarize_facts(self, facts: str) -> str:
        summarized_case_facts = summarization_model(facts)[0]['summary_text']
        return summarized_case_facts
