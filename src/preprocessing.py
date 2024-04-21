# global
import string
from typing import List, Tuple

import numpy as np
import pandas as pd

import re
import nltk

from sklearn.utils import resample

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
from keras.layers import TextVectorization
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# local
from utils import Doc2VecModel


punct = string.punctuation
stemmer = nltk.stem.PorterStemmer()
eng_stopwords = nltk.corpus.stopwords.words("english")


class Preprocessor:
    def __init__(self) -> None:
        pass

    def _nltk_tokenizer(self, text: str) -> List[str]:
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text)

        return tokens

    def _tokenize_text(self, text_column: pd.Series) -> pd.Series:
        tokenized_text = text_column.apply(self._nltk_tokenizer)
        return tokenized_text

    def _convert_to_tagged_document(self, text_column: pd.Series) -> Tuple[List[str], List[TaggedDocument]]:
        tokens_list = text_column.to_list()
        tagged_docs = [TaggedDocument(t, [str(i)])
                       for i, t in enumerate(tokens_list)]

        return tokens_list, tagged_docs

    def _vectorize_text(self, doc2vec_model: Doc2Vec, df: pd.Series, tokens_list: List[str]) -> pd.DataFrame:
        text_vectors = [doc2vec_model.infer_vector(doc) for doc in tokens_list]
        text_vectors_df = pd.DataFrame(text_vectors, index=df.index)

        return text_vectors_df

    def _anonymize_case_facts(self, first_party_name: str, second_party_name: str, facts: str) -> str:
        # remove any commas and any non alphabet characters
        first_party_name = re.sub(r"[\,+]", " ", first_party_name)
        first_party_name = re.sub(r"[^a-zA-Z]", " ", first_party_name)

        second_party_name = re.sub(r"[\,+]", " ", second_party_name)
        second_party_name = re.sub(r"[^a-zA-Z]", " ", second_party_name)

        for name in first_party_name.split():
            facts = re.sub(name, " _PARTY_ ", facts)

        for name in second_party_name.split():
            facts = re.sub(name, " _PARTY_ ", facts)

        # replace any consecutive _PARTY_ tags with only one _PARTY_ tag.
        regex_continous_tags = r"(_PARTY_\s+){2,}"
        anonymized_facts = re.sub(regex_continous_tags, " _PARTY_ ", facts)
        # remove ant consecutive spaces
        anonymized_facts = re.sub(r"\s+", " ", anonymized_facts)

        return anonymized_facts

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        # remove quotation marks
        text = re.sub(r"\'", "", text)
        # remove digits
        text = re.sub(r"\d+", "", text)
        # remove punctuation but with keeping '_' letter
        text = "".join([ch for ch in text if (ch == "_") or (ch not in punct)])
        # remove brackets, braces, and parantheses
        text = re.sub(r"[\[\]\(\)\{\}]+", " ", text)
        tokens = nltk.word_tokenize(text)
        # remove stopwords and stemming tokens
        tokens = [stemmer.stem(token)
                  for token in tokens if token not in eng_stopwords]
        # convert tokens back to string
        processed_text = " ".join(tokens)

        return processed_text

    def convert_text_to_vectors_doc2vec(self, text_column: pd.Series, train: bool = True, embeddings_doc2vec: Doc2Vec = None) -> Tuple[Doc2Vec, pd.DataFrame] | pd.DataFrame:
        tokenized_text = self._tokenize_text(text_column)
        tokens_list, tagged_docs = self._convert_to_tagged_document(
            tokenized_text)

        if train:
            doc2vec_model = Doc2VecModel()
            embeddings_doc2vec = doc2vec_model.train_doc2vec_embeddings_model(
                tagged_docs
            )
            text_vectors_df = self._vectorize_text(
                embeddings_doc2vec, text_column, tokens_list
            )
            return embeddings_doc2vec, text_vectors_df

        assert (
            embeddings_doc2vec is not None
        ), "`embedding_doc2vec` argument must be not None."
        assert isinstance(
            embeddings_doc2vec, Doc2Vec
        ), "`embedding_doc2vec` argument must be an instance of Doc2Vec to infer vectors."
        text_vectors_df = self._vectorize_text(
            embeddings_doc2vec, text_column, tokens_list
        )

        return text_vectors_df

    def convert_text_to_vectors_tf_idf(
        self,
        text_column: pd.Series,
        ngrams: int = 2,
        max_tokens: int = 10000,
        output_mode: str = "tf-idf",
        train: bool = True,
        text_vectorizer: TextVectorization = None,
    ) -> Tuple[TextVectorization, tf.Tensor] | tf.Tensor:
        if train:
            text_vectorizer = TextVectorization(
                ngrams=ngrams, max_tokens=max_tokens, output_mode=output_mode
            )
            text_vectorizer.adapt(text_column)
            text_vectors = text_vectorizer(text_column)

            return text_vectorizer, text_vectors

        assert (
            text_vectorizer is not None
        ), "`text_vectorizer` argument must be not None."
        assert isinstance(
            text_vectorizer, TextVectorization
        ), "`text_vectorizer` argument must be an instance of TextVectorization to infer vectors."
        text_vectors = text_vectorizer(text_column)

        return text_vectors

    def convert_text_to_vectors_cnn(
        self,
        text_column: pd.Series,
        max_tokens: int = 2000,
        output_sequence_length: int = 500,
        output_mode: str = "int",
        train: bool = True,
        text_vectorizer: TextVectorization = None,
    ) -> Tuple[TextVectorization, tf.Tensor] | tf.Tensor:
        if train:
            text_vectorizer = TextVectorization(
                max_tokens=max_tokens,
                output_mode=output_mode,
                output_sequence_length=output_sequence_length,
            )
            text_vectorizer.adapt(text_column)
            text_vectors = text_vectorizer(text_column)
            return text_vectorizer, text_vectors

        assert (
            text_vectorizer is not None
        ), "`text_vectorizer` argument must be not None."
        assert isinstance(
            text_vectorizer, TextVectorization
        ), "`text_vectorizer` argument must be an instance of TextVectorization to infer vectors."
        text_vectors = text_vectorizer(text_column)

        return text_vectors

    def convert_text_to_vectors_glove(
        self,
        text_column: pd.Series,
        train: bool = True,
        glove_tokenizer: Tokenizer = None,
        vocab_size: int = 1000,
        oov_token: str = "<OOV>",
        max_length: int = 50,
        padding_type: str = "post",
        truncation_type: str = "post",
    ) -> Tuple[Tokenizer, np.ndarray] | np.ndarray:
        if train:
            glove_tokenizer = Tokenizer(
                num_words=vocab_size, oov_token=oov_token)
            glove_tokenizer.fit_on_texts(text_column)
            text_sequences = glove_tokenizer.texts_to_sequences(text_column)
            text_padded = pad_sequences(
                text_sequences,
                maxlen=max_length,
                padding=padding_type,
                truncating=truncation_type,
            )

            return glove_tokenizer, text_padded

        assert (
            glove_tokenizer is not None
        ), "`glove_tokenizer` argument must be not None."
        assert isinstance(
            glove_tokenizer, Tokenizer
        ), "`glove_tokenizer` argument must be an instance of Tokenizer."
        text_sequences = glove_tokenizer.texts_to_sequences(text_column)
        text_padded = pad_sequences(
            text_sequences,
            maxlen=max_length,
            padding=padding_type,
            truncating=truncation_type,
        )

        return text_padded

    def balance_data(self, X_train: pd.Series, y_train: pd.Series) -> pd.DataFrame:
        df = pd.concat([X_train, y_train], axis=1)

        first_party = df[df["winner_index"] == 0]
        second_party = df[df["winner_index"] == 1]

        upsample_second_party = resample(
            second_party, replace=True, n_samples=len(first_party), random_state=42
        )

        upsample_df = pd.concat([upsample_second_party, first_party])

        shuffled_indices = np.arange(upsample_df.shape[0])
        np.random.shuffle(shuffled_indices)

        shuffled_balanced_df = upsample_df.iloc[shuffled_indices, :]

        return shuffled_balanced_df

    def anonymize_data(self, first_party_names: pd.Series, second_party_names: pd.Series, text_column: pd.Series) -> pd.Series:
        all_anonymized_facts = []

        for i in range(text_column.shape[0]):
            facts = text_column.iloc[i]
            first_party_name = first_party_names.iloc[i]
            second_party_name = second_party_names.iloc[i]
            anonymized_facts = self._anonymize_case_facts(
                first_party_name, second_party_name, facts
            )
            all_anonymized_facts.append(anonymized_facts)

        return pd.Series(all_anonymized_facts)

    def preprocess_data(self, text_column: pd.Series) -> pd.Series:
        preprocessed_text = text_column.apply(self._preprocess_text)
        return preprocessed_text
