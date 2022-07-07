import os.path
from data_load_save_pandas import load_csv, save_csv
import pandas as pd
import spacy
from sklearn.pipeline import Pipeline
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class TagRecommendation:
    def __init__(self, directory):
        self.directory = directory
        self.nlp = spacy.load('en_core_web_sm')
        self.train_df = ""
        self.predict_df = ""
        self.test_df = ""

    def load_data(self):
        self.train_df = load_csv(path=self.directory, file_name="train_tag.csv").sample(frac=1)
        print(self.train_df.shape)
        self.train_df['tags'] = self.train_df['tags'].str.lower()
        self.train_df['tags'] = self.train_df['tags'].str.split(",")

        self.train_df, self.test_df = train_test_split(self.train_df, test_size=0.2, random_state=42)

    def preprocess_title(self, sentence):
        doc = self.nlp(sentence)
        return " ".join([word.lemma_.lower() for word in doc])
        # word.is_alpha == True and word.is_stop == False and word.pos_ in ["NOUN", "PROPN", "PRON"]])

    def knn_pipe(self, model, mode, file, df_or_string, threshold=0.7):
        if mode == "train":
            pipe = Pipeline([("lemma", LemmaClean(directory="")),
                             ("tfidf", TfidfVectorizer()),
                             ("knn", NearestNeighbors(n_neighbors=3)),
                             # cannot exten beyond knn cause nearest neighbours has no fit , transofrm type method
                             ])
            model = pipe.fit(self.train_df)

            #joblib.dump(model, filename=(os.path.join(self.directory, file)))
            return model
        elif mode == "predict":
            model = model  # joblib.load(os.path.join(self.directory, file))
            knn_out = model["knn"].kneighbors(model['tfidf'].transform(model["lemma"].transform(df_or_string)),
                                              return_distance=False)
            return self.predict_tags(knn_out=knn_out, df_or_string=df_or_string, threshold=threshold)

    def process_tags(self, count, string, knn_out, threshold):
        indexs = knn_out[count]
        inp = string
        tags = list(self.train_df['tags'].loc[indexs])
        tags_comb = sum(tags, [])
        # print(tags_comb)
        inp = self.nlp(inp)
        out_tag = [tag for tag in tags_comb if
                   self.nlp(tag).similarity(self.nlp(inp)) > threshold]  # and tag.isalnum()]
        return out_tag

    def predict_tags(self, knn_out, df_or_string, threshold):
        self.train_df = self.train_df.reset_index()
        if isinstance(df_or_string, pd.DataFrame):
            df_or_string = df_or_string.reset_index()
            out_tag = [" ".join(self.process_tags(count, str(df_or_string['title'].loc[count]), knn_out, threshold)) for
                       count in
                       tqdm(range(len(knn_out)))]
            df_or_string["out_tag"] = out_tag

            return df_or_string
        if isinstance(df_or_string, str):
            return self.process_tags(count=0, knn_out=knn_out, string=df_or_string, threshold=threshold)


class LemmaClean(TagRecommendation):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):
            return [self.preprocess_title(sentence=X)]
        elif isinstance(X, pd.DataFrame):
            X["title_ps"] = X["title"].apply(self.preprocess_title)
            return list(X["title_ps"])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


## dont run this in main joblib will not work in other file other wise
