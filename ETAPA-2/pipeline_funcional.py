import pandas as pd
import pickle
import spacy
import nltk
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from additional_functions import join_text_columns, extract_text_column

nlp = spacy.load("es_core_news_sm")
nltk.download('punkt_tab')
nltk.download('stopwords')

text_pipeline = Pipeline([
    ("join_text_columns", FunctionTransformer(join_text_columns)),
    ("extract_text_column", FunctionTransformer(extract_text_column)),
    ("tfidf", TfidfVectorizer(max_df=0.95, min_df=2))
])

column_transformer = ColumnTransformer([
    ("text_processing", text_pipeline, ["Titulo", "Descripcion"]),
], remainder="passthrough")

pipe = Pipeline([
    ("column_transformer", column_transformer),
    ("modelo", LinearSVC(C=1.0, max_iter=5000))
])

if __name__ == "__main__":
    df = pd.read_csv("fake_news_spanish.csv", sep=";")
    df.drop_duplicates(subset=["Titulo", "Descripcion"], keep="first", inplace=True)
    X = df[["Titulo", "Descripcion"]]
    y = df["Label"]
    pipe.fit(X, y)
    with open("modelo_entrenado.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print("Pipeline entrenado y guardado en 'modelo_entrenado.pkl'")