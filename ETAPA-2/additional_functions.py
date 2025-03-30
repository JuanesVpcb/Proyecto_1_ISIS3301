import unicodedata
import inflect
import spacy
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

nlp = spacy.load("es_core_news_sm")

def remove_non_ascii(words):
    new_words = []
    for word in words:
        if word is not None:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
    return new_words

def to_lowercase(words):
    new_words = []
    for i in words:
        new_words.append(i.lower())
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    new_words = []
    stop_words = set(stopwords.words('spanish'))
    for i in words:
        if i.lower() not in stop_words:
            new_words.append(i)
    return new_words

def remove_domains(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[\(\<]?\w+\.(com|net|org|gov|edu|es)[\)\>]?', '', word)
        new_word = new_word.strip()
        if new_word:
            new_words.append(new_word)
    return new_words

def clean_text(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = remove_domains(words)
    return words

def stem_words(words):
    stemmer = SpanishStemmer()
    new_words = []
    for word in words:
        new_words.append(stemmer.stem(word))
    return new_words

def lemmatize_verbs(words):
    new_words = []
    text = " ".join(words)
    doc = nlp(text)
    for sentence in doc.sents:
        new_words.extend([token.lemma_ for token in sentence])
    return new_words

def stem_and_lemmatize(words):
   stems = stem_words(words)
   lemma = lemmatize_verbs(words)
   return stems + lemma

def stringify_list(words):
    return " ".join(words)

# Funciones para pipeline
def join_text_columns(data):
    data = data.copy()
    data["Titulo"] = data["Titulo"].fillna(" ")
    data["Descripcion"] = data["Descripcion"].fillna(" ")
    data["Texto"] = data["Titulo"] + " " + data["Descripcion"]
    data["Texto"] = data["Texto"].apply(word_tokenize).apply(clean_text).apply(stem_and_lemmatize).apply(stringify_list)
    return data[["Texto"]]

def extract_text_column(df):
    return df["Texto"]