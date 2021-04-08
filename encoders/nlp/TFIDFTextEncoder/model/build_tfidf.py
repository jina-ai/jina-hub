from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pickle


def load_data():
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')
    return newsgroups_train.data

if __name__ == '__main__':

    X = load_data()
    
    # fit text featurizer descriptor
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)

    # store the object to disk
    pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pickle", "wb"))
    
    # load the object later
    # tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pickle", "rb"))
