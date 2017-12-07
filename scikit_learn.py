import nltk
import random
from nltk.corpus import movie_reviews
import pickle
import os.path

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

feature_sets = [(find_features(rev), category) for (rev, category) in documents]
training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

if os.path.exists("naivebayes.pickle"):
    with open("naivebayes.pickle", "rb") as classifier_f:
        classifier = pickle.load(classifier_f)
else:
    with open("naivebayes.pickle", "wb") as save_classifier:
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        pickle.dump(classifier, save_classifier)

print("Naive Bayes Algo accuracy perecent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)
