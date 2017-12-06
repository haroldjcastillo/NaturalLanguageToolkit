import nltk
import random
from nltk.corpus import movie_reviews
import pickle
import os.path
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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
featuresets = [(find_features(rev), category) for (rev, category) in documents]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]


if os.path.exists("naivebayes.pickle"):
    with open("naivebayes.pickle", "rb") as classifier_f:
        classifier = pickle.load(classifier_f)
else:
    with open("naivebayes.pickle", "wb") as save_classifier:
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        pickle.dump(classifier, save_classifier)

print("Original Naive Bayes Algo accuracy perecent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MMB_classifier = SklearnClassifier(MultinomialNB())
MMB_classifier.train(training_set)
print("MultinomialNB accuracy perecent: ", (nltk.classify.accuracy(MMB_classifier, testing_set)) * 100)

# GMB_classifier = SklearnClassifier(GaussianNB())
# GMB_classifier.train(training_set)
# print("GaussianNB accuracy perecent: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB accuracy perecent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy perecent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGDClassifier accuracy perecent: ", (nltk.classify.accuracy(SGD_classifier, testing_set)) * 100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy perecent: ", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy perecent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC accuracy perecent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)