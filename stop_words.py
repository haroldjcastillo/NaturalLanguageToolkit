from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

example_test = "This is an example showing off stop word filtration."
stop_words = set(stopwords.words("english"))
words = word_tokenize(example_test)

filtered_sentence = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
print(example_test, filtered_sentence)
