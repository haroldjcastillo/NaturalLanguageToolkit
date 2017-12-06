from nltk.tokenize import sent_tokenize, word_tokenize

example_test = "Hello Mr. Smith, how are you doing today? The weather is great and Python awesome. " \
               "The sky is pinkish-blue. You should not eat cardboard."
print(sent_tokenize(example_test))
print(word_tokenize(example_test))
for i in word_tokenize(example_test):
    print(i)
