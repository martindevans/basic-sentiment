## Projects

### sentiment

This is a basic LSTM sentiment analyser, train on an ad-hoc dataset of movie reviews, yelp reviews and manually classified tweets. Data should be in files with each line being a single entry:

`Sentence<tab>classification`

`Classification` should be a number. `0` indicates negative, `1` indicates positive and `2` indicates neutral.

### w2v

This is an attempt at training a word2vec model. Using three datasets:

 - The ad-hoc sentiment dataset in the previous project
 - `Gutenbergdammit` dataset from <https://github.com/aparrish/gutenberg-dammit>
 - `Wikipedia` dataset from polyglot project <https://sites.google.com/site/rmyeid/projects/polyglot>

### pretrained-w2v