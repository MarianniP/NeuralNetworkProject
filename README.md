# NeuralNetworkProject
Classification Problem. We want to detect if a fake news headline was published in an humorous website.
Our data is in a csv file, which is imported with the use of pandas library. Before we construct our neural network we create a function that pre-process our data. It stems, tokenizes and removes stop words with the use of NLTK and scikit-learn library. After that we construct a Pipeline and initialize our neural network. We use precision,recall and F1score to evaluate the results.
