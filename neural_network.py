import nltk
import sklearn
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
porter_stemmer = PorterStemmer()
stops = set(stopwords.words("english"))
#nltk.download('punkt') #μόνο για τη πρώτη φορά που το τρέξαμε μετά είναι uptodate'



#read the csv file and put it in a pandas  dataframe

data=pd.read_csv("onion-or-not.csv")


#everything in lower case

data['text']=data['text'].str.lower()


#TOKENIZATION, STEMMING, REMOVE STOP WORDS

def identify_tokens(row):
    tokens=nltk.word_tokenize(row) #tokens: τα πάντα μαζί με σημεία στίξης και νούμερα αν υπάρχουν
    token_words=[w for w in tokens if w.isalpha()] #αφαίρεση σημείων στίξης με χρήση μεθόδου .isalpha(.isalnum αν θέλαμε να κραστήσουμε και νούμερα)
    stemmed_list=[porter_stemmer.stem(word) for word in token_words] #stemming , κρατάω τη ρίζα των λέξεων 
    no_stops=[w for w in stemmed_list if not w in stops]
    final =[] #οριζω κενή λίστα
    for word in no_stops:
        final.append(word) #βάζω κάθε στοιχείο στο τέλος της 
    return final
 

text_train,text_test,label_train,label_test=train_test_split(data['text'],data['label'],test_size=0.25)

#Ορίζω τα βήματα του Pipeline
pipeline=Pipeline([  
                     ('Vectorizing',CountVectorizer(analyzer = identify_tokens)), # βάζω σαν όρισμα τα επεξεργασμένα δεδομένα που περνούν απο τη συνάρτηση identify_tokens
                     ('tfidf', TfidfTransformer()),
                     ('MLP',MLPClassifier(verbose=True))])


pipeline.fit(text_train,label_train) #εκαπαίδευση νευρωνικού δικτύου

predictions=pipeline.predict(text_test)

#Εμφάνιση αποτελεσμάτων
print("\nConfusin Matrix\n\n {} \n\n {}".format(confusion_matrix(label_test,predictions),classification_report(label_test,predictions))) 


