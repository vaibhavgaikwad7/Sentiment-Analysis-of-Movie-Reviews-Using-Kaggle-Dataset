'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
from xml.sax.handler import feature_external_ges
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import sentiment_read_subjectivity
import sentiment_read_LIWC_pos_neg_words
import crossval
from nltk.metrics import ConfusionMatrix
from nltk.collocations import *
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


nltkstopwords = nltk.corpus.stopwords.words('english')
stopwords1 = [
    'could', 'would', 'might', 'must', 'need', 'sha', 'wo', 'y', "'s", "'d", "'ll",
    "'t", "'m", "'re", "'ve", "n't", "'i", 'not', 'no', 'can', 'don', 'nt',
    'actually', 'also', 'always', 'even', 'ever', 'just', 'really', 'still', 
    'yet', 'however', 'nevertheless', 'furthermore', 'therefore', 'otherwise', 
    'meanwhile', 'though', 'although', 'thus', 'hence', 'indeed', 'perhaps', 
    'especially', 'specifically', 'usually', 'often', 'sometimes', 'certainly', 
    'sometimes', 'typically', 'mostly', 'generally', 'about', 'above', 'across', 
    'after', 'against', 'among', 'around', 'at', 'before', 'behind', 'below', 
    'beneath', 'beside', 'between', 'beyond', 'during', 'inside', 'onto', 'outside', 
    'through', 'under', 'upon', 'within', 'without'
]
stopwords = nltkstopwords + stopwords1


# initialize the positive, neutral and negative word lists
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types('C:/Users/gaikw/Downloads/FinalProjectData2/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')


# initialize positve and negative word prefix lists from LIWC 
#   note there is another function isPresent to test if a word's prefix is in the list
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

dpath = 'C:/Users/gaikw/Downloads/FinalProjectData2/FinalProjectData/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff'
SL = sentiment_read_subjectivity.readSubjectivity(dpath)

#Defining preprocessing function
def preprocessing(line):
  #converting to lower
  w = re.split(r'\s+', line.lower())
  #removing punctuations
  punc = re.compile(r'[!#$%&()*+,"-./:;<=>?@[\]^_`{|}~]')
  words = [punc.sub("",word) for word in w]
  #removing stop words
  words_final = []
  for i in words:
    if i in stopwords:
      continue   
    else:
      words_final.append(i)
  l = " ".join(words_final)
  return l

def filter_token2(tokens):
  word_list=[]
  for word in tokens[0]:
    if len(word)>2:
      word_list.append(word)
  return (word_list,tokens[1])




# Different Functions for feature sets :

def bagOfWords(list,i):
  list = nltk.FreqDist(list)
  wf = [w for (w,c) in list.most_common(i)]
  return wf   

def unigram_features(d,wf):
  df= set(d)
  f = {}
  for word in wf:
    f['V_%s'% word] = (word in df)
  return f


def bigram_bow(wordlist,n):
  bigram_measure = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(wordlist)
  finder.apply_freq_filter(2)
  b_features = finder.nbest(bigram_measure.chi_sq,4000)
  return b_features[:n]


def bigram_features(doc,word_features,bigram_feature):
  doc_words = set(doc)
  doc_bigrams = nltk.bigrams(doc)
  features = {}

  for word in word_features:
    features['V_{}'.format(word)] = (word in doc_words)
  
  for b in bigram_feature:
    features['B_{}_{}'.format(b[0],b[1])] = (b in doc_bigrams)

  return features


def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features

negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 
'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor','oppressive','revenge','revolting','rude','sticky','sick',
'sorry','tense','ugly','unwanted','unwelcome','pain','vile','vicious','quit','nonsense','guilty','impossible',
'hate','damage','dead','alarming','angry','annoy','corrupt','creepy','cruel','cry','dishonest','dirty','evil',
'enraged','reject','sad','terrifying','stupid','yell']


# Define preprocessing function with additional steps like lemmatization
def preprocessing(text):
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())                                                                     # type: ignore
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to extract TF-IDF features
def extract_tfidf_features(texts):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) # type: ignore
    tfidf_features = tfidf_vectorizer.fit_transform(texts)
    return tfidf_features


def NOT_features(document, new_word_features, negationwords):
    features = {}
    for word in new_word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in new_word_features)
        else:
            features['V_{}'.format(word)] = (word in new_word_features)
    return features


def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg) 

    if 'positivecount' not in features:
      features['positivecount'] = 0
    if 'negativecount' not in features:
      features['negativecount'] = 0

    return features


def liwc_features(doc,word_features,poslist,neglist):
  doc_words = set(doc)
  features= {}

  for word in word_features:
    features['contains({})'.format(word)] = (word in doc_words)
  
  pos = 0
  neg = 0
  for word in doc_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      pos+=1
    elif sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      neg+=1
    features ['positivecount'] = pos
    features ['negativecount'] = neg


  if 'positivecount' not in features:
    features['positivecount'] = 0
  if 'negativecount' not in features:
    features['negativecount'] = 0

  return features


def combo_sl_liwc_features(doc,word_features,SL,poslist,neglist):
  doc_words = set(doc)
  features={}

  for word in word_features:
    features['contains({})'.format(word)] = (word in doc_words )
  
  weakPos = 0
  strongPos = 0
  weakNeg = 0
  strongNeg = 0
  for word in doc_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      strongPos +=1
    elif sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      strongNeg +=1
    elif word in SL:
      strength, posTag, isStemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weakPos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strongPos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weakNeg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strongNeg += 1
    features['positivecount'] = weakPos + (2 * strongPos)
    features['negativecount'] = weakNeg + (2 * strongNeg)

  if 'positivecount' not in features:
    features['positivecount'] = 0
  if 'negativecount' not in features:
    features['negativecount'] = 0

  return features    



# Saving feature sets for for other classifier training
def savingfeatures(features, path):
    f = open(path, 'w')
    featurenames = features[0][0].keys()
    fnameline = ''
    for fname in featurenames:
        fname = fname.replace(',','COM')
        fname = fname.replace("'","SQ")
        fname = fname.replace('"','DQ')
        fnameline += fname + ','
    fnameline += 'Level'
    f.write(fnameline)
    f.write('\n')
    for fset in features:
        featureline = ''
        for key in featurenames:
            # Check if the key exists in the feature set
            if key in fset[0]:
                featureline += str(fset[0][key]) + ','
            else:
                featureline += 'NA,'  # If the key does not exist, write 'NA' instead
        if fset[1] == 0:
          featureline += str("-1lev")
        elif fset[1] == 1:
          featureline += str("-2lev")
        elif fset[1] == 2:
          featureline += str("0lev")
        elif fset[1] == 3:
          featureline += str("2lev")
        elif fset[1] == 4:
          featureline += str("1lev")
        f.write(featureline)
        f.write('\n')
    f.close()



def naivebayesaccuracy(features):
  train_set,test_set = features[int(0.1*len(features)):], features[:int(0.1*len(features))]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("\nAccuracy : ")
  print(nltk.classify.accuracy(classifier,test_set),"\n")
  l1 = []
  tl=[]
  for (features,label) in test_set:
    l1.append(label)
    tl.append(classifier.classify(features))
  print(ConfusionMatrix(l1,tl))




def dt(featuresets):
    n = 0.1
    cutoff = int(n * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_dt = SklearnClassifier(DecisionTreeClassifier())
    classifier_dt.train(train_set)
    print("Classifier-DecisionTree \n")
    print("Accuracy : ", nltk.classify.accuracy(classifier_dt, test_set))




def svm(featuresets):
    n = 0.1
    cutoff = int(n * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_svm = SklearnClassifier(SVC())
    classifier_svm.train(train_set)
    print("Classifier-SVM \n")
    print("Accuracy : ", nltk.classify.accuracy(classifier_svm, test_set))


def lr(featuresets):
    n = 0.1
    cutoff = int(n * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_lr = SklearnClassifier(LogisticRegression())
    classifier_lr.train(train_set)
    print("Classifier-Logistic Regression \n")
    print("Accuracy : ", nltk.classify.accuracy(classifier_lr, test_set))
    
    

def knn(featuresets):
    n = 0.1
    cutoff = int(n * len(featuresets))
    train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]
    classifier_knn = SklearnClassifier(KNeighborsClassifier())
    classifier_knn.train(train_set)
    print("Classifier-KNN \n")
    print("Accuracy : ", nltk.classify.accuracy(classifier_knn, test_set))



# define a feature definition function here

# use NLTK to compute evaluation measures from a reflist of gold labels
#    and a testlist of predicted labels for all labels in a list
# returns lists of precision and recall for each label


# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('C:/Users/gaikw/Downloads/FinalProjectData2/FinalProjectData/kagglemoviereviews/corpus/train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:

    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore th
      # e phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

  for phrase in phraselist[:10]:
    print (phrase)
  
  # create list of phrase documents as (list of words, label)
  phrasedocs_withpre = []
  phrasedocs_withoutpre= []
  # add all the phrases
  for phrase in phraselist:

    #Without preprocessing
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs_withoutpre.append((tokens, int(phrase[1])))

    #With preprocessing
    #tokenizer = Regexptokenizer(r'\w+')
    phrase[0] = preprocessing(phrase[0])
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs_withpre.append((tokens, int(phrase[1])))
  
  # possibly filter tokens:  
  phrasedocs_withpre_filter=[]
  # filtering with preprocessing
  for phrase in phrasedocs_withpre:
    phrasedocs_withpre_filter.append(filter_token2(phrase))

  filtered_tokens =[]
  unfiltered_tokens = []
  for (d,s) in phrasedocs_withpre_filter:
    for i in d:
      filtered_tokens.append(i)

  for (d,s) in phrasedocs_withoutpre:
    for i in d:
      unfiltered_tokens.append(i)
  


  # continue as usual to get all words and create word features
  
  # feature sets from a feature definition function

  filtered_bow_features = bagOfWords(filtered_tokens,350)
  unfiltered_bow_features = bagOfWords(unfiltered_tokens,350)

  filtered_unigram_features = [(unigram_features(d,filtered_tokens),s) for (d,s) in phrasedocs_withpre_filter]
  unfiltered_unigram_features = [(unigram_features(d,unfiltered_tokens),s) for (d,s) in phrasedocs_withoutpre]

  filtered_bigram_features = [(bigram_features(d,filtered_bow_features,bigram_bow(filtered_tokens,350)),s) for (d,s) in phrasedocs_withpre_filter]
  unfiltered_bigram_features = [(bigram_features(d,unfiltered_bow_features,bigram_bow(unfiltered_tokens,350)),s) for (d,s) in phrasedocs_withoutpre]

  filtered_pos_features = [(POS_features(d,filtered_bow_features),s) for (d,s) in phrasedocs_withpre_filter]
  unfiltered_pos_features = [(POS_features(d,unfiltered_bow_features),s) for (d,s) in phrasedocs_withoutpre]

  filtered_not_features = [(NOT_features(d, filtered_bow_features, negationwords), c) for (d, c) in phrasedocs_withpre_filter]
  unfiltered_not_features = [(NOT_features(d, unfiltered_bow_features, negationwords), c) for (d, c) in phrasedocs_withoutpre]

  filtered_sl_features = [(SL_features(d, filtered_bow_features, SL), c) for (d, c) in phrasedocs_withpre_filter]
  unfiltered_sl_features = [(SL_features(d, unfiltered_bow_features, SL), c) for (d, c) in phrasedocs_withoutpre]


  filtered_liwc_features = [(liwc_features(d, filtered_bow_features, poslist,neglist), c) for (d, c) in phrasedocs_withpre_filter]
  unfiltered_liwc_features = [(liwc_features(d, unfiltered_bow_features, poslist,neglist), c) for (d, c) in phrasedocs_withoutpre]

  filtered_combo_features =  [(combo_sl_liwc_features(d, filtered_bow_features,SL, poslist,neglist), c) for (d, c) in phrasedocs_withpre_filter]
  unfiltered_combo_features = [(combo_sl_liwc_features(d, unfiltered_bow_features,SL, poslist,neglist), c) for (d, c) in phrasedocs_withoutpre]


  

  #Saving features
  #savingfeatures(filtered_bow_features,'filtered_bow.csv')
  #savingfeatures(unfiltered_bow_features,'unfiltered_bow.csv')
  

  savingfeatures(filtered_unigram_features,'filtered_unigram.csv')
  savingfeatures(unfiltered_unigram_features,'unfiltered_unigram.csv')

  savingfeatures(filtered_bigram_features,'filtered_bigram.csv')
  savingfeatures(unfiltered_bigram_features,'unfiltered_bigram.csv')

  savingfeatures(filtered_pos_features,'filtered_pos.csv')
  savingfeatures(unfiltered_pos_features,'unfiltered_pos.csv')

  savingfeatures(filtered_not_features,'filtered_not.csv')
  savingfeatures(unfiltered_not_features,'unfiltered_not.csv')

  savingfeatures(filtered_sl_features,'filtered_sl.csv')
  savingfeatures(unfiltered_sl_features,'unfiltered_sl.csv')

  savingfeatures(filtered_liwc_features,'filtered_liwc.csv')
  savingfeatures(unfiltered_liwc_features,'unfiltered_liwc.csv')

  savingfeatures(filtered_combo_features,'filtered_combo.csv')
  savingfeatures(unfiltered_combo_features,'unfiltered_combo.csv')
  


  # train classifier and show performance in cross-validation

  labels = [0,1,2,3,4]
  print("Cross Validation for all features(unfiltered) : \n ")

  print("\n Unigram Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_unigram_features,labels)
  print("\n Bigram Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_bigram_features,labels)
  print("\n Pos Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_pos_features,labels)
  print("\n SL Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_sl_features,labels)
  print("\n LIWC Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_liwc_features,labels)
  print("\n Combined SL LIWC Unfiltered : ")
  crossval.cross_validation_PRF(5,unfiltered_combo_features,labels)

  print("\n Unigram filtered : ")
  crossval.cross_validation_PRF(5,filtered_unigram_features,labels)
  print("\n Bigram filtered : ")
  crossval.cross_validation_PRF(5,filtered_bigram_features,labels)
  print("\n Pos filtered : ")
  crossval.cross_validation_PRF(5,filtered_pos_features,labels)
  print("\n SL filtered : ")
  crossval.cross_validation_PRF(5,filtered_sl_features,labels)
  print("\n LIWC filtered : ")
  crossval.cross_validation_PRF(5,filtered_liwc_features,labels)
  print("\n Unigram filtered : ")
  crossval.cross_validation_PRF(5,filtered_combo_features,labels)



  print("\n Unigram Unfiltered : ")
  naivebayesaccuracy(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  naivebayesaccuracy(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  naivebayesaccuracy(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  naivebayesaccuracy(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  naivebayesaccuracy(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  naivebayesaccuracy(unfiltered_combo_features)


  print("\n Unigram filtered : ")
  naivebayesaccuracy(filtered_unigram_features)
  print("\n Bigram filtered : ")
  naivebayesaccuracy(filtered_bigram_features)
  print("\n Pos filtered : ")
  naivebayesaccuracy(filtered_pos_features)
  print("\n SL filtered : ")
  naivebayesaccuracy(filtered_sl_features)
  print("\n LIWC filtered : ")
  naivebayesaccuracy(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  naivebayesaccuracy(filtered_combo_features)

  print("--------------------------------------------------For desicion tree -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  dt(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  dt(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  dt(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  dt(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  dt(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  dt(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  dt(filtered_unigram_features)
  print("\n Bigram filtered : ")
  dt(filtered_bigram_features)
  print("\n Pos filtered : ")
  dt(filtered_pos_features)
  print("\n SL filtered : ")
  dt(filtered_sl_features)
  print("\n LIWC filtered : ")
  dt(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  dt(filtered_combo_features)


  print("--------------------------------------------------For svm -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  svm(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  svm(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  svm(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  svm(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  svm(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  svm(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  svm(filtered_unigram_features)
  print("\n Bigram filtered : ")
  svm(filtered_bigram_features)
  print("\n Pos filtered : ")
  svm(filtered_pos_features)
  print("\n SL filtered : ")
  svm(filtered_sl_features)
  print("\n LIWC filtered : ")
  svm(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  svm(filtered_combo_features)



  print("--------------------------------------------------For logistic regression -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  lr(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  lr(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  lr(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  lr(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  lr(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  lr(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  lr(filtered_unigram_features)
  print("\n Bigram filtered : ")
  lr(filtered_bigram_features)
  print("\n Pos filtered : ")
  lr(filtered_pos_features)
  print("\n SL filtered : ")
  lr(filtered_sl_features)
  print("\n LIWC filtered : ")
  lr(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  lr(filtered_combo_features)


  print("--------------------------------------------------For knn -----------------------------------------------")
  print("\n Unigram Unfiltered : ")
  knn(unfiltered_unigram_features)
  print("\n Bigram Unfiltered : ")
  knn(unfiltered_bigram_features)
  print("\n Pos Unfiltered : ")
  knn(unfiltered_pos_features)
  print("\n SL Unfiltered : ")
  knn(unfiltered_sl_features)
  print("\n LIWC Unfiltered : ")
  knn(unfiltered_liwc_features)
  print("\n Combined SL LIWC Unfiltered : ")
  knn(unfiltered_combo_features)

  print("===== for filtered =====")


  print("\n Unigram filtered : ")
  knn(filtered_unigram_features)
  print("\n Bigram filtered : ")
  knn(filtered_bigram_features)
  print("\n Pos filtered : ")
  knn(filtered_pos_features)
  print("\n SL filtered : ")
  knn(filtered_sl_features)
  print("\n LIWC filtered : ")
  knn(filtered_liwc_features)
  print("\n Combined SL LIWC filtered : ")
  knn(filtered_combo_features)

  td = pd.read_csv('C:/Users/gaikw/Downloads/FinalProjectData2/FinalProjectData/kagglemoviereviews/corpus/train.tsv', sep='\t')
  print(td.shape)
  td.head()



"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])


