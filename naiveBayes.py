#!/usr/bin/python
import sys
#alternative python path
sys.path.append("/usr/local/lib/python2.7/site-packages")
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import nltk
from copy import deepcopy
import math
#Haowen Qu (hq5rc) 
#Machine Learning HW5
###############################################################################

vocabulary =  {"love":0, "wonderful":0, "best":0, "great":0, "superb":0, "still":0,  
         "beautiful":0, "bad":0, "worst":0, "stupid":0,"waste":0, "boring":0, "?":0, "!":0, "UNK":0}


def transfer(fileDj, vocabulary):
    BOWDj = deepcopy(vocabulary)
    with open(fileDj) as f:
        for line in f:
            line.replace("loved","love")
            line.replace("loves","love")
            line.replace("loving","love")            
            for token in line.split():
                    if token not in vocabulary:   
                        BOWDj['UNK']+=1
                    else:
                        BOWDj[token]+=1
    return BOWDj


def loadData(Path):  

    Xtrain = np.empty(shape=[0,15])
    Xtest = np.empty(shape=[0,15])
    ytest = []
    ytrain = []

    wordList = ["love", "wonderful", "best", "great", "superb", "still", "beautiful", 
                    "bad", "worst", "stupid", "waste", "boring", "?", "!", "UNK"]
    #Training - Positive
    for filename in os.listdir(Path+"training_set/pos"):
        filepath = Path + "training_set/pos/" +filename
        BOWDj=transfer(filepath,vocabulary)
        row = []
        for i in wordList:
            row.append(BOWDj[i])
        Xtrain = np.vstack([Xtrain,row])
        ytrain.append(1)

    #Training - Negative 
    for filename in os.listdir(Path+"training_set/neg"):
        filepath = Path + "training_set/neg/" +filename
        BOWDj=transfer(filepath,vocabulary)
        row = []
        for i in wordList:
            row.append(BOWDj[i])
        Xtrain = np.vstack([Xtrain,row])
        ytrain.append(-1)

    #Testing - Positive
    for filename in os.listdir(Path+"test_set/pos"):
        filepath = Path + "test_set/pos/" +filename
        BOWDj=transfer(filepath,vocabulary)
        row = []
        for i in wordList:
            row.append(BOWDj[i])
        Xtest = np.vstack([Xtest,row])
        ytest.append(1)

    #Testing - Negative 
    for filename in os.listdir(Path+"test_set/neg"):
        filepath = Path + "test_set/neg/" +filename
        BOWDj=transfer(filepath,vocabulary)
        row = []
        for i in wordList:
            row.append(BOWDj[i])
        Xtest = np.vstack([Xtest,row])
        ytest.append(-1)

    #XTrain SHOULD BE INTEGER

    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    thetaPos=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    thetaNeg=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    sumOfPos = 0
    sumOfNeg = 0

    for i in range(0,Xtrain.shape[0]):
    
        if ytrain[i]==1:
            for x in range(0,15):
                sumOfPos += Xtrain.item((i,x))
                thetaPos[x] += Xtrain.item((i,x))

        else:
            for x in range(0,15):
                sumOfNeg += Xtrain.item((i,x))
                thetaNeg[x] += Xtrain.item((i,x))

    for x in range(0,15):

        thetaPos[x] = ((thetaPos[x])+1)/float(sumOfPos+15)
        thetaNeg[x] = ((thetaNeg[x])+1)/float(sumOfNeg+15)

    print thetaPos

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    yPredict = []

    for i in range (0, Xtest.shape[0]):
        posPoss = 0
        negPoss = 0
        for j in range(0,15):
            negPoss += Xtest.item((i,j)) * math.log(thetaNeg[j])
            posPoss += Xtest.item((i,j)) * math.log(thetaPos[j])

        if(negPoss<posPoss):
            yPredict.append(1)
        else:
            yPredict.append(-1)

    numOfCorrect = 0
    len_ytest = len(ytest)
    for i in range (0, len_ytest):
        if ytest[i] == yPredict[i]:
            numOfCorrect+=1
    Accuracy = float(numOfCorrect)/len_ytest


   # print Accuracy
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    yPredict = clf.predict(Xtest)


    numOfCorrect = 0
    len_ytest = len(ytest)
    for i in range (0, len_ytest):
        if ytest[i] == yPredict[i]:
            numOfCorrect+=1
    Accuracy = float(numOfCorrect)/len_ytest

    return Accuracy


def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg, vocabulary):
    vocab= ["love","wonderful", "best","great", "superb", "still", "beautiful", "bad", "worst", "stupid",
"waste", "boring", "?", "!", "UNK"]
#The order will change
#vocabulary.keys() = ['beautiful', 'love', 'worst', 'wonderful', 'still', 'best', '!', 'great', 'UNK', 'boring', 'superb', 'bad', 'stupid', 'waste', '?']
  #  print vocabulary.keys()
    posPoss = 0
    negPoss = 0
    yPredict = 0
    with open(path) as f:
        for line in f:
            for token in line.split():
                if token in vocab:
                    index = vocab.index(token)
                    posPoss += math.log(thetaPos[index])
                    negPoss += math.log(thetaNeg[index])
                else:
                    posPoss += math.log(thetaPos[14])
                    negPoss += math.log(thetaNeg[14])


    if posPoss>negPoss:
        yPredict = 1
    else:
        yPredict = -1
    return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg, vocabulary):
    yPredict = []
    numOfCorrect = 0
    total = 0

    for filename in os.listdir(path+"neg/"):
        filepath = path+"neg/" +filename
        preY =  naiveBayesMulFeature_testDirectOne(filepath,thetaPos, thetaNeg, vocabulary)
        yPredict.append(preY)
        total += 1
        if (preY == -1):
            numOfCorrect += 1

    for filename in os.listdir(path+"pos/"):
        filepath = path+"pos/" +filename
        preY =  naiveBayesMulFeature_testDirectOne(filepath,thetaPos, thetaNeg, vocabulary)
        yPredict.append(preY)
        total += 1
        if (preY == 1):
            numOfCorrect += 1


    Accuracy = float(numOfCorrect)/total

    print Accuracy

    return yPredict, Accuracy





def naiveBayesBernFeature_train(Xtrain, ytrain):
    thetaPosTrue=[]
    thetaNegTrue=[]
    numOfPos = 0
    numOfNeg = 0
    index = 0

    pos_Matrix = Xtrain[0:700]
    index = 0
    for k in range(0,15):
        sum = 0
        for i in pos_Matrix:
            if i[index] > 0:
                sum+=1
        thetaPosTrue.append(float(sum+1)/(700+2))
        index+=1

    neg_Matrix = Xtrain[700:1400]
    index = 0
    for k in range(0,15):
        sum = 0
        for i in neg_Matrix:
            if i[index] > 0:
                sum+=1
        thetaNegTrue.append(float(sum+1)/(700+2))
        index+=1

    # print thetaPosTrue
    # print thetaNegTrue
    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []

    for i in range (0, Xtest.shape[0]):
        posPoss = 0
        negPoss = 0
        for j in range(0,15):
            negPoss += Xtest.item((i,j)) * math.log(thetaNegTrue[j])
            posPoss += Xtest.item((i,j)) * math.log(thetaPosTrue[j])
        if(negPoss<posPoss):
            yPredict.append(1)
        else:
            yPredict.append(-1)

    numOfCorrect = 0
    len_ytest = len(ytest)
    for i in range (0, len_ytest):
        if ytest[i] == yPredict[i]:
            numOfCorrect+=1
    Accuracy = float(numOfCorrect)/len_ytest
    
    return yPredict, Accuracy



# textDataSetsDirectoryFullPath = './data_sets/'
# testFileDirectoryFullPath = './data_sets/test_set/'
# Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)
# thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
# naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg)
# Accuracy = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
# naiveBayesMulFeature_testDirectOne("data_sets/test_set/pos/cv701_14252.txt",thetaPos, thetaNeg, vocabulary)
# Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, vocabulary)
# thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
# naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()
    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg,vocabulary)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"

