# NaiveBayesClassifier

This programs implements a naive bayes classifier using pyhton. 


### Dataset
- Training data: 700 positive reviews and 700 negative reviews
- Testing data: 300 positive reviews and 300 negatives reviews

### 1. Preprocessing
Implement a dictionary

### 2. Build “bag of words” (BOW) Document Representation

convert a text document into a feature vector:

    BOWDj = transf er(f ileDj, vocabulary)
where fileDj  is the location of file j.

Read in the training and test documents into BOW vector representations using the above function. Then store features into matrix Xtrain and Xtest, and use ytrain and ytest to store the labels. 

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)
    
– “textDataSetsDirectoryFullPath” is the real full path of the file directory that you get from
unzipping the datafile. For instance, it is “/HW3/data sets/” on the instructor’s laptop.\
– loadData should call transfer()

### 3. Multinomial Naive Bayes Classifier (MNBC) Training Step

* We need to learn the P(cj) and P(wi|cj) through the training set. Through MLE, we use the relative- frequency estimation with Laplace smoothing to estimate these parameters.
* Since we have the same number of positive samples and negative samples, P(c = −1) = P(c = 1) = 1 .
 ``` 
thetaPos,thetaNeg = naiveBayesMulFeature train(Xtrain,ytrain)
```
Note: Pay attention to the MLE estimator plus smoothing; Here we choose α = 1.\
Note: thetaPos and thetaNeg should be python lists or numpy arrays (both 1-d vectors)

### 4. Multinomial Naive Bayes Classifier (MNBC) Testing+Evaluate Step
```
yPredict,Accuracy = naiveBayesMulFeature test(Xtest,ytest,thetaPos,thetaNeg)
```
* Use ”sklearnn ̇aive bayes.MultinomialNB” from the scikit learn package to perform training and testing. Compare the results with your MNBC. Add the resulting Accuracy into the writeup.

Important: Do not forget perform log in the classification process.

### 5.Multinomial Naive Bayes Classifier (MNBC) Testing through non-BOW feature representation

* For the step of classifying a test sample using MNBC, It is actually not necessary to first perform the BOW transformation for feature vectors.

```
yPredictOne = naiveBayesMulFeature testDirectOne(XtestTextFileNameInFullPathOne,thetaPos,thetaNeg)
```
* Use the above function on all the possible testing text files, calculate the ”classification accuracy” based on ”yPredict” versus the testing label.

```
yPredict, Accuracy = naiveBayesMulFeature testDirect(testFileDirectoryFullP ath, thetaPos, thetaNeg)
```
### 6.Multivariate Bernoulli Naive Bayes Classifier (BNBC)
* We need to learn the P(cj), P(wi = false|cj) and P(wi = true|cj) through the training. MLE gives the relative-frequency as the estimation of parameters. We will add with Laplace smoothing for estimating these parameters.
```
thetaPosTrue,thetaNegTrue = naiveBayesBernFeature train(Xtrain,ytrain)
yPredict,Accuracy = naiveBayesBernFeature test(Xtest,ytest,thetaPosTrue,thetaNegTrue)
```
### Analysis
Not surprisingly, the algorithm with continuous taken into consideration is more effective than the one without. Our original algorithm has an average of 0.675 accuracy of all the accuracy. 

Overall, this project demonstrated that NaiveBayes algorithm is very easy to implement and gives a pretty reliable result.

