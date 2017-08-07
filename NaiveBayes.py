import math
from collections import defaultdict

###Joshua Beresford-Davis

trainingFile = open("sampleTrain.txt","r")
vocabFile = open("sampleTrain.vocab.txt","r")
testFile = open("sampleTest.txt","r")

vocabList = []

with open("sampleTrain.vocab.txt") as fp:
    for line in fp:
        vocabList.append(line[:-1].lower()) ##Strip out the /n character
vocabList = set(vocabList)

trfClassCounts = defaultdict(int)
trfTotalCount = 0
trfFreqTable = dict()
trfClassSizes = defaultdict(int)

with open("sampleTrain.txt") as fp:
    for line in fp:
        trfTotalCount += 1
        trfClass = line.split("\t")[1]
        if not trfClass in trfFreqTable:
            trfFreqTable[trfClass] = defaultdict(int)
        trfClassCounts[trfClass] += 1
        trfFeatures = (line.split("\t")[2])[:-1].split()
        for word in trfFeatures:
            trfClassSizes[trfClass] += 1
            if word in vocabList:
                trfFreqTable[trfClass][word] += 1

trfSmoothedProbTable  = dict()

for key in trfFreqTable:
    for word in vocabList:
        if not key in trfSmoothedProbTable:
            trfSmoothedProbTable[key] = defaultdict(float)
        smoothedWordFreq = trfFreqTable[key][word] + 1
        smoothedVocabSize = trfClassSizes[key] + len(vocabList)
        trfSmoothedProbTable[key][word] = math.log(smoothedWordFreq / smoothedVocabSize)

trfPriorProbs = {}

for key in trfClassCounts:
    trfPriorProbs[key] = trfClassCounts[key] / trfTotalCount

featureLikelihoodString1 = "\t"
featureLikelihoodString2 = "class 0\t"
featureLikelihoodString3 = "class 1\t"

for word in vocabList:
    featureLikelihoodString1 += "{:>10}".format(word)
    featureLikelihoodString2 += "{:>10.5}".format(trfSmoothedProbTable["0"][word])
    featureLikelihoodString3 += "{:>10.5}".format(trfSmoothedProbTable["1"][word])

testRealClasses = dict()
testPredictedClasses = dict()

with open("sampleTest.txt") as fp:
    for line in fp:
        lineContents  =  line.split("\t")
        testRealClasses[lineContents[0]] = lineContents[1]

        maxClass = ""
        maxProb = 0
        firstGuess = True
        
        for word in trfSmoothedProbTable:
        
            wordProb = math.log(trfPriorProbs[word])
                                
            for word2 in lineContents[2].split():
                wordProb += trfSmoothedProbTable[word][word2]

            if (wordProb > maxProb or firstGuess == True):
                firstGuess = False
                maxClass = word
                maxProb = wordProb

        testPredictedClasses[lineContents[0]] = maxClass

correct = 0
for word in testRealClasses:
    if testRealClasses[word] == testPredictedClasses[word]:
        correct += 1
        
testAccuracy = 100 * correct / len(testRealClasses)

###Joshua Beresford-Davis

print("Prior probabilities")
print("class 0 = {}".format(trfPriorProbs["0"]))
print("class 1 = {}".format(trfPriorProbs["1"]))
print("")
print("Feature likelihoods (log probablities)")
print(featureLikelihoodString1)
print(featureLikelihoodString2)
print(featureLikelihoodString3)
print("Predictions on test data")
print("d5  = {}".format(testPredictedClasses["d5"]))
print("d6  = {}".format(testPredictedClasses["d6"]))
print("d7  = {}".format(testPredictedClasses["d7"]))
print("d8  = {}".format(testPredictedClasses["d8"]))
print("d9  = {}".format(testPredictedClasses["d9"]))
print("d10 = {}".format(testPredictedClasses["d10"]))
print("")
print("Accuracy on test data = {:.3}%".format(testAccuracy))
