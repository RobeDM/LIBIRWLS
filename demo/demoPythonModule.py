
import LIBIRWLS
import urllib
import numpy as np
import os
import time 

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

print "***************************************************"
print "* DOWNLOADING DATASETS FROM THE LIBSVM REPOSITORY *"
print "***************************************************"

dataDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","data")

if not os.path.exists(dataDirectory):
    os.makedirs(dataDirectory)

print "Downloading ADULT dataset for training: a9a"
urllib.urlretrieve ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a", os.path.join(dataDirectory,"a9a"))
print "Downloading ADULT dataset for testing: a9a.t"
urllib.urlretrieve ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t", os.path.join(dataDirectory,"a9a.t"))

#######################################################
#
# Loading the dataset and creating the numpy matrices.
#
#######################################################

Xtr,Ytr = load_svmlight_file(os.path.join(dataDirectory,"a9a"))
Xtst,Ytst = load_svmlight_file(os.path.join(dataDirectory,"a9a.t"))

Xtr=Xtr.todense()
Xtst=Xtst.todense()
print " "
print "*************************************************"
print "* RUNNING SVM (PIRWLS algorithm) USING 1 THREAD *"
print "*************************************************"


start = time.time()
model = LIBIRWLS.PIRWLStrain(Xtr, Ytr, gamma=0.001, C=1000, threads=1,verbose=0)
end = time.time()
print "Training time",(end - start)
predictions = LIBIRWLS.predict(model, Xtst, threads=1)
print "Accuracy", accuracy_score(Ytst, predictions)
print " "
print "*************************************************"
print "* RUNNING SVM (PIRWLS algorithm) USING 2 THREAD *"
print "*************************************************"

start = time.time()
model = LIBIRWLS.PIRWLStrain(Xtr, Ytr, gamma=0.001, C=1000, threads=2,verbose=0)
end = time.time()
print "Training time",(end - start)
predictions = LIBIRWLS.predict(model, Xtst, threads=2)
print "Accuracy", accuracy_score(Ytst, predictions)

print " "
print "********************************************************"
print "* RUNNING SEMIPARAMETRIC SVM (PSIRWLS) USING 1 THREADS *"
print "********************************************************"

start = time.time()
model = LIBIRWLS.PSIRWLStrain(Xtr, Ytr, gamma=0.0001, C=1000, size=75, threads=1,verbose=0)
end = time.time()
print "Training time",(end - start)
predictions = LIBIRWLS.predict(model, Xtst, threads=1)
print "Accuracy", accuracy_score(Ytst, predictions)

print " "
print "********************************************************"
print "* RUNNING SEMIPARAMETRIC SVM (PSIRWLS) USING 2 THREADS *"
print "********************************************************"

start = time.time()
model = LIBIRWLS.PSIRWLStrain(Xtr, Ytr, gamma=0.0001, C=1000, size=75, threads=2,verbose=0)
end = time.time()
print "Training time",(end - start)
predictions = LIBIRWLS.predict(model, Xtst, threads=2)
print "Accuracy", accuracy_score(Ytst, predictions)

