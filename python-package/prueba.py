import numpy as np
import LIBIRWLS
import time

from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file("../data/a9a")
X_train =  X_train.todense()
X_test, y_test = load_svmlight_file("../data/a9a.t")
X_test =  X_test.todense()


start_time = time.time()
dato = LIBIRWLS.PIRWLStrain(X_train,y_train,kernel=1,gamma=0.001,C=1000,threads=4)
end_time = time.time()
print("Trained PIRWLS in %s seconds " % (time.time() - start_time))
LIBIRWLS.predict(dato,X_test,y_test)

del dato

start_time = time.time()
dato = LIBIRWLS.PSIRWLStrain(X_train,y_train,kernel=1,gamma=0.01,C=10000,threads=4,algorithm=0,size=250)
end_time = time.time()
print("Trained PSIRWLS in %s seconds " % (time.time() - start_time))
LIBIRWLS.predict(dato,X_test,y_test)







