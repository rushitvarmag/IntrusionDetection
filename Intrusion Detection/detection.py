
import pandas as pd
import urllib
import csv


from sklearn.metrics import classification_report
import pickle
import warnings
import pickle
warnings.filterwarnings("ignore")

# Load the csv file
sasi = pd.read_csv("main_data.csv")
print(sasi.head())

# Select independent and dependent variable
X = sasi[["Duration", "src_bytes", "dst_bytes", "logged_in","Count"]]
Y = sasi["Class"]

from sklearn.model_selection import train_test_split
# Split the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

from sklearn.preprocessing import StandardScaler
# Feature scaling
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test= SS.transform(X_test)

from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import GaussianNB
# Instantiate the model
GNB = GaussianNB()
# Fit the model
GNB.fit(X_train, Y_train)

from sklearn.tree import DecisionTreeClassifier

# Instantiate the models
DTC1=DecisionTreeClassifier(random_state=1)
DTC2=DecisionTreeClassifier(random_state=10)
DTC3=DecisionTreeClassifier(random_state=20)
DTC4=DecisionTreeClassifier(random_state=50)

# Instantiate the model
DTC = VotingClassifier(estimators=[('dtc1', DTC1), ('dtc2', DTC2), ('dtc3', DTC3), ('dtc4', DTC4)], voting='hard') #Type of voting SOFT
# Fit the model
DTC.fit(X_train, Y_train)

from sklearn.svm import SVC
# Instantiate the models
svc1=SVC(kernel='poly')
svc2=SVC(kernel='rbf')
svc3=SVC(kernel='linar')
svc4=SVC(kernel='sigmoid')

# Instantiate the model
svc = VotingClassifier(estimators=[('svc1', svc1), ('svc2', svc2), ('svc3', svc3), ('svc4', svc4)], voting='hard') #Type of voting SOFT
# Fit the model
svc.fit(X_train, Y_train)
# Fit the model



from sklearn.ensemble import VotingClassifier
#Generating ensemble
ensemble = VotingClassifier(estimators=[('gnb', GNB), ('dtc', DTC), ('svc', svc)], voting='soft') #Type of voting SOFT
# Fit the model
ensemble.fit(X_train,Y_train)

# Make pickle file of our model
pickle.dump(ensemble, open("model.pkl", "wb"))

