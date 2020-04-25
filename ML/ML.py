from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors


df=pd.read_csv('parkinsons.csv')
#Get the features and labels 
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels
#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
#Train the model
model=XGBClassifier()
model.fit(x_train,y_train)
Rf=RandomForestClassifier(n_estimators=100)
Rf.fit(x_train,y_train)
Lr=LogisticRegression()
Lr.fit(x_train,y_train)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
svc_linear=SVC()
svc_linear.fit(x_train,y_train)
#Calculate the accuracy
print ("XGB Classifier accuracy :",model.score(x_test,y_test))
print ("Random Forest Classifier accuracy :",Rf.score(x_test,y_test))
print ("Logistic Regression accuracy :",Lr.score(x_test,y_test))
print ("KNeighborsClassifier accuracy :",clf.score(x_test,y_test))
print ("SVC accuracy :",svc_linear.score(x_test,y_test))
