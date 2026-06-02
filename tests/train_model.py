import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib


df=pd.read_csv("Dataset/ear_dataset.csv")

X=df[["EAR"]]
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=RandomForestClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)

joblib.dump(model,"Models/drowsiness_model.pkl")

print("model trained & saved")
print("accuracy:",acc)