import sklearn.pipeline as skpipe
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.cross_validation as cv
from helpers import PandasEncoder
import numpy as np

pd.options.mode.chained_assignment = None # Pandas is whiny, so we shut it up

data = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
train_labels = data['Survived']
test_ids = data_test['PassengerId']

def fix_frame(frame): # Do some cleaning
    frame['Age'] = frame['Age'].fillna(20)
    frame['Pclass'] = frame['Pclass'].astype('category') # By default pandas doesn't encode numbers - but Pclass is probably better as a category - we want to one hot encode it
    return frame

features = ["Pclass","Sex","Age","SibSp","Embarked","Cabin"]
train = fix_frame(data[features])
test = fix_frame(data_test[features])


pipe = skpipe.Pipeline([
    ('to_numbers',PandasEncoder()), # One hot encodes a pandas frame - everything except numbers
    ('random_forest',RandomForestClassifier(n_estimators=100))
])

score = np.mean(cv.cross_val_score(pipe,train,train_labels,cv=5,n_jobs=-1))
print("Estimated score over 5 tests:",score)

pipe.fit(train,train_labels)
predictions = pipe.predict(test)

pd.DataFrame(predictions,index=test_ids,columns=["Survived"]).to_csv("submission.csv")
print("Done writing to file")



