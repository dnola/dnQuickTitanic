import pandas as pd
import numpy as np

# Pipelines are dope:
class PandasEncoder:                # All we need to put something in a pipeline is a class with fit(self,X,y=None) and transform(self,X)
    def fit(self,X,y=None):
        X = pd.get_dummies(X) # literally means 'one hot encode' - its a new pandas function
        self.columns_in_test_data = X.columns
        return self                 # fit() returns self

    def transform(self,X): # transform() returns the transformed data
        return pd.get_dummies(X).reindex(columns = self.columns_in_test_data, fill_value=0) # Sometimes test data and train data have different columns in them after one hot encoding - so we throw away anything that we havent seen in train data

class Printer:
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        print("Printer called, data looks like:")
        print(np.array(X)[0])
        return X