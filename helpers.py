import pandas as pd

# Pipelines are dope:
class PandasEncoder:                # All we need to put something in a pipeline is a class with fit(self,X,y=None) and transform(self,X)
    def fit(self,X,y=None):
        X = pd.get_dummies(X) # literally means 'one hot encode' - its a new pandas function
        self.columns = X.columns
        return self                 # fit() returns self

    def transform(self,X): # transform() returns the transformed data
        return pd.get_dummies(X).reindex(columns = self.columns, fill_value=0) # Sometimes test data and train data have different values in them - so we throw away anything that we havent seen in train data
