import pandas as pd

# Pipelines are dope:
class PandasEncoder:                # All we need to put something in a pipeline is a class with fit(self,X,y=None) and transform(self,X)
    def fit(self,X,y=None):
        X = pd.get_dummies(X)
        self.columns = X.columns
        return self                 # fit() returns self

    def transform(self,X):
        return pd.get_dummies(X).reindex(columns = self.columns, fill_value=0)      # transform() returns the transformed data
