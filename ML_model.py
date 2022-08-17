## Machine learning library and functions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## pandas
import pandas as pd

## Saving model
import pickle

#load the dataset
iris_bunch = load_iris()
iris_df = pd.DataFrame(iris_bunch['data'], columns=iris_bunch['feature_names'])
print(iris_df.head(3))

target = iris_bunch['target']

## Split the train and the test (Logistic regression in this case)
X_train, X_test, y_train, y_test = train_test_split(iris_df, target, random_state=1, test_size=0.2)  # It takes two features, train data and test data (target)
                                                                                                     # Random state is for everyone in the class to have the same results
                                                                                                     # test_size states that 20% of the data is included in the split
#Create the model
logistic = LogisticRegression(max_iter=1000)

#Training the model
logistic.fit(X_train, y_train)

# This is only to check the model, not needed :)
## prediction
#print(logistic.predict(X_test))
## Evaluate
#print(logistic.score(X_test, y_test))

# Save the model
pkl_file = 'logistic_model.p'

with open(pkl_file, 'wb') as file:
    pickle.dump(logistic, file)   # pickle.dump(<What you want to save>, <where you want to save>)