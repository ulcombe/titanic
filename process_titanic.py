# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Import data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print df_test.head()

# Figures inline and set visualization style
# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
data.info()

print data['Age']
# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
print data['Age']

# Check out info of data
data.info()

data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()

# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()

data.info()

data_train = data.iloc[:891]
data_test = data.iloc[891:]

X = data_train.values
test = data_test.values
y = survived_train.values

# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#            splitter='best')

# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred

df_test[['PassengerId', 'Survived']].to_csv('1st_dec_tree.csv', index=False)

