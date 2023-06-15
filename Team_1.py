#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,confusion_matrix,accuracy_score,recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree 

def genderize(df):
    df.rename(columns={'Sex': 'Gender','sex':'gender'},inplace=True)
    
def drop_irrelevant_cols(df):
    df.drop(columns=['Name', 'PassengerId','Ticket'], axis=1, inplace=True)
    
def dummy_cols (df):
    df= pd.get_dummies(data=df, columns=['Gender','Embarked'], drop_first=True)
    return df

#combining Sbps and Parch into Fam
def combining_sbps_parch_to_fam(df):
    df['Fam']= df['Parch'] + df['SibSp']
    df.drop(['Parch', 'SibSp'], axis=1, inplace=True)
    
def drop_cabin(df):
    df.drop(columns=['Cabin'], axis=1, inplace=True)
    
categories = ['Survived', 'Pclass', 'Age', 'Fare' ]
def checking_outliers(categories):
    for c in categories:
        sns.boxplot(data[c])
        plt.title(f"{c}")
        plt.grid() 
        plt.show()
def bins (df):
    df.loc[df['Age']<=5, 'Age'] = 1
    df.loc[(df['Age'] > 5) & (df['Age'] <= 20), 'Age']=2
    df.loc[(df['Age'] > 20) & (df['Age'] <=64), 'Age']=3
    df.loc[(df['Age'] > 64) & (df['Age'] <= 80), 'Age']=4
    
def fill_nan_age(data):
   data['Age']= data['Age'].fillna(3)
   return data

def bin_fare(data):
    data['Fare'], cut_bin = pd.qcut(data['Fare'], q = 6, labels = [1, 2, 3, 4, 5, 6],retbins = True)
    
def create_feature_target(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y

def scale(df):    
    scaler = MinMaxScaler()
    scaler.fit(X)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df
# Rochel Dewick
def knn_fit():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(metric = 'minkowski', n_neighbors=5, weights = 'distance', algorithm='brute' )
    knn.fit(X,y)
    return knn

# Hadassa Taller
def Logistic_Regression():
    model = LogisticRegression(solver='liblinear', penalty='l2' , fit_intercept=True, random_state=25)
    model.fit(X,y)
    return model

# Libby Roberts
def decision_tree_classifier_function():
    dt = DecisionTreeClassifier()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    dt.fit(X, y)
    predictions = dt.predict(X)
    cv_scores = cross_val_score(dt, X, y, cv=10)
    accuracy = accuracy_score(predictions, y)
    val_prediction = dt.predict(X_val)
    dt_accuracy_val = accuracy_score(val_prediction, y_val)
    print("Train accuracy: ", accuracy, " Validation accuracy: ", dt_accuracy_val)
    TN, FP, FN, TP = confusion_matrix(y_val, val_prediction).ravel()
    recall=recall_score(y_val, val_prediction)
    precision = precision_score(y_val, val_prediction)
    print('recall:', recall)
    print('precision:', precision)
    return dt


# Shani Rubin
def random_forest():
    print('random forset',X.columns)
    X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=42)
    rf = RandomForestClassifier(n_estimators=30, max_depth=3,min_samples_leaf= 6,random_state=42)
    rf.fit(X_train, y_train)
    rf_pred_train = rf.predict(X_train)
    rf_accuracy_train = accuracy_score(rf_pred_train, y_train)
    rf_pred_test = rf.predict(X_test)
    rf_accuracy_test = accuracy_score(rf_pred_test, y_test)
    print(rf_accuracy_train, rf_accuracy_test)

# Zeesel Kitay    
def forest_ranger():
    print('forest ranger',X.columns)
    # x=X.drop(columns=['Fam','Enbarked_Q'],axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=142,test_size=.16)
    rf = RandomForestClassifier(n_estimators=100,random_state=95, max_depth=6)
    rf.fit(X_train, y_train)
    rf_pred_train = rf.predict(X_train)
    rf_accuracy_train = accuracy_score(rf_pred_train, y_train)
    rf_pred_test = rf.predict(X_test)
    rf_accuracy_test = accuracy_score(rf_pred_test, y_test)
    print(rf_accuracy_train, rf_accuracy_test)
    return rf

def pred_to_csv(model, name):
    test = pd.read_csv('./test.csv')
    genderize(test)
    drop_irrelevant_cols(test)
    combining_sbps_parch_to_fam(test)
    test = dummy_cols(test)
    drop_cabin(test)
    fill_nan_age(test)
    test['Fare'].fillna(6.2375 , inplace=True)
    bins(test)
    bin_fare(test)
    preds = pd.DataFrame(model.predict(test))
    test_new = pd.read_csv('./test.csv')
    test_pass = pd.DataFrame(test_new['PassengerId'])
    result =pd.concat([test_pass, preds], axis=1)
    result.rename({0:'Survived'}, axis=1, inplace=True)
    result.to_csv(name+'_team1.csv',index=False)


data = pd.read_csv('./train.csv')

genderize(data)
drop_irrelevant_cols(data)
combining_sbps_parch_to_fam(data)
data = dummy_cols(data)
drop_cabin(data)

#Replacing outliers in Fare to the avg of the outliers
avg = (data['Fare'][(data['Fare'] > 85) & (data['Fare'] <= 300)]).mean()
data['Fare'][data['Fare'] > 85] = avg

fill_nan_age(data)
bins(data)
data.drop(columns=['Fam', 'Embarked_Q'], axis=1, inplace=True)
data['Fare'], cut_bin = pd.qcut(data['Fare'], q = 6, labels = [1, 2, 3, 4, 5, 6],retbins = True)

X, y = create_feature_target(data)


X = scale(X)

model = forest_ranger()
pred_to_csv(model, 'rf')


model = knn_fit()
pred_to_csv(model, 'kn')


model =Logistic_Regression()
pred_to_csv(model, 'lr')

model = decision_tree_classifier_function()
pred_to_csv(model, 'dt')















