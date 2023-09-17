# -*- coding: utf-8 -*-
"""Lab1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11RJR7X_VPeabcDPl4rfxlnO8RJTbanVz

## Lab 1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Read data"""

names=[]
# Assign column names to the dataset
for i in range(1,257):
 names.append("feature_"+str(i))
labels=["label_1","label_2","label_3","label_4"]
names+=labels

# Read in the dataset
train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Lab 1/DataSet/train.csv')
valid_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Lab 1/DataSet/valid.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Lab 1/DataSet/Test.csv')

train_df.head()

train_df.shape

train_df.isnull().sum()

valid_df.isnull().sum()

test_df.isnull().sum()

"""Scaling"""

from sklearn.preprocessing import RobustScaler

x_train ={}
y_train ={}
x_valid ={}
y_valid ={}
x_test = {}

df_t = train_df
df_v = valid_df

for label in labels:
  scaler = RobustScaler()
  if label == 'label_2':
    df_t = train_df.dropna()
    df_v = valid_df.dropna()

  x_train[label] = scaler.fit_transform(df_t.drop(labels, axis=1))
  y_train[label] = df_t[label]
  x_valid[label] = scaler.transform(df_v.drop(labels, axis=1))
  y_valid[label] = df_v[label]
  x_test[label] = scaler.transform(test_df.drop(labels, axis=1))

pd.DataFrame(x_train['label_2']).shape

"""##KNN"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def KNN(x_t,x_v,y_t):
  classifier = KNeighborsClassifier(n_neighbors=5)
  classifier.fit(x_t, y_t)

  y_pred = classifier.predict(x_v)
  return y_pred

  # print(confusion_matrix(y_v, y_pred))
  # print(classification_report(y_v, y_pred))

"""## SVM"""

def SVM(x_t,x_v,y_t,y_v):
  svm_classifier = SVC(kernel='linear', class_weight='balanced')
  svm_classifier.fit(x_t, y_t)

  # Make predictions on the test data
  y_pred = svm_classifier.predict(x_v)

  # Evaluate model performance
  print(classification_report(y_v, y_pred))

"""## Regression"""

from sklearn.linear_model import LinearRegression  # You can replace this with other regression algorithms
from sklearn.metrics import mean_squared_error, r2_score  # For evaluation
from sklearn.ensemble import RandomForestRegressor

def reg(x_t,x_v,y_t,y_v):
  regressor = LinearRegression()
  regressor.fit(x_t, y_t)

  y_pred = regressor.predict(x_v)

  mse = mean_squared_error(y_v, y_pred)
  r2 = r2_score(y_v, y_pred)

  print("Mean Squared Error:", mse)
  print("R-squared:", r2)

"""## Label 1"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train['label_1'], y_train['label_1'])

y_pred = classifier.predict(x_valid['label_1'])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

"""### Filter Method"""

from sklearn.feature_selection import SelectKBest,f_classif

# scaler = RobustScaler()
# x_train_scaled = scaler.fit_transform(x_train[['label_1']])
# y_train_scaled = scaler.transform(y_train[['label_2']])

chi2_features = SelectKBest(f_classif, k=70)
x_kbest = chi2_features.fit_transform(x_train['label_1'], y_train['label_1'])

x_kbest.shape

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_kbest, y_train['label_1'])

y_pred = classifier.predict(chi2_features.transform(x_valid['label_1']))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

"""### Estimator Model"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier  # You can replace this with any other model

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train['label_1'], y_train['label_1'])  # Assuming x_train and y_train are your feature and target data

selector = SelectFromModel(model, threshold='mean')  # You can adjust the threshold as needed
x_train_selected = selector.fit_transform(x_train['label_1'], y_train['label_1'])

x_train_selected.shape

x_test_selected = selector.transform(x_valid['label_1'])  # Assuming x_test is your test data

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_selected, y_train['label_1'])

y_pred = classifier.predict(x_test_selected)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

selector = SelectFromModel(model, threshold='median')  # You can adjust the threshold as needed
x_train_selected_2 = selector.fit_transform(x_train['label_1'], y_train['label_1'])

x_train_selected_2.shape

x_test_selected_2 = selector.transform(x_valid['label_1'])  # Assuming x_test is your test data

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_selected_2, y_train['label_1'])

y_pred = classifier.predict(x_test_selected_2)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

"""### Random Forest"""

impotance = model.feature_importances_
final_df = pd.DataFrame({"Features":pd.DataFrame(x_train['label_1']).columns,"importance":impotance})
final_df.set_index('importance')
final_df_sorted = final_df.sort_values('importance')
final_df_sorted

# Get the indices of the least important features to remove
indices_to_remove = final_df_sorted["Features"][:202]

# Remove the least important features from the training dataset
X_train_removed = np.delete(x_train['label_1'], indices_to_remove, axis=1)
x_valid_removed = np.delete(x_valid['label_1'], indices_to_remove, axis=1)

X_train_removed.shape

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_removed, y_train['label_1'])

y_pred = classifier.predict(x_valid_removed)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

"""### PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components= 0.84, svd_solver="full")
pca.fit(x_train['label_1'])
x_train_pca = pca.transform(x_train['label_1'])
x_valid_pca = pca.transform(x_valid['label_1'])

x_train_pca.shape

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_pca, y_train['label_1'])

y_pred = classifier.predict(x_valid_pca)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

label_1_pred_before = KNN(x_train["label_1"],x_test['label_1'],y_train['label_1'])

from sklearn.neighbors import KNeighborsClassifier
x_tets_pca = pca.transform(x_test['label_1'])
label_1_pred_after = classifier.predict(x_tets_pca)

pd.DataFrame(x_tets_pca)

label1_features = pd.DataFrame(data=x_tets_pca, columns=[f'new_feature_{i+1}' for i in range(x_tets_pca.shape[1])])
label1_features.insert(0,'Predicted labels before feature engineering',label_1_pred_before)
label1_features.insert(1,'Predicted labels after feature engineering', label_1_pred_after)
label1_features.insert(2,'No of new features', x_tets_pca.shape[1])

def write_csv(feature_df, label):
  for i in range(feature_df['No of new features'][0], 256):
        feature_df[f'new_feature_{i+1}'] = pd.NA
  filename = f'/content/drive/MyDrive/Colab Notebooks/Lab 1/DataSet/190200X_label_{label}.csv'
  feature_df.to_csv(filename,index=False)

write_csv(label1_features,"label_1")

"""### Random Forest + PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components= 0.99, svd_solver="full")
pca.fit(X_train_removed)
x_train_pca = pca.transform(X_train_removed)
x_valid_pca = pca.transform(x_valid_removed)

x_train_pca.shape

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_pca, y_train['label_1'])

y_pred = classifier.predict(x_valid_pca)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))



"""### Ridge"""

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha =1.0)
ridge_reg.fit(x_train['label_1'],y_train['label_1'])
coefs = ridge_reg.coef_
final_df = pd.DataFrame({"Features":pd.DataFrame(x_train['label_1']).columns,"coefs":coefs})
final_df.set_index('coefs')
final_df_sorted = final_df.sort_values('coefs')

# Get the indices of the least important features to remove
indices_to_remove = final_df_sorted["Features"][:170]

# Remove the least important features from the training dataset
X_train_removed = np.delete(x_train['label_1'], indices_to_remove, axis=1)
x_valid_removed = np.delete(x_valid['label_1'], indices_to_remove, axis=1)

X_train_removed.shape

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_removed, y_train['label_1'])

y_pred = classifier.predict(x_valid_removed)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_1'], y_pred))
print(classification_report(y_valid['label_1'], y_pred))

"""## Label 2"""

from sklearn.linear_model import LinearRegression  # You can replace this with other regression algorithms
from sklearn.metrics import mean_squared_error, r2_score  # For evaluation
from sklearn.ensemble import RandomForestRegressor

regressor = LinearRegression()  # You can replace this with other regression algorithms
regressor.fit(x_train['label_2'], y_train['label_2'])

y_pred = regressor.predict(x_valid['label_2'])

mse = mean_squared_error(y_valid['label_2'], y_pred)
r2 = r2_score(y_valid['label_2'], y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

label_2_pred_before = regressor.predict(x_test['label_2'])

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Create and train the Gradient Boosting Regressor
regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
regressor.fit(x_train['label_2'], y_train['label_2'])

# Make predictions
y_pred = regressor.predict(x_valid['label_2'])

# Evaluate model performance
mse = mean_squared_error(y_valid['label_2'], y_pred)
r2 = r2_score(y_valid['label_2'], y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

"""### Filter Method"""

from sklearn.feature_selection import SelectKBest,f_regression

# scaler = RobustScaler()
# x_train_scaled = scaler.fit_transform(x_train[['label_1']])
# y_train_scaled = scaler.transform(y_train[['label_2']])

chi2_features = SelectKBest(f_regression, k=200)
x_kbest = chi2_features.fit_transform(x_train['label_2'], y_train['label_2'])

x_kbest.shape

regressor = LinearRegression()  # You can replace this with other regression algorithms
regressor.fit(x_kbest, y_train['label_2'])

y_pred = regressor.predict(chi2_features.transform(x_valid['label_2']))

mse = mean_squared_error(y_valid['label_2'], y_pred)
r2 = r2_score(y_valid['label_2'], y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

"""### Estimator Model"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier  # You can replace this with any other model

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train['label_2'], y_train['label_2'])  # Assuming x_train and y_train are your feature and target data

selector = SelectFromModel(model, threshold='mean')  # You can adjust the threshold as needed
x_train_selected = selector.fit_transform(x_train['label_2'], y_train['label_2'])

x_train_selected.shape

x_test_selected = selector.transform(x_valid['label_2'])  # Assuming x_test is your test data

regressor = LinearRegression()  # You can replace this with other regression algorithms
regressor.fit(x_train_selected, y_train['label_2'])

y_pred = regressor.predict(x_test_selected)

mse = mean_squared_error(y_valid['label_2'], y_pred)
r2 = r2_score(y_valid['label_2'], y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

selector = SelectFromModel(model, threshold='median')  # You can adjust the threshold as needed
x_train_selected_2 = selector.fit_transform(x_train['label_2'], y_train['label_2'])

x_train_selected_2.shape

x_test_selected_2 = selector.transform(x_valid['label_2'])  # Assuming x_test is your test data

regressor = LinearRegression()  # You can replace this with other regression algorithms
regressor.fit(x_train_selected_2, y_train['label_2'])

y_pred = regressor.predict(x_test_selected_2)

mse = mean_squared_error(y_valid['label_2'], y_pred)
r2 = r2_score(y_valid['label_2'], y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

"""### Random Forest"""

impotance = model.feature_importances_
final_df = pd.DataFrame({"Features":pd.DataFrame(x_train['label_2']).columns,"importance":impotance})
final_df.set_index('importance')
final_df_sorted = final_df.sort_values('importance')
final_df_sorted

# Get the indices of the least important features to remove
indices_to_remove = final_df_sorted["Features"][:140]

# Remove the least important features from the training dataset
X_train_removed = np.delete(x_train['label_2'], indices_to_remove, axis=1)
x_valid_removed = np.delete(x_valid['label_2'], indices_to_remove, axis=1)

regressor = LinearRegression()  # You can replace this with other regression algorithms
regressor.fit(X_train_removed, y_train['label_2'])

y_pred = regressor.predict(x_valid_removed)

mse = mean_squared_error(y_valid['label_2'], y_pred)
r2 = r2_score(y_valid['label_2'], y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

"""### PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components= 0.99, svd_solver="full")
pca.fit(x_train['label_2'])
x_train_pca = pca.transform(x_train['label_2'])
x_valid_pca = pca.transform(x_valid['label_2'])

x_train_pca.shape

regressor = LinearRegression()  # You can replace this with other regression algorithms
regressor.fit(x_train_pca, y_train['label_2'])

y_pred = regressor.predict(x_valid_pca)

mse = mean_squared_error(y_valid['label_2'], y_pred)
r2 = r2_score(y_valid['label_2'], y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

x_tets_pca = pca.transform(x_test['label_2'])
label_2_pred_after = regressor.predict(x_tets_pca)

pd.DataFrame(x_tets_pca)

label2_features = pd.DataFrame(data=x_tets_pca, columns=[f'new_feature_{i+1}' for i in range(x_tets_pca.shape[1])])
label2_features.insert(0,'Predicted labels before feature engineering',label_2_pred_before)
label2_features.insert(1,'Predicted labels after feature engineering', label_2_pred_after)
label2_features.insert(2,'No of new features', x_tets_pca.shape[1])

write_csv(label2_features,"label_2")

"""## Label 3"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train['label_3'], y_train['label_3'])

y_pred = classifier.predict(x_valid['label_3'])

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_3'], y_pred))
print(classification_report(y_valid['label_3'], y_pred))

label_3_pred_before = classifier.predict(x_test['label_3'])

"""### Filter method"""

from sklearn.feature_selection import SelectKBest,f_classif

# scaler = RobustScaler()
# x_train_scaled = scaler.fit_transform(x_train[['label_1']])
# y_train_scaled = scaler.transform(y_train[['label_2']])

chi2_features = SelectKBest(f_classif, k=19)
x_kbest = chi2_features.fit_transform(x_train['label_3'], y_train['label_3'])

x_kbest.shape

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_kbest, y_train['label_3'])

y_pred = classifier.predict(chi2_features.transform(x_valid['label_3']))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_3'], y_pred))
print(classification_report(y_valid['label_3'], y_pred))

"""### PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components= 0.55, svd_solver="full")
pca.fit(x_train['label_3'])
x_train_pca = pca.transform(x_train['label_3'])
x_valid_pca = pca.transform(x_valid['label_3'])

x_train_pca.shape

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train_pca, y_train['label_3'])

y_pred = classifier.predict(x_valid_pca)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid['label_3'], y_pred))
print(classification_report(y_valid['label_3'], y_pred))



x_tets_pca = pca.transform(x_test['label_3'])
label_3_pred_after = classifier.predict(x_tets_pca)

pd.DataFrame(x_tets_pca)

label3_features = pd.DataFrame(data=x_tets_pca, columns=[f'new_feature_{i+1}' for i in range(x_tets_pca.shape[1])])
label3_features.insert(0,'Predicted labels before feature engineering',label_3_pred_before)
label3_features.insert(1,'Predicted labels after feature engineering', label_3_pred_after)
label3_features.insert(2,'No of new features', x_tets_pca.shape[1])

write_csv(label3_features,"label_3")

"""## Label 4"""

from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear', class_weight='balanced')
svm_classifier.fit(x_train['label_4'], y_train['label_4'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid['label_4'])

# Evaluate model performance
print(classification_report(y_valid['label_4'], y_pred))

label_4_pred_before = svm_classifier.predict(x_test['label_4'])

"""### Random Forest"""

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier  # You can replace this with any other model

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train['label_4'], y_train['label_4'])  # Assuming x_train and y_train are your feature and target data

impotance = model.feature_importances_
final_df = pd.DataFrame({"Features":pd.DataFrame(x_train['label_4']).columns,"importance":impotance})
final_df.set_index('importance')
final_df_sorted = final_df.sort_values('importance')
final_df_sorted

# Get the indices of the least important features to remove
indices_to_remove = final_df_sorted["Features"][:80]

# Remove the least important features from the training dataset
X_train_removed = np.delete(x_train['label_4'], indices_to_remove, axis=1)
x_test_remove =  np.delete(x_test['label_4'], indices_to_remove, axis=1)

svm_classifier = SVC(kernel='linear', class_weight='balanced')
svm_classifier.fit(X_train_removed, y_train['label_4'])

# Make predictions on the test data
y_pred = svm_classifier.predict(np.delete(x_valid['label_4'], indices_to_remove, axis=1))

# Evaluate model performance
print(classification_report(y_valid['label_4'], y_pred))

"""j

### PCA
"""

from sklearn.decomposition import PCA

pca = PCA(n_components= 0.99, svd_solver="full")
pca.fit(x_train['label_4'])
x_train_pca = pca.transform(x_train['label_4'])
x_valid_pca = pca.transform(x_valid['label_4'])

x_train_pca.shape

svm_classifier = SVC(kernel='linear', class_weight='balanced')
svm_classifier.fit(x_train_pca, y_train['label_4'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(classification_report(y_valid['label_4'], y_pred))

"""### Filter Method"""

from sklearn.feature_selection import SelectKBest,f_classif

# scaler = RobustScaler()
# x_train_scaled = scaler.fit_transform(x_train[['label_1']])
# y_train_scaled = scaler.transform(y_train[['label_2']])

chi2_features = SelectKBest(f_classif, k=190)
x_kbest = chi2_features.fit_transform(x_train['label_4'], y_train['label_4'])

x_kbest.shape

svm_classifier = SVC(kernel='linear', class_weight='balanced')
svm_classifier.fit(x_train_pca, y_train['label_4'])

# Make predictions on the test data
y_pred = svm_classifier.predict(x_valid_pca)

# Evaluate model performance
print(classification_report(y_valid['label_4'], y_pred))



x_tets_pca = pca.transform(x_test['label_4'])
label_4_pred_after = svm_classifier.predict(x_test_remove)

pd.DataFrame(x_test_remove)

label4_features = pd.DataFrame(data=x_test_remove, columns=[f'new_feature_{i+1}' for i in range(x_test_remove.shape[1])])
label4_features.insert(0,'Predicted labels before feature engineering',label_4_pred_before)
label4_features.insert(1,'Predicted labels after feature engineering', label_4_pred_after)
label4_features.insert(2,'No of new features', x_test_remove.shape[1])

write_csv(label4_features,"label_4")