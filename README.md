# ML_Lab_1

Name : P.G.M.S.Gunarathna 
Index no: 190200X
Colab link : Lab_1_Colab_NoteBook

Filter Method
For filter method I have used statistical tests as f_classif. K is the number of features to be selected. The best K numbers of features will be selected with this method.

from sklearn.feature_selection import SelectKBest,f_classif


chi2_features = SelectKBest(f_classif, k=70)
x_kbest = chi2_features.fit_transform(x_train['label_1'], y_train['label_1'])


x_kbest.shape

Random Forest
Select features based on the importance scores using RandomForest. Here I have found the importance scores of each feature and sorted them according to it. Then dropped N number of features that are less important.

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train['label_1'], y_train['label_1'])  

impotance = model.feature_importances_
final_df = pd.DataFrame({"Features":pd.DataFrame(x_train['label_1']).columns,"importance":impotance})
final_df.set_index('importance')
final_df_sorted = final_df.sort_values('importance')


# Get the indices of the least important features to remove
indices_to_remove = final_df_sorted["Features"][:202]


# Remove the least important features from the training dataset
X_train_removed = np.delete(x_train['label_1'], indices_to_remove, axis=1)
x_valid_removed = np.delete(x_valid['label_1'], indices_to_remove, axis=1)
PCA
Pricipal Componet Analysis method gave me the least number of features with better accuracy.

from sklearn.decomposition import PCA


pca = PCA(n_components= 0.84, svd_solver="full")
pca.fit(x_train['label_1'])
x_train_pca = pca.transform(x_train['label_1'])
x_valid_pca = pca.transform(x_valid['label_1'])


Ridge regression
This is an embedded method I have used to select features.  Which takes the coefficients values and remove the least important features.
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

Select From Model
SelectFromModel is a feature selection meta-transformer in scikit-learn that allows you to select features based on the importance scores provided by an underlying estimator (model). 
selector = SelectFromModel(model, threshold='mean')  # You can adjust the threshold as needed
x_train_selected = selector.fit_transform(x_train['label_1'], y_train['label_1'])

The threshold parameter determines how features are selected. It can take various forms: mean or median. I tested both ways. 


selector = SelectFromModel(model, threshold='median')  # You can adjust the threshold as needed
x_train_selected_2 = selector.fit_transform(x_train['label_1'], y_train['label_1'])

Label_1
To predict the labels I have used K Nearest Neighbor(KNN) model with a accuracy of 99%. Then I have used following methods to feature selection.
Filter Method
Random Forest 
PCA
Ridge regression
Select From Model
Following graph shows the accuracy of each approach after feature selection

Approach
No. Features
Accuracy
Filter Method
70
0.98
Random Forest 
54
0.98
PCA
37
0.98
Ridge regression
86
0.98
Select From Mode
105
0.98
Random Forest, PAC
46
0.97


I have used the PCA approach and predicted the test data set since it gives the least number of features with a better accuracy compared to the other approaches.
Label_2
To predict the labels I have used regression model with a accuracy of 23.83 mean square error. Then I have used following methods to feature selection.
Filter Method
Random Forest 
PCA
Select From Model
Following graph shows the accuracy of each approach after feature selection


Approach
No. Features
MSE
Filter Method
200
25.43
Random Forest 
116
27.98
PCA
106
27.94
Select From Mode
128
27.83


I used PCA approach and then predict the label_2 of test data set since it gives the least number of features among the methods I tried.
Label_3
To predict the labels I have used K Nearest Neighbor(KNN) model with a accuracy of 100%. Then I have used following methods to feature selection.
Filter Method
PCA
Following graph shows the accuracy of each approach after feature selection


Approach
No. Features
Accuracy
Filter Method
19
1.00
PCA
12
1.00


Both the methods I have tried gave me almost same accuracy. I choosed the PCA approach which gives the least number of features to predict the test data set. 
Label_4
To predict the labels I have used Support Vector Machine model with a accuracy of 93%. Then I have used following methods to feature selection.
Filter Method
Randon Forest 
PCA
Following graph shows the accuracy of each approach after feature selection.
The label 4 values had to be balance. So, I set the class_weight parameter in SVC while training and predicting using the model.


Approach
No. Features
Accuracy
Filter Method
190
0.86
Randon Forest 
176
0.91
PCA
106
0.86



I have used the Random Forest with importance score to predict the test data set since it gives a better accuracy than other methods.

Conclusion 
Commanly the PCA gives a better feature selection approach.
We combine and use the methods get different feature combinations
Sort by importance score is also a better approach.
