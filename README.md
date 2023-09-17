# ML_Lab_1

**Name : P.G.M.S.Gunarathna** 

**Index no: 190200X**

**Colab link : [Lab_1_Colab_NoteBook**](https://colab.research.google.com/drive/11RJR7X_VPeabcDPl4rfxlnO8RJTbanVz?usp=sharing)**

1. **Filter Method**

For filter method I have used statistical tests as f\_classif. K is the number of features to be selected. The best K numbers of features will be selected with this method.

from sklearn.feature\_selection import SelectKBest,f\_classif

chi2\_features = SelectKBest(f\_classif, k=70)

x\_kbest = chi2\_features.fit\_transform(x\_train['label\_1'], y\_train['label\_1'])

x\_kbest.shape

1. **Random Forest**

Select features based on the importance scores using RandomForest. Here I have found the importance scores of each feature and sorted them according to it. Then dropped N number of features that are less important.

model = RandomForestClassifier(n\_estimators=100)

model.fit(x\_train['label\_1'], y\_train['label\_1'])  

impotance = model.feature\_importances\_

final\_df = pd.DataFrame({"Features":pd.DataFrame(x\_train['label\_1']).columns,"importance":impotance})

final\_df.set\_index('importance')

final\_df\_sorted = final\_df.sort\_values('importance')

\# Get the indices of the least important features to remove

indices\_to\_remove = final\_df\_sorted["Features"][:202]

\# Remove the least important features from the training dataset

X\_train\_removed = np.delete(x\_train['label\_1'], indices\_to\_remove, axis=1)

x\_valid\_removed = np.delete(x\_valid['label\_1'], indices\_to\_remove, axis=1)

1. **PCA**

Pricipal Componet Analysis method gave me the least number of features with better accuracy.

from sklearn.decomposition import PCA

pca = PCA(n\_components= 0.84, svd\_solver="full")

pca.fit(x\_train['label\_1'])

x\_train\_pca = pca.transform(x\_train['label\_1'])

x\_valid\_pca = pca.transform(x\_valid['label\_1'])


1. **Ridge regression**

This is an embedded method I have used to select features.  Which takes the coefficients values and remove the least important features.

ridge\_reg = Ridge(alpha =1.0)

ridge\_reg.fit(x\_train['label\_1'],y\_train['label\_1'])

coefs = ridge\_reg.coef\_

final\_df = pd.DataFrame({"Features":pd.DataFrame(x\_train['label\_1']).columns,"coefs":coefs})

final\_df.set\_index('coefs')

final\_df\_sorted = final\_df.sort\_values('coefs')

\# Get the indices of the least important features to remove

indices\_to\_remove = final\_df\_sorted["Features"][:170]

\# Remove the least important features from the training dataset

X\_train\_removed = np.delete(x\_train['label\_1'], indices\_to\_remove, axis=1)

x\_valid\_removed = np.delete(x\_valid['label\_1'], indices\_to\_remove, axis=1)

1. **Select From Model**

SelectFromModel is a feature selection meta-transformer in scikit-learn that allows you to select features based on the importance scores provided by an underlying estimator (model). 

selector = SelectFromModel(model, threshold='mean')  # You can adjust the threshold as needed

x\_train\_selected = selector.fit\_transform(x\_train['label\_1'], y\_train['label\_1'])

The threshold parameter determines how features are selected. It can take various forms: mean or median. I tested both ways. 

selector = SelectFromModel(model, threshold='median')  # You can adjust the threshold as needed

x\_train\_selected\_2 = selector.fit\_transform(x\_train['label\_1'], y\_train['label\_1'])

## <a name="_63wno6bipilg"></a>**Label\_1**
To predict the labels I have used K Nearest Neighbor(KNN) model with a accuracy of 99%. Then I have used following methods to feature selection.

- Filter Method
- Random Forest 
- PCA
- Ridge regression
- Select From Model

Following graph shows the accuracy of each approach after feature selection


|**Approach**|**No. Features**|**Accuracy**|
| :- | :- | :- |
|Filter Method|70|0\.98|
|Random Forest |54|0\.98|
|PCA|37|0\.98|
|Ridge regression|86|0\.98|
|Select From Mode|105|0\.98|
|Random Forest, PAC|46|0\.97|


I have used the **PCA approach** and predicted the test data set since it gives the least number of features with a better accuracy compared to the other approaches.
## <a name="_9geedt6rkc0a"></a>**Label\_2**
To predict the labels I have used regression model with a accuracy of 23.83 mean square error. Then I have used following methods to feature selection.

- Filter Method
- Random Forest 
- PCA
- Select From Model

Following graph shows the accuracy of each approach after feature selection



|**Approach**|**No. Features**|**MSE**|
| :- | :- | :- |
|Filter Method|200|25\.43|
|Random Forest |116|27\.98|
|PCA|106|27\.94|
|Select From Mode|128|27\.83|

I used PCA approach and then predict the label\_2 of test data set since it gives the least number of features among the methods I tried.
## <a name="_1yt3c7rf76kc"></a>**Label\_3**
To predict the labels I have used K Nearest Neighbor(KNN) model with a accuracy of 100%. Then I have used following methods to feature selection.

- Filter Method
- PCA

Following graph shows the accuracy of each approach after feature selection



|**Approach**|**No. Features**|**Accuracy**|
| :- | :- | :- |
|Filter Method|19|1\.00|
|PCA|12|1\.00|

Both the methods I have tried gave me almost same accuracy. I choosed the PCA approach which gives the least number of features to predict the test data set. 
## <a name="_d19nx41hpnkd"></a>**Label\_4**
To predict the labels I have used Support Vector Machine model with a accuracy of 93%. Then I have used following methods to feature selection.

- Filter Method
- Randon Forest 
- PCA

Following graph shows the accuracy of each approach after feature selection.

The label 4 values had to be balance. So, I set the class\_weight parameter in SVC while training and predicting using the model.



|**Approach**|**No. Features**|**Accuracy**|
| :- | :- | :- |
|Filter Method|190|0\.86|
|Randon Forest |176|0\.91|
|PCA|106|0\.86|


I have used the Random Forest with importance score to predict the test data set since it gives a better accuracy than other methods.

## <a name="_p337m3i862je"></a>**Conclusion** 
Commanly the PCA gives a better feature selection approach.

We combine and use the methods get different feature combinations

Sort by importance score is also a better approach.
