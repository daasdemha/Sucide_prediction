# 9789180-SI-s1

## Suicide risk prediction - Classification problem

### Aims

This project aims to predict the risk factor of a person committing suicide so it could be prevented using a machine learning model.

### Dataset
The dataset used in this project was obtained from kaggle.com and it is called “suicidedataextrafestures” it was collected by the Kaggle user DORNA NIROOMAND. The dataset has 26 columns containing 5 categorical and 21 continuous variables. The data is collected from the year 1985 to 2016 from 48 different countries. The data is from 2 sexes, male and female and 6 different age groups. There are 17 columns in the dataset with 15110 rows; the rest of the columns have rows of different lengths. Outliers and null values are present in the dataset.

The dataset used in this project is available online at: https://www.kaggle.com/dornani/widandsuicide?select=suicidedataextrafestures.csv

### pre processing 
Firstly, the data frame is imported, columns with high nan values and columns with repeated information are removed.
Nan values are again checked from the remaining column and are replaced using the mean of the 10 K nearest neighbours.
After observation it was determined that the outliers are providing valuable informational so they were kept.
Now all columns of type objects containing categorical features are one hot encoded except the country column.
A heat map is plotted so the highest correlated features can be identified and be assigned as a new data frame to a variable.  They will be used later to test whether they improve the model’s performance or not.
The country column is one hot encoded and after the encoding, there are a total of 64 columns in the dataset. Labels 1, 2, 3, 4, 5 are made using the sucidesper100k column. 1 being minimum risk, 2 being low risk, 3 being medium risk, 4 being high risk and 5 being maximum risk. This was done so there are enough categories to model and the model is not too computationally heavy.
The label samples are observed for each of the newly created label classes that will be used to perform oversampling. Tthe label samples are imbalanced. 1 having 9264 samples comprises 61.31% of the data, 2 having 2850 samples covers 18.86 % of the data 3 having 2177 samples covers 14.41 % of the data, 4 having 678 samples covers 4.49 % of the data and 5 having 141 samples covers 0.93 % of the total data in the label column suicideper100k.
The features x and x2 are selected using the original dataset and the dataset obtained after using correlation respectively. The label y is used for both features. Principal component analysis is applied to all 9 continuous features in the dataset. 
After applying PCA it was determined that the components could be reduced to 6 dimensions for above 90 % of data to be kept, so it was decided that PCA would not have a great impact on the model evaluation, and they were not utilized for model evaluation. An Oversampled label is selected y_over so that results from the original feature, selected using correlation matrix and oversampled data, could be applied to different machine learning algorithms and the results could be evaluated. All these selected features are split into testing and training sets. It is split 80 % into the training and 20 % into the testing sets with the random state set to 1.

### Model Selection
Supervised classification learning algorithms will be performed in this project. Cross-validation was performed on Logistic, KNN (K nearest neighbours), Naive Bayes, Neural Networks, Decision tree and Random Forest models. It was decided that Decision tree with a score of 93 %, Random forest scoring 94 % and KNN having a score of 71 % had the best cross-validated accuracy. 

### Model optimization
Different hyperparameter optimization techniques were applied to the three selected models. They were applied using the following logic and order. First cross-validation was applied to narrow down on which hyperparameters give the best results individually, then random search and grid search was applied to a range of hyperparameters that had the most effect on increasing the accuracy of the model. Accuracy was chosen as a measure to optimize the hyperparameters as it was important how many correct predictions the model made for the test dataset. 

After the hyperparameter selection. All the models were applied on all 3 feature variables X, X2 and X_over and the  continuous features  of the X and X_over variables were feature scaled using Standardscaler, MinMaxscaler, Robustscaler and Normaliser feature scaling techniques. 

#### KNN
KNN had the best testing and training set results with hyperparameters 3 neighbours, weight set as distance using MinMaxScalar and it had similar overall scores using other scaling techniques, but the AUC (Area under the curve) was highest using the MinMaxScalar with 3 neighbours.

When evaluated with cross validation it performed best using 15 neighbours, weight set as distance with 80.22 % accuracy and 1.40 % standard deviation.

#### Random Forest
Random Forest had the best overall results on the testing and training sets with 1 min_samples_leaf ,2  min_samples_split ,  None max_leaf_nodes ,  0.0 min_weight_fraction_leaf , 'entropy' criterion, None max_features, 121 max_depth,  'random' splitter as hyperparameters score with MinMaxScalar and it had similar overall scores using other scaling techniques, but the AUC (Area under the curve) was highest using the MinMaxScalar. The defaults model results were 93.11 % with 0.86 % standard deviation When evaluated with cross validation.


#### Decision Tree
Decision Tree has the best overall results with 1 min_samples_leaf ,2  min_samples_split ,  None max_leaf_nodes ,  0.0 min_weight_fraction_leaf , 'entropy' criterion, None max_features, 121 max_depth,  'random' splitter as hyperparameters with MinMaxScalar
has the best area under the curve. The defaults model results were 93.11 % with 0.86 % standard deviation. The results after using minmax scaler with hyperparameter optimization were 94.44 % with 1.05 standard deviation.

After using the hyperparameters obtained by model optimization the models are tested using cross validation on the whole dataset using  accuracy, precision, recall, f1 score and area under the curve as measures. 

### Conclusion

Random forest outperforms the others models in all measures with the lowest standard deviation.

The algorithms are predicting the risk of suicide in individuals, and thus looking at actual positive data, how many times it is predicted correctly is more important than predicting something positive, how many times they were actually positive recall will have a high priority. It will not be the only scoring method as a model with 100% recall could always output a positive prediction, it would have 100% recall but be completely uninformative. so f1 score will be used to evaluate the model as it gives recall and precision the same priority. 

the Random Forest model has the highest performance when predicting the f1 score with a mean f1 score 0.950965 and the mean accuracy is 0.956131 with 10 cross validations. 

Random Forest  with hyperparameters 1 min_samples_leaf ,2  min_samples_split ,  None max_leaf_nodes ,  0.0 min_weight_fraction_leaf , 'entropy' criterion, None max_features, 121 max_depth,  'random' splitter gave give 0.956131 accuracy, 0.961529  precision, 0.954255  recall, 0.950965  f1 score and 0.998497 Area under the curve when run with 10 cross validations using MinMaxScalar gave the best results.


### Code

The code can be found in the Sucide_Prediction.ipynb file it is commented fully and all functions have proper docstrings for use and undestanding. 

A demonstration video running and exapling the code can also be found in Project_videio.mp4 file.

The data used is found in suicidedataextrafestures.csv file.
