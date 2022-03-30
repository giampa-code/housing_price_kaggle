# Housing price kaggle

This notebooks will be developed for educational porpouses to apply the knowledge acquired in Coursera's Machine Learning course.

# Steps

## 1 Feature Engineering / Data manipulaton

The data was downloaded from the kaggle competition.
It consist of 79 features of houses to predict the sale price.
Reviewing, NaN treatment, categorical encoding, outlier treatment and normalization were made in this part.

### 1.1 NaN Treatment

The data set cames with a NaN ratio of 6.3%.
Most of the features were full of NaN values that corresponded to NA 'Not Apply' or another type of categorical value instead.
For example, in the feature "Pool Quality" a house with no pool will be calified with NA = No Pool. This 'NA' was treated as a NaN in the dataset, so we need to return it back to its original value 'NA'.
This NaNs were transformed back into the 'NA' values for later categorical encodig.

For GarageYrBlt and LotFrontage it was seen that the NaN values came from properties without a garage or without frontage respectively. So in this cases the NaN were filled with 0.

The final NaN ratio was 0.014%

The rest of the NaNs of the train set were dropped. The rest of the NaNs for the test set were filled with the mode.

### 1.2 Categorical encoding

Because there are son many features with categorical data, using dummy for all variables will leave the df with hundreds columns more.
Also, many of the categorical features have some ordinal properties, so we want to keep that information. Variables will be processed to asign numerical values going from 0 (worst) to n (best) (n=number of labels)

I.e. 

AllPub	All public Utilities (E,G,W,& S)	-->3

NoSewr	Electricity, Gas, and Water (Septic Tank) -->2

NoSeWa	Electricity and Gas Only  --> 1

ELO	Electricity only	--> 0

This process will be done automatically, grouping by the feature's labels and ranking by the mean of the target (SalePrice).

v2: The MSSubClass was re-encoded to find a tendency between the labels.

![SalePrice=f(Neighborhood)](https://github.com/giampa14/housing_price_kaggle/blob/master/feature_engineering/scatter_Neighborhood_SalePrice.png/?raw=true)


### 1.3 Outliers treatment

V2: around 15 outlier point were dropped and several data columns were dropped because the target seemed independent from them.
V2_2: 10 outlier points and 11 features were dropped. It was shown in the charts that this features does not have much relation with the Sale Price.

![Outlier_treatment](https://github.com/giampa14/housing_price_kaggle/blob/master/feature_engineering/scatter_YrSold_SalePrice.png/?raw=true)

### 1.4 Adittional features

Having that much Bathroom features did not seemed to help. A unique 'Bath' feature was created with the following consideration: train['FullBath'] + 0.75 * train['BsmtFullBath'] + 0.5 * train['HalfBath'] + 0.375 * train['BsmtHalfBath']

### 1.5 Normalization

Most of the features don't follow a normal distribution and the mean/median/mode are distortioned due to high "Not apply" or zero values. Considering this, each feature was divided by it's range to scale it between 0 and 1.

For the case of the years variables, I will substract the min year first, because otherwise the final enconding will end with high values I.e. 1988/200 =~ 10


### 1.6 Feature correlation

A correlation heat map was made for the 15 most relevant/correlated features.

pic 'MRF_heatmap.png'
![MRF_heatmap](https://github.com/giampa14/housing_price_kaggle/blob/master/feature_engineering/MRF_heatmap.png/?raw=true)

With this MRF, it can be seen that the most significant variables are related to quality, space and neighborhood.


### 1.7 Polynomical features

V1: quadratic and squared powers were added to the most relevant features (selected as the N features that weight represents the 80% of the total).

V2_2: Powers 2, 3 and 4 have been added for the 15 MRF.


### 1.8 Versions

The files used were named feature_engineering_vx, corresponding to each version of the processed data:

- v0: The categorical encoding was made taking in acount that many of the categorical features have some ordinal properties, so we want to keep that information.

- v1: cuadratic values were added for the most_relevant_features (MRF) of the regularized LR.

- v2: Extra analysis for outliers and non useful funcs. Extra 2, 3 and 4 powers to the 15 most correlated features.

- v2_1: Idem v2, but without the powers. I will try to use this with a NN.

- v2_2: variation of v2 but dropping less outliners and features. This data showed the best results.

- v2_3: v2_2 but with the 5th power.

- v3: Idem v3, but with the 15 most correlated features and adding the 0.5 power and 5th. (This did not seem to make further improvement, indeed using more features (25) led to worser results)

## 2 Model

### 2.1 Splitting data and fitting the model

Train data was splitted with a test size of 0.4. Given we have another separated test set, this test set can be qualified as our cross validation set.

The best result were shown using Regularized Linear Regression (Ridge).
R2 score and mean absolute error (MAE) were used for results comparisson.

### 2.2 Veryfing results

I find that plotting the target value SalePrice with the most correlated feature in the x axis was a good way of visualizing the model accuracy. I made this plot comparnt the test target values with the test predicted values.

pic 'test_vs_test_pred.png'
![MRF_heatmap](https://github.com/giampa14/housing_price_kaggle/blob/master/feature_engineering/MRF_heatmap.png/?raw=true)

### 2.3 Looking for the best regularization parameter

Plotting the R2 score as a function of the regularization parameter shows that no substancial changes are happening between 0.01 and 1.

![R2=f(regularization)](https://github.com/giampa14/housing_price_kaggle/blob/master/models/R2=f(regularization).png/?raw=true)

Learning curves for these two regularization parameters were plotted. As the images show the cost fuction for the train and the test set are approaching asymptotically with a relativelly low final error.
This can indicate us that the model is not overfitted (high variance) nor underfitted (high bias).

![learning_curves_Alpha_0.01](https://github.com/giampa14/housing_price_kaggle/blob/master/models/learning_curves_Alpha_0_01.png/?raw=true)

![learning_curves_Alpha_1](https://github.com/giampa14/housing_price_kaggle/blob/master/models/learning_curves_Alpha_1.png/?raw=true)

Despite the good results, lower regularization parameter gives higher variance, hence more error in Kaggle. So it's important to find a trade-off between variance and bias.
For this, I plotted the squared sum of all the coefficients of the model and selected the highest regularization parameter where the squared sum started to decrease slowly. (lambda = 0.3)

![coefs_lambda](https://github.com/giampa14/housing_price_kaggle/blob/master/models/coefs_lambda.png/?raw=true)

For this exercise and this model, the best Kaggle result was achieved with lambda=1. Reaching the rank 491 of 36284. (Top 1.5%).


# 2. Models result summary

Diverse ML models were used, starting from plain Linear Regression 

- Linear Regression R2_mean = 0.77   MAE = 20662

- Regularized LR (Ridge with alpha=3 optimized for R2) R2_mean = 0.822 and kaggle MAE = 20142
- Regularized LR (Ridge with alpha=12 optimized for MAE) R2_mean = 0.822 and kaggle  MAE = 20142 
- Regularized LR with MRF powers (Ridge with alpha=0.4) R2_mean = 0.862 and kaggle MAE = 91591
- Regularized LR with MRF powers (Ridge with alpha=1) R2_mean = 0.86 and kaggle MAE = 21748 

- Regularized LR with 15 MRF powers (2,3,4) (alpha = 0.1) R2_mean = 0.922 and kaggle MAe = 14805
- Regularized LR with 15 MRF powers (2,3,4) (alpha = 0.05) R2_mean = 0.922 and kaggle MAe = 14496
- Regularized LR with 15 MRF powers (2,3,4) (alpha = 0.1) R2_mean = 0.922 and kaggle MAe = 14440
- Regularized LR with 15 MRF powers (2,3,4) (alpha = 2) R2_mean = 0.922 and kaggle MAe = 14470

- Regularized LR with 15 MRF powers (2,3,4) FE v2_2 alpha = 0.3  R2_mean = 0.924 and kaggle MAE = 14340
- Regularized LR with 15 MRF powers (2,3,4) FE v2_2 alpha = 0.4  R2_mean = 0.924 and kaggle MAE = 14324
- Regularized LR with 15 MRF powers (2,3,4) FE v2_2 alpha = 1  R2_mean = 0.925 and kaggle MAE = 14288 (best)
- Regularized LR with 15 MRF powers (2,3,4) FE v2_2 alpha = 1.5  R2_mean = 0.922 and kaggle MAE = 14296 

- Regularized LR with 15 MRF powers (2,3,4,5) FE v2_3 alpha = 0.1  R2_mean = 0.925 and kaggle MAE = 14609 
- Regularized LR with 15 MRF powers (2,3,4,5) FE v2_3 alpha = 1  R2_mean = 0.923 and kaggle MAE = 14305 

- XGBoost (FE data v0) R2 = 0.89 and kaggle MAE = 18243
- XGBoost (FE data v1) R2 = 0.89 and kaggle MAE = 18243 (The same)
- XGBoost (FE data v2) R2 = 0.91 and kaggle MAE = 14849
- XGBoost (FE data v2_2) R2 = 0.917 and kaggle MAE = 14413
- XGBoost (FE data v3) R2 = 0.91 and kaggle MAE = 14849

- Skl Neural network R2 = 0.77 and MAE = 21000 (I just copy some code, I used skl NN because I could not install tensorflow)


# 3. Notes

A relativelly good result was accomplished with the first model of Linear regression with the data of FEv0 (MAE=~20k)

No major improvement could be done manipulating the datasets using the Linear Regression model with the FE v0 and v1.
A significant improvement was acomplished using the alogorithm that everybody uses for the competition (XGBoost). (MAE=~18k).

With FE v2 (that includes some extra preprocessing and powers of the 15 MRF), significant improvement was acomplished.
With Regularized LR kaggle MAE = 14288 and for XGBoost MAE = 14849. With this submission I achieved the 491th place, inside the 1.5% best submissions.

It was shown that optimizing the regularization parameter for the R2 score of the cross validation set was not the best option.
Increasing the regularization, hence lowering the variance, showed better results in the Kaggle test set.

todo: I will add a 0.5 power and include more MRFs.
This approximation did not give better results, indeed, the result went worse.

todo: In next steps, I want to research about different algorithms and how to implement them in this competition. Also, using kfoldin may be usefull with the train set, it was shown that the accuracy varies with the random_state used in the split_data func.

todo: use something like sklearn.pipeline.Pipeline Refer. It helps you bundle all your transformations into a single object so you wouldn't miss a transformation by mistake. Also keeps your code clean!








