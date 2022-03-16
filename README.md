# Housing price kaggle

This notebooks will be developed for educational porpouses to apply the knowledge acquired in Coursera's Machine Learning course.

# Steps

## 1. Feature Engineering / Data manipulaton

The data was downloaded from the kaggle competition.
It consist of 79 features of houses to predict the sale price.
Reviewing, NaN treatment, categorical encoding, outlier treatment and normalization were made in this part.

### NaN Treatment

For the NaN treatment most of the features were full of NaN values that corresponded to NA 'Not Apply' insted. This NaNs were transformed back into the 'NA' values for later categorical encodig.

For GarageYrBlt and LotFrontage it was seen that the NaN values came from properties without a garage or without frontage respectively. So in this cases the NaN were filled with 0.

The rest of the NaNs of the train set were dropped. The rest of the NaNs for the test set were filled with the mode.

### Categorical encoding

Because there are son many features with categorical data, using dummy for all variables will leave the df with hundreds columns more.
Also, many of the categorical features have some ordinal properties, so we want to keep that information. Variables will be processed to asign numerical values going from 0 (worst) to n (best) (n=number of labels)

I.e. 

AllPub	All public Utilities (E,G,W,& S)	-->3

NoSewr	Electricity, Gas, and Water (Septic Tank) -->2

NoSeWa	Electricity and Gas Only  --> 1

ELO	Electricity only	--> 0

This process will be done automatically, grouping by the feature's labels and ranking by the mean of the target (SalePrice).


### Outlier treatment

### Normalization

Most of the features don't follow a normal distribution and the mean/median/mode are distortioned due to high "Not apply" or zero values. Each feature was divided by it's range to scale it between 0 and 1.


### Additional features

V1: quadratic and squared powers were added to the most relevant features (selected as the N features that weight represents the 80% of the total).

V2:

### Versions

The files used were named feature_engineering_vx, corresponding to each version of the processed data:

- v0: The categorical encoding was made taking in acount that many of the categorical features have some ordinal properties, so we want to keep that information.

- v1: cuadratic values were added for the most_relevant_features (MRF) of the regularized LR.

- v2: Extra analysis for outliers and non useful funcs. No powered funcs.


## 2. Models result summary

Diverse ML models were used, starting from plain Linear Regression 

- Linear Regression R2_mean = 0.77   MAE = 20662

- Regularized LR (Ridge with alpha=3 optimized for R2) R2_mean = 0.822 and kaggle MAE = 20142
- Regularized LR (Ridge with alpha=12 optimized for MAE) R2_mean = 0.822 and kaggle  MAE = 20142 (same results)
- Regularized LR with MRF powers (Ridge with alpha=0.4) R2_mean = 0.862 and kaggle MAE = 91591
- Regularized LR with MRF powers (Ridge with alpha=1) R2_mean = 0.86 and kaggle MAE = 21748 

- XGBosst (FE data v0) R2 = 0.92 and kaggle MAE = 18243
- XGBosst (FE data v1) R2 = 0.92 and kaggle MAE = 18243 (The same)


## 3. Problems found

In the model analysis notebook I made some analysis of how R2 score varies with changing the random_state and the test_size of the split fuction.

![R2 = f(random_state]( https://github.com/giampa14/housing_price_kaggle/blob/master/feature_engineering/R2_f(random_state).png )


The model performs between 0.7 and 0.85, but in some weird cases, the R2 score gets almost infinite negative values. I could not sort out why this was happening.

Varying a little bit the regularization parameter alpha has a very big impact in the MAE result of Kaggle.
Later I will do some research.

## 4. Process

![Alt text](url "Title")
