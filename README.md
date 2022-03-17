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

v2: The MSSubClass was re-encoded to find a tendency between the labels.

### Feature correlation


### Outlier treatment

V2: around 15 outlier point were dropped and several data columns were dropped because the target seemed independent from them.

### Normalization

Most of the features don't follow a normal distribution and the mean/median/mode are distortioned due to high "Not apply" or zero values. Each feature was divided by it's range to scale it between 0 and 1.


### Additional features

V1: quadratic and squared powers were added to the most relevant features (selected as the N features that weight represents the 80% of the total).

V2: The feature Baths was created like a combination of all the baths features.

### Versions

The files used were named feature_engineering_vx, corresponding to each version of the processed data:

- v0: The categorical encoding was made taking in acount that many of the categorical features have some ordinal properties, so we want to keep that information.

- v1: cuadratic values were added for the most_relevant_features (MRF) of the regularized LR.

- v2: Extra analysis for outliers and non useful funcs. Extra 2, 3 and 4 powers to the 15 most correlated features.


## 2. Models result summary

Diverse ML models were used, starting from plain Linear Regression 

- Linear Regression R2_mean = 0.77   MAE = 20662

- Regularized LR (Ridge with alpha=3 optimized for R2) R2_mean = 0.822 and kaggle MAE = 20142
- Regularized LR (Ridge with alpha=12 optimized for MAE) R2_mean = 0.822 and kaggle  MAE = 20142 (same results)
- Regularized LR with MRF powers (Ridge with alpha=0.4) R2_mean = 0.862 and kaggle MAE = 91591
- Regularized LR with MRF powers (Ridge with alpha=1) R2_mean = 0.86 and kaggle MAE = 21748 
- Regularized LR with 15 MRF powers (2,3,4) (alpha = 0.1) R2_mean = 0.922 and kaggle MAe = 14805
- Regularized LR with 15 MRF powers (2,3,4) (alpha = 0.05) R2_mean = 0.922 and kaggle MAe = 14496

- XGBoost (FE data v0) R2 = 0.92 and kaggle MAE = 18243
- XGBoost (FE data v1) R2 = 0.92 and kaggle MAE = 18243 (The same)
- XGBoost (FE data v2) R2 = 0.92 and kaggle MAE = 18243


## 3. Problems found

In the model analysis notebook I made some analysis of how R2 score varies with changing the random_state and the test_size of the split fuction.

![R2 = f(random_state]( ../feature_engineering/R2_f(random_state).png?raw=true "R2 = f(random_state" )


The model performs between 0.7 and 0.85, but in some weird cases, the R2 score gets almost infinite negative values. I could not sort out why this was happening.

Sometimes the models perform well with the test set (cv), but very bad in the Kaggle set, like there is a lot of noise.

Good test set performance. HORRIBLE kaggle performance:

It happend to me that I get very good test (my "test" set, the cross validation) set performance (R2=0.92, MAE = 15k) but when I upload the solution to Kaggle I end with a 4 BILLION MAE Score. (???)


Varying a little bit the regularization parameter alpha has a very big impact in the MAE result of Kaggle.
Later I will do some research.

## 4. Notes

A relativelly good result was accomplished with the first model of Linear regression with the data of FEv0 (MAE=~20k)

No major improvement could be done manipulating the datasets using the Linear Regression model with the FE v0 and v1.
A significant improvement was acomplished using the alogorithm that everybody uses for the competition (XGBoost). (MAE=~18k).

With FE v2 (that includes some extra preprocessing and powers of the 15 MRF), significant improvement was acomplished.
With Regularized LR kaggle MAE = 14440 and for XGBoost MAE = 14849. With this submission I achieved the 518th place, inside the 2% best submissions.

It was shown that optimizing the regularization parameter for the R2 score of the cross validation set was not the best option.
Increasing the regularization, hence lowering the variance, showed better results in the Kaggle test set.

In a next step, I will add a 0.5 power and include more MRFs.







