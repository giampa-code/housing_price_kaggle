# Housing price kaggle

This notebooks will be developed for educational porpouses to apply the knowledge acquired in Coursera's Machine Learning course.

# Steps

## 1. Feature Engineering / Data manipulaton

The data was downloaded from the kaggle competition.
It consist of 79 features of houses to predict the sale price.
d
Reviewing, NaN treatment, categorical encoding and normalization were made in this part.

The files used were named feature_engineering_vx, corresponding to each version of the processed data:

- v0: The categorical encoding was made taking in acount that many of the categorical features have some ordinal properties, so we want to keep that information.

- v1: cuadratic values were added for the most_relevant_features (MRF) of the regularized LR.

- v2: Starting with outlier treatment.

## 2. Models

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
The model performs between 0.7 and 0.85, but in some weird cases, the R2 score gets almost infinite negative values. I could not sort out why this was happening.
Varying a little bit the regularization parameter alpha has a very big impact in the MAE result of Kaggle.
Later I will do some research.

