# Regression Algorithms

Regression algorithms estimate continuous numerical outputs based on input features.

- **Linear Regression** – Models the linear relationship between features and a continuous target variable.
- **Ridge Regression / RidgeCV** – L2-regularized regression; RidgeCV includes cross-validation for automatic α selection.
- **Lasso Regression / LassoCV** – L1-regularized regression promoting sparsity; LassoCV tunes penalty strength.
- **Elastic Net / ElasticNetCV** – Combines L1 and L2 penalties; ElasticNetCV automatically tunes them.
- **Bayesian Ridge Regression** – Probabilistic linear regression that infers distributions over weights with Bayesian priors.
- **Support Vector Regression (SVR / LinearSVR / NuSVR)** – SVM-based regression variants: kernelized SVR, linear version for scalability, and NuSVR for control over support vectors.
- **KNeighborsRegressor / RadiusNeighborsRegressor** – Instance-based regressors predicting from neighbors in feature space.
- **DecisionTreeRegressor** – Tree model that splits features to predict continuous outcomes.
- **RandomForestRegressor / ExtraTreesRegressor** – Ensembles of randomized decision trees for robust regression.
- **GradientBoostingRegressor / HistGradientBoostingRegressor** – Sequential boosting with decision trees; histogram-based variant is optimized for large data.
- **AdaBoostRegressor** – Boosting ensemble that iteratively reweights hard-to-predict samples.
- **HuberRegressor / TheilSenRegressor / RANSACRegressor** – Robust regression estimators designed to reduce outlier influence.
- **QuantileRegressor** – Estimates conditional quantiles, useful for modeling heteroscedasticity and prediction intervals.
- **Orthogonal Matching Pursuit / LARS / LassoLars** – Sparse solvers and greedy algorithms for feature selection in regression.
- **SGDRegressor / PassiveAggressiveRegressor** – Online/large-scale linear models using stochastic updates.
- **GaussianProcessRegressor** – Nonparametric probabilistic regression modeling distributions over functions.
- **MultiOutputRegressor / RegressorChain / TransformedTargetRegressor** – Wrappers for multi-target regression and transformed targets.
