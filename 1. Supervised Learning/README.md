# **Supervised Learning**

Supervised learning is a paradigm where models are trained on labeled datasets, meaning each input sample is paired with a known output.  
The goal is to learn a mapping from inputs (features) to outputs (targets), so the model can make predictions on unseen data.  

It is broadly divided into two categories:  
- **Regression** → predicting continuous values  
- **Classification** → predicting discrete categories  

## **1. Regression Algorithms**

Regression algorithms estimate continuous numerical outputs based on input features.

- **Linear Regression** – Models the linear relationship between features and a continuous target variable.
- **Ridge Regression / RidgeCV** – L2-regularized regression; RidgeCV includes cross-validation for automatic α selection.
- **Lasso Regression / LassoCV** – L1-regularized regression promoting sparsity; LassoCV tunes penalty strength.
- **Elastic Net / ElasticNetCV** – Combines L1 and L2 penalties; ElasticNetCV automatically tunes them.
- **Bayesian Ridge Regression** – Probabilistic linear regression that infers distributions over weights with Bayesian priors.
- **Support Vector Regression (SVR / LinearSVR / NuSVR)** – SVM-based regression variants: kernelized SVR, linear version for scalability, and `NuSVR` for control over support vectors.
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

## **2. Classification Algorithms**

Classification algorithms assign discrete categorical labels to input samples.

- **K-Nearest Neighbors (KNN) / RadiusNeighborsClassifier** – Classifies samples by majority vote among nearest or radius-defined neighbors.
- **Support Vector Classifiers (SVC / LinearSVC / NuSVC)** – SVM-based classifiers: kernelized SVC, linear variant for scalability, and `NuSVC` with support vector control.
- **Logistic Regression** – Linear classifier modeling probabilities for binary or multiclass outcomes, with regularization.
- **DecisionTreeClassifier** – Tree-based model that predicts class labels via recursive feature splits.
- **RandomForestClassifier / ExtraTreesClassifier** – Ensembles of randomized/bagged decision trees for robust classification.
- **GradientBoostingClassifier / HistGradientBoostingClassifier** – Boosted tree ensembles providing high predictive performance.
- **AdaBoostClassifier** – Adaptive boosting that increases weight on misclassified samples.
- **Gaussian Naive Bayes / MultinomialNB / BernoulliNB** – Naive Bayes variants suited to continuous, count-based, and binary features respectively.
- **Linear Discriminant Analysis (LDA) / Quadratic Discriminant Analysis (QDA)** – Generative models separating classes via linear or quadratic decision boundaries.
- **SGDClassifier / PassiveAggressiveClassifier / Perceptron** – Online/iterative linear classifiers for large-scale datasets.
- **NearestCentroid** – Classifies samples by assigning them to the nearest class centroid.
- **GaussianProcessClassifier** – Kernel-based probabilistic classifier modeling distributions over functions.
- **CalibratedClassifierCV** – Wrapper that calibrates classifier probability outputs (Platt scaling or isotonic regression).
- **MultiOutputClassifier / ClassifierChain** – Wrappers for handling multi-label and multi-output classification.

## Reference
- [Scikit-learn Documentation: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)