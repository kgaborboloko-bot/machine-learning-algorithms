# machine-learning-algorithms

## Overview

This project applies supervised and unsupervised machine learning techniques
across three real-world datasets: healthcare dementia diagnosis, food delivery
time prediction, and wine variety clustering.

---

## Question 1 & 2 — Dementia Classification (Feature Selection + ML Models)

Built a classification model to predict dementia diagnosis from a dataset of
2,149 patients containing demographic, lifestyle, medical, and cognitive features.

- Preprocessed data: handled missing values, encoded categorical variables,
  and applied feature standardisation
- Implemented two feature selection methods:
  - Filter-based: SelectKBest
  - Wrapper-based: Recursive Feature Elimination (RFE)
- Trained and evaluated two models using both feature sets:
  - Logistic Regression
  - Decision Tree

### Results

The Decision Tree consistently outperformed Logistic Regression across both
feature selection methods, achieving over 92% accuracy and 94%+ precision.
RFE slightly improved both models compared to SelectKBest. Precision was
prioritised as a key metric given the high cost of false positives in a
medical diagnosis context.

**Key concepts:** Scikit-learn, feature selection, classification,
precision/recall, confusion matrix, model evaluation

---

## Question 3 — Uber Eats Delivery Time Prediction (Linear Regression + Gradient Descent)

Developed a regression model to predict food delivery times using historical
order data from Uber Eats.

- Cleaned data and handled missing values
- Encoded categorical variables (weather, traffic, vehicle type, time of day)
- Applied feature selection and standardisation
- Implemented Linear Regression using Gradient Descent
- Evaluated model performance using standard regression metrics

### Results

The model achieved an R² of 0.756, meaning it explains approximately 75.6%
of the variance in delivery times. The MAE of 7.29 minutes indicates
reasonable prediction accuracy for practical delivery time estimation.

**Key concepts:** Linear regression, gradient descent, feature engineering,
regression metrics

---

## Question 4 — Wine Clustering (K-Means)

Used K-Means clustering to identify natural groupings in wine chemical
properties across 13 features for an Italian wine consortium.

- Standardised all features prior to clustering
- Applied the elbow method to determine the optimal number of clusters,
  with the silhouette score used for validation (score: 0.285)
- Visualised the final clusters using a colour-coded scatter plot

### Results

The elbow method indicated an optimal cluster count of 3, consistent with
the three known wine cultivars in the dataset. The resulting clusters show
clear separation in the feature space, with the K-Means model successfully
distinguishing between the three wine varieties based on their chemical
profiles.

**Key concepts:** K-Means, unsupervised learning, elbow method,
silhouette scoring, cluster visualisation, Scikit-learn

---

## Tools & Libraries

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
