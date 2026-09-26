# Food.com Recipes and Interactions

Analysis and machine learning pipeline built on the [Food.com Recipes and
Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
dataset, covering data cleaning, feature engineering, clustering, and
classification of recipes and user reviews.

**Authors:** Anja Petrović, Marina Nešković

## Overview

The dataset combines two sources: recipe metadata (preparation time, steps,
ingredients, nutrition facts) and user interactions (ratings and text
reviews). The project builds a full data pipeline — preprocessing, clustering,
and classification — to explore structure in the recipes and patterns in user
behavior.

## Data preparation

- Loaded `RAW_recipes.csv` and `RAW_interactions.csv`, inspected structure,
  missing values, and data types.
- Removed recipes without a name; filled missing text fields with empty
  strings; confirmed recipe IDs are unique.
- Capped `minutes` at 1440 to remove unrealistic outliers, and replaced
  zero-minute entries with a minimum of 1.
- Parsed the `nutrition` field into separate numeric columns (calories, fat,
  sugar, sodium, protein, saturated fat, carbohydrates) and removed records
  with unrealistic calorie values.
- Converted `submitted` to a datetime format and extracted tag counts from
  `tags`.
- Cleaned the interactions dataset, checked rating distribution, and verified
  consistency between recipes and interactions (removing interactions that
  reference nonexistent recipes).
- Aggregated interactions per recipe into `avg_rating` and `num_ratings`,
  then merged with recipe data.
- One-hot encoded the `ingredients` list using `MultiLabelBinarizer`, using a
  sparse matrix representation and dropping rarely-used ingredients to reduce
  dimensionality.

## Clustering

Two clustering approaches were applied to explore natural groupings in the
data:

**Numerical features** — nutritional values and recipe characteristics were
scaled and reduced with PCA before applying K-Means. Multiple values of *k*
were evaluated using Silhouette score, Davies-Bouldin index, and
Calinski-Harabasz index to select the optimal number of clusters.

**Text reviews** — reviews were normalized and represented with TF-IDF, then
clustered with K-Means. Although the Silhouette score for text clustering was
low (~0.005, expected given short review length and repetitive vocabulary),
inspecting each cluster's top terms revealed meaningful thematic groups (e.g.
recipe modifications, ease-of-preparation comments, generic praise).

## Classification

A binary target `high_rating` was defined (rating ≥ threshold) and tested in
two separate experiments:

**Review-level classification** — each row is a single user interaction.
Tested on numeric features, TF-IDF text features, and a combination of both,
across five algorithms: Random Forest, Gradient Boosting, Logistic
Regression, KNN, and Naive Bayes. Text-based features outperformed
numeric-only models; Logistic Regression was the best performer, KNN the
weakest (accuracy 0.687).

**Recipe-level classification** — each row is one recipe with an average
rating and one-hot encoded ingredients. Same five algorithms tested; results
were closer together, with Gradient Boosting performing best (accuracy 0.600)
and Logistic Regression weakest (accuracy 0.540).

**Conclusion:** review-level classification with text features produced
noticeably better results than recipe-level classification based on
ingredient combinations alone.

## Tech stack

- Python
- pandas, NumPy
- scikit-learn (K-Means, PCA, Random Forest, Gradient Boosting, Logistic
  Regression, KNN, Naive Bayes, TF-IDF, MultiLabelBinarizer)
- matplotlib
- ast (for parsing stringified list columns)

## Dataset

This project uses the [Food.com Recipes and Interactions dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions),
specifically `RAW_recipes.csv` and `RAW_interactions.csv`. The dataset is not
included in this repository — download it from Kaggle and place the files in
the expected data directory before running the notebooks.

## Running the project

```bash
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in order to reproduce the preprocessing, clustering, and
classification steps.

