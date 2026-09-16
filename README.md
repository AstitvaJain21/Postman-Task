# Postman-Task

Sorry I am so late i will make changes after BOSM most probably 

# Decision Tree from Scratch — Iris

A CART-style classification tree implemented with NumPy only (no `sklearn`
estimators). It trains on the Iris dataset and reports Gini impurity, the best
split, tree structure, a depth sweep, and a 3x3 confusion matrix with per-class
precision.

Exported from a Colab notebook (`Postman task.ipynb`). It runs top to bottom as
a plain script.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `seaborn`, `scikit-learn`


## Knobs worth changing

| Parameter            | Where                 | Effect |
| max_depth            | build_tree            | Hard cap on depth. |
| min_samples_split    | build_tree            | Minimum rows before a node may split. Matters more than `max_depth` here. |
| max_candidates       | candidate_thresholds  | Thresholds tried per feature (20). |
| test_size,  seed     | train_test_split      | Split ratio and shuffle; `seed=1` at the call site. |
| feature_subset_size  | build_tree            | Random feature sampling per node — the hook for a random forest. |


## What it prints
1.Dataset head and row count (150 rows, 112 train / 38 test).
2.Gini impurity of the training labels (~0.666 for three balanced classes).
3.The best first split, e.g. petal_length <= 1.9.
4.Train and test accuracy for a depth-4 tree.
5.An indented text rendering of a depth-3 tree.
6.A depth sweep over [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20] with leaf counts — this is where overfitting shows up: train accuracy climbs to 1.000 while test accuracy plateaus.
7.A 3x3 confusion matrix plus per-class precision.


## Everything lives in one file, in this order:
1.data loading and encoding → 
2.train_test_split / accuracy → 
3.gini → 
4.candidate_thresholds / best_split → 
5.Node / build_tree → 
6.predict → 
7.print_tree → 
8.depth sweep → 
9.confusion_matrix and precision.


for my commits you can check my colab notebook
Link:https://colab.research.google.com/drive/1LnhQJ8BFs9t2NzAotMZfJuQVuysJlguG?usp=sharing
