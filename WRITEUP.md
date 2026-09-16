# Writeup: A Decision Tree Built from Scratch on Iris

## 1. Objective

The goal was to implement the CART classification tree end to end — impurity
measure, split search, recursive growth, prediction, and evaluation — without
calling any ready-made estimator, and then to use it to show where a tree's
capacity actually comes from. Iris is the dataset: 150 samples, four continuous
measurements, three balanced species.

Iris is small and famously separable, which makes it a poor benchmark but a good
teaching instrument. Everything interesting here is about the *mechanics* of the
tree, not about beating a score.

## 2. Data and setup

The dataset loads through `seaborn`, drops nulls (there are none), and encodes
`species` to integers with `LabelEncoder`, giving the alphabetical mapping
`setosa=0, versicolor=1, virginica=2`. Features are the four measurements, in
the order `sepal_length, sepal_width, petal_length, petal_width`.

Splitting is hand-rolled: permute the indices with a seeded
`np.random.default_rng`, take the first 75% as train. With `seed=1` that yields
112 training rows and 38 test rows, with test class counts of 12 / 14 / 12 —
close enough to balanced that accuracy is a fair summary statistic.

All numbers below come from an actual run using the scikit-learn copy of Iris.
That copy differs from seaborn's in two cell values (the well-known
UCI-vs-Fisher discrepancy at rows 35 and 38), so a threshold may differ in the
third decimal, but no conclusion depends on it.

## 3. Impurity and the split search

Gini impurity of a label set is `1 - Σ pᵢ²`. On the training set this comes out
at **0.666**, which is exactly what three equally likely classes should give:
`1 - 3 × (1/3)² = 2/3`. That number is the starting point the tree has to reduce.

Candidate thresholds are generated per feature by `candidate_thresholds`. If a
column has 20 or fewer distinct values it tries all of them; otherwise it takes
20 quantiles spanning the 5th to 95th percentile. This is a deliberate
efficiency trade: exhaustive threshold search is O(n) per feature per node, and
the quantile grid caps it at 20.

The trade has a visible cost. The best first split prints as:

```
petal_length <= 2.036   gain 0.3363
```

The true boundary between setosa and everything else sits at petal length 1.9
(setosa maxes out at 1.9, versicolor starts at 3.0). Any threshold in that gap
is equally correct, and the quantile grid happened to land on 2.036. Accuracy is
unaffected, but the printed tree reads as though 2.036 were meaningful. A
midpoint-based candidate generator — sort the unique values, take
`(vᵢ + vᵢ₊₁)/2` — would print 2.45 or 1.9 and be easier to interpret. Worth
changing if the tree is meant to be read by a human.

That single split removes half the total impurity in one question: a gain of
0.336 against a parent Gini of 0.666. This is the central fact about Iris —
setosa is linearly separable from the other two on a single petal measurement,
and the remaining difficulty is entirely versicolor-vs-virginica.

## 4. Growing the tree

`build_tree` recurses with three stopping rules: depth cap, minimum samples to
attempt a split, and node purity. Each leaf stores the majority label and its
sample count. There is no pruning pass — this is pre-pruning only, via the
stopping rules.

The depth-3 tree, with `min_samples_split=10`:

```
petal_length <= 2.036 ?
   yes: -> setosa  (n=38)
   no:  petal_width <= 1.6 ?
      yes: petal_length <= 4.9 ?
         yes: -> versicolor  (n=34)
         no:  -> versicolor  (n=2)
      no:  petal_length <= 4.8 ?
         yes: -> virginica  (n=3)
         no:  -> virginica  (n=35)
```

Two things stand out. First, all four splits are on petal measurements; sepal
width and sepal length never get chosen. Second, the bottom two splits are
**useless** — both children of `petal_length <= 4.9` predict versicolor, and both
children of `petal_length <= 4.8` predict virginica. The tree spent two
questions and gained nothing. This is what an unpruned tree does when the
stopping rule is depth rather than gain: `best_split` returns a positive-gain
split, the recursion happily takes it, and the resulting partition is invisible
at prediction time. A minimum-gain threshold, or post-pruning by collapsing
leaves with identical predictions, would remove both.

Note also the `feature_subset_size` argument. It is wired through `build_tree`
into `best_split` but not exercised by default. It is the hook for a random
forest: sample a subset of features at each node, build many trees on bootstrap
samples, vote. The implementation currently uses `np.random.choice`, which draws
from the legacy global RNG rather than the seeded `default_rng` used elsewhere,
so that path is not reproducible.

## 5. Depth sweep and the overfitting question

| depth | train | test | leaves |
|---|---|---|---|
| 1 | 0.679 | 0.632 | 2 |
| 2 | 0.982 | 0.895 | 3 |
| 3 | 0.982 | 0.895 | 5 |
| 4 | 1.000 | 0.921 | 7 |
| 5 | 1.000 | 0.921 | 7 |
| 6+ | 1.000 | 0.921 | 7 |

The sweep uses `min_samples_split=2`, so depth is the only active constraint.

The curve is the textbook shape only at the left end. A depth-1 stump gets
0.679 train — it can only answer "setosa or not", capping it near two-thirds.
Depth 2 jumps to 0.982 train and 0.895 test. By depth 4 the training set is
perfectly fit at 7 leaves.

What does *not* happen is the expected divergence. Test accuracy goes **up** to
0.921 at depth 4 and then flatlines. Past depth 4 the tree stops growing
entirely — 7 leaves at depth 4 and 7 leaves at depth 20 — because every leaf is
pure, and purity is a stopping rule. The depth cap becomes inactive.

So this sweep does not demonstrate overfitting; it demonstrates saturation.
Iris has only ~4 misclassifiable points near the versicolor/virginica boundary,
and with 112 training rows a 7-leaf tree memorises them without needing to carve
the space into noise-fitting slivers. To actually show overfitting you would
need a noisier or higher-dimensional dataset, or injected label noise — flip 15%
of the training labels and the test curve will turn over as advertised.

One detail worth flagging: the depth-4 tree in the sweep scores 0.921, while the
depth-4 tree evaluated earlier in the script scores 0.895. Same depth, different
result. The difference is `min_samples_split` — 2 in the sweep, 5 by default
elsewhere. On a dataset this small, the sample-count rule binds harder than the
depth rule, and it costs one test point. Worth knowing before quoting "the
depth-4 accuracy" as a single number.

Feature ablation makes the same point from the other side. Dropping any one of
`sepal_length`, `sepal_width`, or `petal_length` leaves test accuracy at 0.895;
dropping `petal_width` leaves it at 0.921. No single feature is load-bearing,
because petal length and petal width are highly correlated and either can carry
the split. A tree's feature importances are unstable in exactly this situation.

## 6. Confusion matrix

Depth-4 tree, `min_samples_split=5`, on the 38 test rows:

```
TRUTH↓ / PRED→   setosa  versicolor  virginica
setosa               12           0          0
versicolor            0          13          1
virginica             0           3          9
```

| class | precision | recall |
|---|---|---|
| setosa | 1.000 | 1.000 |
| versicolor | 0.812 | 0.929 |
| virginica | 0.900 | 0.750 |

Setosa is perfect, as the first split already implied. All four errors lie on
the versicolor/virginica boundary, and they are asymmetric: three virginica
called versicolor, one the other way. That asymmetry is why precision and recall
diverge for those two classes and why accuracy alone (0.895) understates the
problem — virginica recall is 0.750, meaning a quarter of true virginica are
missed.

The script's precision formulas are correct: `cm.ravel()` unpacks row-major, so
`aa/(aa+ba+ca)` is diagonal over column sum, which is precision. Recall is not
computed; it would be diagonal over row sum.

The printed baseline, however, is wrong. `1 - y_test.mean()` is a binary
majority-class formula. With labels in {0,1,2} the mean is ~1.0 and it prints
**0.000** — not a baseline, an artifact. The correct comparison is the largest
test class share, `np.bincount(y_test).max() / len(y_test)` = **0.368**. Against
that, 0.895 is a real result; against 0.000 the comparison is meaningless.

## 7. Conclusions

The implementation is correct in its core: Gini, split search, recursion,
prediction, and the precision calculation all do what they claim. The measured
result — 0.895 to 0.921 test accuracy against a 0.368 majority baseline — is in
line with what a depth-limited CART tree achieves on Iris.

The findings that generalise beyond this dataset:

1. **One feature can dominate.** Petal length alone removes half the impurity.
   Correlated features make individual importances unreliable.
2. **Depth is the wrong knob when leaves go pure.** Growth here stops at 7
   leaves regardless of the cap; `min_samples_split` is the constraint that
   actually binds, and it is worth a full test point.
3. **Unpruned trees produce dead splits.** Two of the depth-3 tree's questions
   partition data whose children share a prediction.
4. **Accuracy hides asymmetric error.** 0.895 looks uniform; virginica recall of
   0.750 is the real story.

The highest-value changes, in order: fix the baseline, switch to midpoint
thresholds, add a minimum-gain stopping rule, and report recall alongside
precision. After that, the `feature_subset_size` hook is one bootstrap loop away
from a working random forest — which would be the natural way to trade the
single tree's variance for a small accuracy gain.
