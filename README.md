# EvalViz

Model evaluation plots for Nx tensors: confusion matrices, ROC, precision-recall
and DET curves.

Scholar already computes these metrics but returns tensors, and there is no
ready-made way to look at them. EvalViz turns that output into Vega-Lite specs,
which Livebook renders on its own.

Nothing here assumes a particular modelling library. Pass the tensors your model
produced and the plot follows, whether they came from Scholar, Axon or anywhere
else.

## Status

Early development. Not published to Hex yet.

- [x] Confusion matrix
- [x] ROC curve
- [x] Precision-recall curve
- [x] DET curve
- [x] Dendrogram
- [x] Silhouette plot
- [x] PCA scree plot
- [x] Calibration curve
- [x] Residuals and predicted vs actual
- [x] Threshold curve
- [x] Learning curve
- [x] Multiclass one-vs-rest, with micro and macro averages
- [x] Projection scatter, side by side
- [x] PCA biplot and loadings heatmap
- [x] Score distribution by true class
- [x] Elbow plot
- [x] Coefficients and correlation matrix
- [x] Residual histogram and normal Q-Q
- [x] Validation curve, grid search heatmap, per-fold scores
- [x] One-call report, and a grid to compose any plots
- [x] Model comparison across curves, calibration, regression and learning curves
- [x] Faceting, a panel per class when the overlay gets crowded

## Trying it

The plots are Vega-Lite specs, so Livebook renders them with no extra step:

```
open -a Livebook notebooks/tour.livemd
```

The tour walks through every plot with generated data, including cases where a
model is deliberately wrong so you can see what that looks like.

Rendering a spec in Livebook needs `kino_vega_lite`, not just `kino`: the
`Kino.Render` implementation for `VegaLite` ships in that package, and without
it Livebook prints the struct rather than drawing the chart.

## The short version

One call for the whole picture:

```elixir
EvalViz.report(y_true, scores, class_names: ["negative", "positive"])
```

A confusion matrix, the ROC and precision-recall curves, and the scores split
by true class. The matrix and the histogram read the same `:threshold`, so the
report shows what one cut-off actually does rather than four unrelated views.

`kind: :regression` reads the second argument as predictions and gives
predicted against actual, the residuals, their distribution and a normal Q-Q.

To choose your own set, build the plots and lay them out:

```elixir
EvalViz.grid([
  EvalViz.roc_curve(y_true, scores, title: "ROC"),
  EvalViz.calibration_curve(y_true, scores, title: "Calibration"),
  EvalViz.silhouette(x, labels, num_clusters: 3, title: "Clusters")
])
```

## Usage

```elixir
y_true = Nx.tensor([0, 0, 1, 1, 2, 2])
y_pred = Nx.tensor([0, 1, 0, 2, 2, 2])

EvalViz.confusion_matrix(y_true, y_pred, num_classes: 3)
```

Every function returns a `VegaLite` struct, so you can keep customising it:

```elixir
EvalViz.confusion_matrix(y_true, y_pred,
  num_classes: 3,
  class_names: ["cat", "dog", "bird"],
  normalize: :true_class
)
|> VegaLite.title("Validation set")
```

### Curves

Pass the score your model gave the positive class:

```elixir
EvalViz.roc_curve(y_true, scores)
EvalViz.precision_recall_curve(y_true, scores)
EvalViz.det_curve(y_true, scores)
```

To compare models, pass a list and every curve lands on the same axes, with
AUC (or average precision) in the legend:

```elixir
EvalViz.roc_curve([
  {"Logistic", y_true, logistic_scores},
  {"Naive Bayes", y_true, nb_scores}
])
```

### Multiclass

Pass class indices and a `{num_samples, num_classes}` score matrix for one
one-vs-rest curve per class. ROC, precision-recall, DET, calibration and the
threshold curve all take this shape.

```elixir
EvalViz.roc_curve(y_true, probabilities,
  class_names: ["cat", "dog", "bird"],
  average: [:micro, :macro]
)
```

`:micro` pools every (sample, class) pair into one binary problem, so every
sample weighs the same. `:macro` averages the per-class curves, so every class
does, however few samples it has. Both match `scikit-learn`, curve values
included.

Past a handful of classes the overlay stops being readable. `facet: true` gives
each curve its own panel, titled with its name and summary, so the legend is no
longer needed:

```elixir
EvalViz.roc_curve(y_true, probabilities, facet: true, columns: 3)
```

Faceting pays for itself twice over on the other two. Precision-recall draws
each class's own baseline, which the overlay has to leave out because the
classes disagree on it. The threshold curve gets colour back for the metric, so
the dash pattern it was pushed onto is no longer needed.

### Dendrogram

```elixir
model = Scholar.Cluster.Hierarchical.fit(data)

EvalViz.dendrogram(model,
  labels: ["alpha", "bravo", "charlie"],
  color_threshold: 1.0
)
```

`color_threshold` colours each subtree that merges below that height as its own
cluster and leaves the links above it grey, which is the height you would cut
the tree at to read clusters off the plot.

Leaf order and bracket coordinates match `scipy.cluster.hierarchy.dendrogram`.

### Clustering and decomposition

```elixir
EvalViz.silhouette(x, labels, num_clusters: 3)

pca = Scholar.Decomposition.PCA.fit(x, num_components: 6)
EvalViz.scree(pca)
```

To choose k, hand `elbow/3` the models you fitted, or the values of k and any
score you computed for them:

```elixir
EvalViz.elbow(Enum.map(2..8, &Scholar.Cluster.KMeans.fit(x, num_clusters: &1)))
```

The rule marks the k furthest from the line joining the first and last points,
measured after rescaling both axes to `0..1`, since k spans single digits and
inertia can span thousands. A straight run of scores has no corner, and nothing
is marked rather than a k being invented.

### Projection

Any dimensionality reduction ends in a `{num_samples, num_components}` tensor,
so one function covers all of them: PCA, kernel PCA, truncated SVD, t-SNE, MDS
and TriMap. Structs carrying the result under `:embedding`, as MDS does, go in
directly.

```elixir
embedding = Scholar.Manifold.TSNE.fit(x, key: key, num_components: 2)

EvalViz.projection(embedding, y_true, label_names: ["cat", "dog", "bird"])
```

Both axes share one range by default. They measure the same kind of thing, so
letting them scale apart would stretch the cloud and invent structure that is
not there.

Pass a list of labellings to draw the same embedding side by side, which is how
a clustering gets checked against the truth:

```elixir
EvalViz.projection(embedding, [
  {"True class", y_true},
  {"KMeans", kmeans.labels}
])
```

### Biplot and loadings

The scree plot says how much variance each component carries. These say what
each one is made of.

```elixir
EvalViz.biplot(pca, x, feature_names: ["height", "weight", "age"])
EvalViz.loadings(pca, feature_names: ["height", "weight", "age"])
```

The biplot draws one arrow per feature over the projected points. Arrows and
points have unrelated units, so the arrows are stretched to sit against the
cloud: their directions and their lengths relative to each other carry the
meaning, not their absolute size.

### Threshold

Every other classification plot says how good the ranking is. This one answers
what follows: given that ranking, where do you cut?

```elixir
EvalViz.threshold_curve(y_true, scores)
```

Precision, recall and F1 against the decision threshold, with the best F1
marked.

```elixir
EvalViz.score_distribution(y_true, scores, threshold: 0.5)
```

One histogram per true class. The threshold curve says where to cut; this says
why the cut works or does not. Two humps that barely touch mean the model
separates the classes; two sitting on top of each other mean no threshold will
save it. Each class sums to one by default, which keeps a rare class visible
next to a common one.

### Learning curve

```elixir
EvalViz.learning_curve(train_sizes, train_scores, validation_scores)
```

This one does not train anything: pass the scores you already measured, the
same split scikit-learn draws between computing a learning curve and displaying
one. Give it a score per fold and the mean is drawn with a band one standard
deviation wide.

Comparing models works here too, and answers a question one curve cannot: which
model is still getting better as the data grows.

```elixir
EvalViz.learning_curve(sizes, [
  {"Linear", linear_train, linear_validation},
  {"Forest", forest_train, forest_validation}
])
```

Colour then carries the model and the dash pattern carries training against
validation, so both readings survive at once.

The threshold curve is the one plot that takes no list of models: colour and
dash are already spent on class and metric there, and it exists to tune one
model's cut-off rather than to compare models.

### Model selection

The learning curve asks whether more data would help. These three work with the
data you have.

```elixir
EvalViz.validation_curve(alphas, train_scores, validation_scores,
  param_name: "alpha",
  scale: :log
)
```

Score against one hyperparameter, with the best marked. Parameter values may be
numbers, and then the axis can be logarithmic, which is how regularisation is
usually swept. Anything else, booleans and atoms included, goes on a nominal
axis in the order given.

```elixir
results = Scholar.ModelSelection.grid_search(x, y, folding_fun, scoring_fun, opts)

EvalViz.grid_search(results, best: :min, metric_name: "MSE")
```

Takes what `Scholar.ModelSelection.grid_search/5` returns and lays it out as a
heatmap. Without `:x` and `:y` it uses the two hyperparameters that vary; name
two when more vary and the rest are collapsed by keeping the best score per
cell, which the subtitle says out loud. Darker always means better, so
`best: :min` reverses the ramp rather than leaving you to invert it by eye.

```elixir
scores = Scholar.ModelSelection.cross_validate(x, y, folding_fun, scoring_fun)

EvalViz.fold_scores(scores, metric_names: ["MSE", "MAE"])
```

A mean says nothing about how much it moved between folds. This is that spread,
one panel per metric. The folds are not joined by a line: they are
interchangeable, and a line would show a trend across an order that means
nothing.

### Calibration

```elixir
EvalViz.calibration_curve(y_true, probabilities, bins: 10)
```

Needs real probabilities rather than arbitrary scores: it asks whether, among
the cases the model called 70% likely, roughly 70% turned out positive.

### Regression

```elixir
EvalViz.predicted_vs_actual(y_true, y_pred)
EvalViz.residuals(y_true, y_pred)
EvalViz.residual_distribution(y_true, y_pred)
EvalViz.qq_plot(Nx.subtract(y_pred, y_true))
```

The first two also take a list of models, which puts them on one set of axes
and one reference line:

```elixir
EvalViz.residuals([
  {"Linear", y_true, linear_pred},
  {"Ridge", y_true, ridge_pred}
])
```

`residuals/3` shows the residuals against the prediction, which is where a
systematic pattern shows up. `residual_distribution/3` shows their shape, and
`qq_plot/2` checks them against a normal: points on the line mean normal, a
curve at one end means that tail is heavier, an S means both are.

Point positions and the fitted line match `scipy.stats.probplot`.

### Coefficients and correlation

```elixir
EvalViz.coefficients(model, feature_names: ["age", "income", "height"])
EvalViz.correlation(x, feature_names: ["age", "income", "height"])
```

`coefficients/2` reads any model exposing `:coefficients`, ordered by magnitude,
with a bar per class when the model fits one coefficient per class. They are
only comparable across features when the features were scaled to begin with.

`correlation/2` pins its colour scale to `-1..1` rather than fitting it, so a
matrix of weak correlations does not colour like a matrix of strong ones.

## License

Apache-2.0
