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

### Threshold

Every other classification plot says how good the ranking is. This one answers
what follows: given that ranking, where do you cut?

```elixir
EvalViz.threshold_curve(y_true, scores)
```

Precision, recall and F1 against the decision threshold, with the best F1
marked.

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
```

## License

Apache-2.0
