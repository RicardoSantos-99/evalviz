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

### Learning curve

```elixir
EvalViz.learning_curve(train_sizes, train_scores, validation_scores)
```

This one does not train anything: pass the scores you already measured, the
same split scikit-learn draws between computing a learning curve and displaying
one. Give it a score per fold and the mean is drawn with a band one standard
deviation wide.

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
