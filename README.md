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
- [ ] ROC curve
- [ ] Precision-recall curve
- [ ] DET curve

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

## License

Apache-2.0
