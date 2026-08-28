# Verdict

Model evaluation plots for [Nx](https://github.com/elixir-nx/nx). Confusion
matrices, ROC and precision-recall curves, calibration, residuals,
projections, and the rest of what you look at after training something.

Scholar computes these metrics and hands back tensors. Verdict turns tensors
into [Vega-Lite](https://vega.github.io/vega-lite/) specs, which Livebook
renders on its own.

Nothing here assumes a modelling library. Pass the tensors your model
produced, whether they came from Scholar, Axon, or a hand-rolled `defn`.

## Installation

Add to your `mix.exs`:

```elixir
def deps do
  [
    {:verdict, "~> 0.1.0"}
  ]
end
```

### Notebooks

```elixir
Mix.install([
  {:verdict, "~> 0.1.0"},
  {:kino_vega_lite, "~> 0.1"}
])
```

> #### kino_vega_lite, not kino {: .warning}
>
> The `Kino.Render` implementation for `VegaLite` ships in `kino_vega_lite`.
> Without it, Livebook prints the struct instead of drawing the chart.

## Getting started

One call for the usual screen:

```elixir
Verdict.report(y_true, scores, class_names: ["negative", "positive"])
```

That is a confusion matrix, the ROC and precision-recall curves, and the
scores split by true class. The matrix and the histogram read the same
`:threshold`, so it shows what one cut-off does rather than four unrelated
views. `kind: :regression` swaps in the regression set.

Every function returns a `VegaLite` struct, so you can keep going:

```elixir
Verdict.roc_curve(y_true, scores, title: "Held-out set")
|> VegaLite.config(axis: [grid: false])
```

To build your own screen, use `Verdict.grid/2`.

## What's in it

**Classification.** Confusion matrix, ROC, precision-recall, DET, calibration,
threshold curve, and scores split by true class. All of them take a list of
models to compare, and a score matrix for one-vs-rest multiclass with micro
and macro averages.

**Regression.** Predicted against actual, residuals, their distribution, and a
normal Q-Q.

**Clustering and decomposition.** Dendrogram, silhouette, elbow, PCA scree,
2-D projection, biplot, and a loadings heatmap.

**Model selection.** Learning curve, validation curve, grid search heatmap,
and per-fold scores, all reading what `Scholar.ModelSelection` returns.

**Everything else.** Coefficients, correlation matrix, sample weights across
the counting plots, and faceting when a one-vs-rest overlay gets crowded.

The [tour](notebooks/tour.livemd) walks through every plot with generated
data, including models that are deliberately wrong so you can see what that
looks like.

## Correctness

Values are checked against scikit-learn and SciPy where a reference exists,
curve points included: the one-vs-rest averages, the calibration binning, the
dendrogram's leaf order and bracket coordinates, and the Q-Q plot's positions
and fitted line. Sample weights are pinned to their definition, which is that
a weight of two matches the same data with that row duplicated.

## License

Copyright (c) 2026 Ricardo Carvalho Santos

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
