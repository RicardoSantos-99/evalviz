defmodule Verdict do
  @moduledoc """
  Model evaluation plots for Nx tensors.

  Every function returns a `VegaLite` specification. Livebook renders one on
  sight, and you can keep customising it through the `VegaLite` API:

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> y_pred = Nx.tensor([0, 1, 1, 1])
      iex> Verdict.confusion_matrix(y_true, y_pred, num_classes: 2, title: "Validation set")
      ...> |> VegaLite.config(axis: [grid: false])
      ...> |> VegaLite.to_spec()
      ...> |> get_in(["config", "axis", "grid"])
      false

  Nothing here assumes a modelling library. Pass the tensors your model
  produced, whether they came from Scholar, Axon, or a hand-rolled `defn`.

  ## Where to start

  `report/3` draws the plots that go together for one model in a single call.
  `grid/2` lays out any set you build yourself. The rest are listed below,
  grouped by what you are looking at.

  ## Reading the options

  Plots that draw several things at once take a list of `{label, ...}` tuples
  instead of bare tensors, and colour them apart. The classification curves
  also take a `{num_samples, num_classes}` score matrix, which gives one
  one-vs-rest curve per class.

  ## Rendering

  Livebook needs `kino_vega_lite` in the dependency list. The `Kino.Render`
  implementation for `VegaLite` ships there rather than in `kino`, and without
  it Livebook prints the struct instead of drawing the chart.
  """

  alias Verdict.Biplot
  alias Verdict.Calibration
  alias Verdict.Coefficients
  alias Verdict.ConfusionMatrix
  alias Verdict.Correlation
  alias Verdict.Curves
  alias Verdict.Dendrogram
  alias Verdict.Distribution
  alias Verdict.Elbow
  alias Verdict.FoldScores
  alias Verdict.Grid
  alias Verdict.GridSearch
  alias Verdict.Internal
  alias Verdict.LearningCurve
  alias Verdict.Loadings
  alias Verdict.Projection
  alias Verdict.Regression
  alias Verdict.Report
  alias Verdict.Scree
  alias Verdict.ScoreDistribution
  alias Verdict.Threshold
  alias Verdict.Silhouette
  alias Verdict.ValidationCurve
  alias Verdict.Zoom

  @doc group: :classification
  @doc """
  Plots a confusion matrix as a heatmap with the value written in each cell.

  `y_true` and `y_pred` are rank-1 tensors of class indices.

  ## Options

  #{NimbleOptions.docs(ConfusionMatrix.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2])
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2])
      iex> plot = Verdict.confusion_matrix(y_true, y_pred, num_classes: 3)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> length()
      9

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2])
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2])
      iex> plot =
      ...>   Verdict.confusion_matrix(y_true, y_pred,
      ...>     num_classes: 3,
      ...>     class_names: ["cat", "dog", "bird"],
      ...>     normalize: :true_class
      ...>   )
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.filter(&(&1["actual"] == "bird"))
      ...> |> Enum.map(&(&1["value"]))
      ...> |> Enum.sum()
      1.0
  """
  def confusion_matrix(y_true, y_pred, opts \\ []) do
    ConfusionMatrix.plot(y_true, y_pred, opts)
  end

  @doc group: :classification
  @doc """
  Plots a ROC curve, with the area under it shown in the legend.

  `y_true` holds 0 and 1, `y_score` the score or probability the model gave the
  positive class. To compare models, pass a list of `{label, y_true, y_score}`
  instead and every curve is drawn on the same axes.

  For a multiclass model, pass `y_true` as class indices and `y_score` as a
  `{num_samples, num_classes}` matrix. That draws one one-vs-rest curve per
  class, and `:average` adds the micro or macro curve over them.

  Past a handful of classes the overlay stops being readable, and `facet: true`
  gives each curve its own panel instead. The panel is titled with the curve's
  name and summary, so the colour legend goes away with it.

  `:sample_weights` cannot be combined with `average: :micro`, which pools one
  row per (sample, class) pair rather than one per sample, leaving the weights
  describing something other than the rows they would apply to.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = Verdict.roc_curve(y_true, scores)
      iex> VegaLite.to_spec(plot)["title"]["subtitle"]
      "AUC = 0.75"

      iex> y_true = Nx.tensor([0, 1, 2, 0, 1, 2])
      iex> scores =
      ...>   Nx.tensor([
      ...>     [0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6],
      ...>     [0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]
      ...>   ])
      iex> plot = Verdict.roc_curve(y_true, scores, class_names: ["cat", "dog", "bird"])
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["series"]))
      ...> |> Enum.uniq()
      ["cat (AUC = 1.0)", "dog (AUC = 1.0)", "bird (AUC = 1.0)"]

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> logistic = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> naive_bayes = Nx.tensor([0.2, 0.3, 0.6, 0.9])
      iex> plot =
      ...>   Verdict.roc_curve([
      ...>     {"Logistic", y_true, logistic},
      ...>     {"Naive Bayes", y_true, naive_bayes}
      ...>   ])
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["series"]))
      ...> |> Enum.uniq()
      ["Logistic (AUC = 0.75)", "Naive Bayes (AUC = 1.0)"]
  """
  def roc_curve(series_or_y_true, y_score_or_opts \\ [], opts \\ [])

  def roc_curve(series, opts, _) when is_list(series) and is_list(opts) do
    Curves.plot(:roc, normalize_series(series), opts)
  end

  def roc_curve(y_true, y_score, opts) do
    Curves.plot(:roc, [{nil, y_true, y_score}], opts)
  end

  @doc group: :classification
  @doc """
  Plots a precision-recall curve, with average precision shown in the legend.

  Takes the same arguments as `roc_curve/3`, multiclass included. The dashed
  baseline sits at the share of positives in `y_true`, which is what a no-skill
  classifier scores. One-vs-rest classes each have their own share, so the
  baseline is only drawn when every curve on the plot agrees on it. Under
  `facet: true` a panel holds one class, so each draws the baseline that
  belongs to it rather than going without.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = Verdict.precision_recall_curve(y_true, scores)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["recall"]))
      [1.0, 1.0, 0.5, 0.5, 0.0]
  """
  def precision_recall_curve(series_or_y_true, y_score_or_opts \\ [], opts \\ [])

  def precision_recall_curve(series, opts, _) when is_list(series) and is_list(opts) do
    Curves.plot(:precision_recall, normalize_series(series), opts)
  end

  def precision_recall_curve(y_true, y_score, opts) do
    Curves.plot(:precision_recall, [{nil, y_true, y_score}], opts)
  end

  @doc group: :classification
  @doc """
  Plots a detection error tradeoff curve: false negative rate against false
  positive rate.

  Takes the same arguments as `roc_curve/3`, multiclass included.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = Verdict.det_curve(y_true, scores)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["fnr"]))
      [0.0, 0.0, 0.5, 0.5]
  """
  def det_curve(series_or_y_true, y_score_or_opts \\ [], opts \\ [])

  def det_curve(series, opts, _) when is_list(series) and is_list(opts) do
    Curves.plot(:det, normalize_series(series), opts)
  end

  def det_curve(y_true, y_score, opts) do
    Curves.plot(:det, [{nil, y_true, y_score}], opts)
  end

  @doc group: :clustering
  @doc """
  Plots a dendrogram of an agglomerative clustering.

  Accepts a `Scholar.Cluster.Hierarchical` model, or a `{clades, heights}` pair
  of tensors shaped `{n - 1, 2}` and `{n - 1}` for anything that produces the
  same merge structure.

  ## Options

  #{NimbleOptions.docs(Dendrogram.schema())}

  ## Examples

      iex> data = Nx.tensor([[1.0], [1.5], [5.0], [5.4], [9.0]])
      iex> model = Scholar.Cluster.Hierarchical.fit(data, linkage: :single)
      iex> plot = Verdict.dendrogram(model)
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["link"]))
      ...> |> Enum.uniq()
      ...> |> length()
      4

      iex> data = Nx.tensor([[1.0], [1.5], [5.0], [5.4], [9.0]])
      iex> model = Scholar.Cluster.Hierarchical.fit(data, linkage: :single)
      iex> plot =
      ...>   Verdict.dendrogram(model,
      ...>     labels: ["a", "b", "c", "d", "e"],
      ...>     color_threshold: 2.0
      ...>   )
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["cluster"]))
      ...> |> Enum.uniq()
      ...> |> Enum.sort()
      ["Above cut", "Cluster 1", "Cluster 2"]
  """
  def dendrogram(model_or_pair, opts \\ [])

  def dendrogram(%{clades: clades, dissimilarities: heights}, opts) do
    Dendrogram.plot(clades, heights, opts)
  end

  def dendrogram({clades, heights}, opts) do
    Dendrogram.plot(clades, heights, opts)
  end

  @doc group: :clustering
  @doc """
  Plots a silhouette diagram: one bar per point, grouped by cluster and sorted
  within it.

  `x` is the data that was clustered and `labels` the cluster each point landed
  in. Wide, evenly tall blocks mean a clean clustering; bars that taper early,
  or run negative, mark points that sit closer to a neighbouring cluster.

  ## Options

  #{NimbleOptions.docs(Silhouette.schema())}

  ## Examples

      iex> x = Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
      iex> labels = Nx.tensor([0, 0, 0, 1, 1])
      iex> plot = Verdict.silhouette(x, labels, num_clusters: 2)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> length()
      5
  """
  def silhouette(x, labels, opts \\ []) do
    Silhouette.plot(x, labels, opts)
  end

  @doc group: :clustering
  @doc """
  Plots a score against the number of clusters, to find where adding another
  stops paying.

  Accepts a list of fitted models exposing `:inertia` and `:clusters`, or the
  values of k and their scores directly, so it works with any score you can
  compute per k.

  The dashed rule marks the k furthest from the line joining the first and last
  points, measured after rescaling both axes to `0..1`. Without that rescaling
  the distance would only ever reflect whichever axis spans more.

  ## Options

  #{NimbleOptions.docs(Elbow.schema())}

  ## Examples

      iex> plot = Verdict.elbow([2, 3, 4, 5], [100.0, 40.0, 35.0, 32.0])
      iex> VegaLite.to_spec(plot)["title"]["subtitle"]
      "elbow at k = 3"

      iex> plot = Verdict.elbow([2, 3, 4, 5], [100.0, 40.0, 35.0, 32.0], mark_elbow: false)
      iex> VegaLite.to_spec(plot)["layer"] |> length()
      1
  """
  def elbow(models_or_ks, scores_or_opts \\ [], opts \\ [])

  def elbow([%{} | _] = models, opts, _) when is_list(opts) do
    ks = Enum.map(models, &Nx.axis_size(&1.clusters, 0))
    scores = Enum.map(models, &Nx.to_number(&1.inertia))

    Elbow.plot(ks, scores, opts)
  end

  def elbow(ks, scores, opts) do
    Elbow.plot(numbers(ks), numbers(scores), opts)
  end

  @doc group: :clustering
  @doc """
  Plots a two-dimensional embedding as a scatter, optionally coloured by label.

  Takes the `{num_samples, num_components}` tensor any dimensionality reduction
  produces, or a struct carrying it under `:embedding`. That covers
  `Scholar.Manifold.TSNE`, `MDS` and `TriMap`, and the decompositions once they
  have transformed the data.

  Pass `labels` to colour the points, or a list of `{title, labels}` to draw the
  same embedding several times side by side, which is how a clustering gets
  compared against the truth.

  Both axes share one range by default. The two directions of an embedding
  measure the same thing, so letting them scale apart would stretch the shape of
  the data, which is the whole reason for looking at it.

  ## Options

  #{NimbleOptions.docs(Projection.schema())}

  ## Examples

      iex> embedding = Nx.tensor([[0.0, 0.0], [1.0, 0.5], [5.0, 5.0], [5.5, 4.8]])
      iex> labels = Nx.tensor([0, 0, 1, 1])
      iex> plot = Verdict.projection(embedding, labels, label_names: ["left", "right"])
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["label"]))
      ["left", "left", "right", "right"]

      iex> embedding = Nx.tensor([[0.0, 0.0], [1.0, 0.5], [5.0, 5.0], [5.5, 4.8]])
      iex> plot =
      ...>   Verdict.projection(embedding, [
      ...>     {"True class", Nx.tensor([0, 0, 1, 1])},
      ...>     {"KMeans", Nx.tensor([1, 0, 1, 1])}
      ...>   ])
      iex> VegaLite.to_spec(plot)["hconcat"] |> length()
      2
  """
  def projection(embedding, labels_or_opts \\ [], opts \\ [])

  def projection(embedding, series, opts) when is_list(series) and is_list(opts) do
    if Keyword.keyword?(series) and opts == [] do
      Projection.plot(embedding, [], series)
    else
      Projection.plot(embedding, normalize_panels(series), opts)
    end
  end

  def projection(embedding, labels, opts) do
    Projection.plot(embedding, [{nil, labels}], opts)
  end

  @doc group: :clustering
  @doc """
  Plots a scree chart of the variance each principal component explains, with
  the running total overlaid.

  Accepts a `Scholar.Decomposition.PCA` model or a rank-1 tensor of explained
  variance ratios.

  ## Options

  #{NimbleOptions.docs(Scree.schema())}

  ## Examples

      iex> plot = Verdict.scree(Nx.tensor([0.6, 0.25, 0.1, 0.05]))
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&Float.round(&1["cumulative"], 4))
      [0.6, 0.85, 0.95, 1.0]
  """
  def scree(model_or_ratios, opts \\ [])

  def scree(%{explained_variance_ratio: ratios}, opts), do: Scree.plot(ratios, opts)
  def scree(ratios, opts), do: Scree.plot(ratios, opts)

  @doc group: :clustering
  @doc """
  Plots a biplot: the projected points, with an arrow per feature showing which
  way it pushes them.

  Accepts a decomposition model and the data it was fitted on, in which case the
  model's own `transform/2` produces the points, or a `{projection, loadings}`
  pair of tensors shaped `{num_samples, num_components}` and
  `{num_components, num_features}`.

  Loadings and scores have unrelated units, so the arrows are stretched to sit
  against the point cloud. Their directions and their lengths relative to each
  other carry the meaning; their absolute length does not.

  ## Options

  #{NimbleOptions.docs(Biplot.schema())}

  ## Examples

      iex> projection = Nx.tensor([[1.0, 0.2], [-1.0, -0.2], [0.5, -0.5], [-0.5, 0.5]])
      iex> loadings = Nx.tensor([[0.8, 0.6], [-0.6, 0.8]])
      iex> plot = Verdict.biplot({projection, loadings}, feature_names: ["height", "weight"])
      iex> VegaLite.to_spec(plot)["layer"] |> length()
      3
  """
  def biplot(model_or_pair, x_or_opts \\ [], opts \\ [])

  def biplot({projection, loadings}, opts, _) when is_list(opts) do
    Biplot.plot(projection, loadings, opts)
  end

  def biplot(%{components: loadings} = model, x, opts) do
    Biplot.plot(transform(model, x), loadings, opts)
  end

  @doc group: :clustering
  @doc """
  Plots the loadings of a decomposition as a heatmap: one row per component, one
  column per feature.

  The scree plot says how much variance each component carries. This says what
  each one is made of, which is what turns an unnamed component back into
  something you can talk about.

  Accepts a model exposing `:components` or the `{num_components, num_features}`
  tensor itself. The colour scale is centred on zero, since a loading's sign
  says which way the feature pushes.

  ## Options

  #{NimbleOptions.docs(Loadings.schema())}

  ## Examples

      iex> components = Nx.tensor([[0.7, 0.7], [-0.7, 0.7]])
      iex> plot = Verdict.loadings(components, feature_names: ["height", "weight"])
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["component"]))
      ...> |> Enum.uniq()
      ["Component 0", "Component 1"]
  """
  def loadings(model_or_components, opts \\ [])

  def loadings(%{components: components}, opts), do: Loadings.plot(components, opts)
  def loadings(components, opts), do: Loadings.plot(components, opts)

  @doc group: :features
  @doc """
  Plots a model's coefficients as a bar per feature, ordered by magnitude.

  Accepts a model exposing `:coefficients` or the tensor itself, shaped
  `{num_features}` for a regression or `{num_features, num_classes}` for a
  classifier, in which case each feature gets a bar per class.

  Coefficients are only comparable across features when the features were
  scaled to begin with. On raw features a large coefficient may say nothing
  more than that the column is measured in small units.

  ## Options

  #{NimbleOptions.docs(Coefficients.schema())}

  ## Examples

      iex> plot = Verdict.coefficients(Nx.tensor([0.2, -1.5, 0.9]))
      iex> VegaLite.to_spec(plot)["layer"] |> hd() |> get_in(["encoding", "y", "sort"])
      ["1", "2", "0"]

      iex> plot =
      ...>   Verdict.coefficients(Nx.tensor([0.2, -1.5, 0.9]),
      ...>     feature_names: ["age", "income", "height"],
      ...>     top: 2
      ...>   )
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["feature"]))
      ["income", "height"]
  """
  def coefficients(model_or_tensor, opts \\ [])

  def coefficients(%{coefficients: tensor}, opts), do: Coefficients.plot(tensor, opts)
  def coefficients(tensor, opts), do: Coefficients.plot(tensor, opts)

  @doc group: :features
  @doc """
  Plots the correlation between every pair of columns as a heatmap.

  Takes the `{num_samples, num_features}` data itself and computes the
  coefficients with `Scholar.Stats.correlation_matrix/1`.

  The colour scale is pinned to `-1..1` rather than fitted to the data, so a
  matrix of weak correlations does not colour like a matrix of strong ones.

  ## Options

  #{NimbleOptions.docs(Correlation.schema())}

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
      iex> plot = Verdict.correlation(x, feature_names: ["a", "b"])
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["label"]))
      ["1.0", "1.0", "1.0", "1.0"]
  """
  def correlation(x, opts \\ []), do: Correlation.plot(x, opts)

  @doc group: :classification
  @doc """
  Plots a calibration curve: how often the positive class actually occurs,
  against the probability the model gave it.

  A well calibrated model tracks the diagonal, meaning that among the cases it
  called 70% likely, roughly 70% turned out positive. Takes the same shapes as
  `roc_curve/3`, so a list of `{label, y_true, y_prob}` compares models.

  A `{num_samples, num_classes}` `y_prob` gives one one-vs-rest curve per class.
  `:average` accepts only `:micro` here: the per-class curves land on different
  bins, so a macro average would compare probabilities that were never
  comparable.

  Unlike the ranking curves, this one needs real probabilities rather than
  arbitrary scores.

  ## Options

  #{NimbleOptions.docs(Calibration.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1, 0, 1])
      iex> probabilities = Nx.tensor([0.1, 0.2, 0.8, 0.9, 0.3, 0.7], type: :f64)
      iex> plot = Verdict.calibration_curve(y_true, probabilities, bins: 2)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["observed"]))
      [0.0, 1.0]
  """
  def calibration_curve(series_or_y_true, y_prob_or_opts \\ [], opts \\ [])

  def calibration_curve(series, opts, _) when is_list(series) and is_list(opts) do
    Calibration.plot(normalize_series(series), opts)
  end

  def calibration_curve(y_true, y_prob, opts) do
    Calibration.plot([{nil, y_true, y_prob}], opts)
  end

  @doc group: :regression
  @doc """
  Plots predicted against actual values, with the diagonal a perfect model
  would sit on.

  Both axes share one range, so distance from the diagonal reads directly as
  error rather than being distorted by two different scales.

  To compare models, pass a list of `{label, y_true, y_pred}` instead. Every
  model lands on the same axes and the same diagonal, which is what makes the
  distances comparable.

  ## Options

  #{NimbleOptions.docs(Regression.schema())}

  ## Examples

      iex> y_true = Nx.tensor([1.0, 2.0, 3.0])
      iex> y_pred = Nx.tensor([1.1, 1.9, 3.2])
      iex> plot = Verdict.predicted_vs_actual(y_true, y_pred)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["actual"]))
      [1.0, 2.0, 3.0]

      iex> y_true = Nx.tensor([1.0, 2.0, 3.0])
      iex> plot =
      ...>   Verdict.predicted_vs_actual([
      ...>     {"Linear", y_true, Nx.tensor([1.1, 1.9, 3.2])},
      ...>     {"Ridge", y_true, Nx.tensor([1.3, 2.2, 2.8])}
      ...>   ])
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["series"]))
      ...> |> Enum.uniq()
      ["Linear", "Ridge"]
  """
  def predicted_vs_actual(series_or_y_true, y_pred_or_opts \\ [], opts \\ [])

  def predicted_vs_actual(series, opts, _) when is_list(series) and is_list(opts) do
    Regression.predicted_vs_actual(normalize_series(series), opts)
  end

  def predicted_vs_actual(y_true, y_pred, opts) do
    Regression.predicted_vs_actual([{nil, y_true, y_pred}], opts)
  end

  @doc group: :regression
  @doc """
  Plots residuals against predicted values.

  Points scattered evenly around zero mean the model has no systematic bias
  left; a curve or a widening fan means it has.

  Takes the same shapes as `predicted_vs_actual/3`, so a list of
  `{label, y_true, y_pred}` puts several models on one zero line.

  ## Options

  #{NimbleOptions.docs(Regression.schema())}

  ## Examples

      iex> y_true = Nx.tensor([1.0, 2.0, 3.0])
      iex> y_pred = Nx.tensor([1.5, 2.0, 2.0])
      iex> plot = Verdict.residuals(y_true, y_pred)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["residual"]))
      [0.5, 0.0, -1.0]
  """
  def residuals(series_or_y_true, y_pred_or_opts \\ [], opts \\ [])

  def residuals(series, opts, _) when is_list(series) and is_list(opts) do
    Regression.residuals(normalize_series(series), opts)
  end

  def residuals(y_true, y_pred, opts) do
    Regression.residuals([{nil, y_true, y_pred}], opts)
  end

  @doc group: :regression
  @doc """
  Plots the distribution of the residuals as a histogram.

  `residuals/3` shows the residuals against the prediction, which is where a
  systematic pattern shows up. This shows their shape: whether they centre on
  zero, and whether the tail on one side is longer than the other.

  ## Options

  #{NimbleOptions.docs(Distribution.histogram_schema())}

  ## Examples

      iex> y_true = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      iex> y_pred = Nx.tensor([1.5, 2.0, 2.0, 4.5])
      iex> plot = Verdict.residual_distribution(y_true, y_pred, bins: 2)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["count"])) |> Enum.sum()
      4
  """
  def residual_distribution(y_true, y_pred, opts \\ []) do
    Internal.assert_paired!(y_true, y_pred, "y_true", "y_pred")
    Distribution.histogram(Nx.subtract(y_pred, y_true), opts)
  end

  @doc group: :regression
  @doc """
  Plots a normal quantile-quantile plot: the sorted values against the
  quantiles a normal sample of the same size would be expected to land on.

  Points on the line mean the sample is normal. A curve at one end means that
  tail is heavier or lighter than a normal's, and an S means both are.

  Pass residuals to check the assumption a linear model makes about them. The
  points sit where `scipy.stats.probplot` puts them, and the dashed line is the
  least-squares fit through them.

  ## Options

  #{NimbleOptions.docs(Distribution.qq_schema())}

  ## Examples

      iex> values = Nx.tensor([0.3, -1.2, 1.5, -0.4, 0.1], type: :f64)
      iex> plot = Verdict.qq_plot(values)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["sample"]))
      [-1.2, -0.4, 0.1, 0.3, 1.5]
  """
  def qq_plot(values, opts \\ []), do: Distribution.qq(values, opts)

  @doc group: :classification
  @doc """
  Plots precision, recall and F1 against the decision threshold.

  Every other classification plot here answers how good the ranking is. This one
  answers the question that follows: given that ranking, where do you cut? The
  dashed rule marks the threshold with the highest F1.

  A `{num_samples, num_classes}` `y_score` gives one one-vs-rest set of curves
  per class, with colour carrying the class and the dash pattern the metric. The
  best-F1 rule is left out there, since each class peaks somewhere different.

  `facet: true` gives each class its own panel, which also hands colour back to
  the metric so the dash pattern is no longer needed to tell the two apart.

  Unlike the other plots here this one takes no list of models. Colour and dash
  are already spent on class and metric, and a third model dimension has no
  channel left to read from. Comparing models is what `roc_curve/3` and
  `precision_recall_curve/3` are for; this one tunes the cut-off of one model.

  ## Options

  #{NimbleOptions.docs(Threshold.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = Verdict.threshold_curve(y_true, scores)
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["metric"]))
      ...> |> Enum.uniq()
      ["Precision", "Recall", "F1"]

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = Verdict.threshold_curve(y_true, scores, metrics: [:precision, :recall])
      iex> VegaLite.to_spec(plot)["layer"] |> length()
      1
  """
  def threshold_curve(y_true, y_score, opts \\ []) do
    Threshold.plot(y_true, y_score, opts)
  end

  @doc group: :classification
  @doc """
  Plots the scores the model gave, as one histogram per true class.

  The threshold curve says where to cut. This says why the cut works or does
  not: two humps that barely touch mean the model separates the classes, and
  two that sit on top of each other mean no threshold will save it.

  Each class's bars sum to one by default, which is what keeps a rare class
  visible next to a common one. Pass `normalize: :none` for raw counts.

  `y_true` may name any number of classes; `y_score` is the single score being
  cut on.

  ## Options

  #{NimbleOptions.docs(ScoreDistribution.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.2, 0.8, 0.9])
      iex> plot = Verdict.score_distribution(y_true, scores, bins: 2)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["class"]))
      ["0", "1"]

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.2, 0.8, 0.9])
      iex> plot = Verdict.score_distribution(y_true, scores, bins: 2, threshold: 0.5)
      iex> VegaLite.to_spec(plot)["layer"] |> length()
      2
  """
  def score_distribution(y_true, y_score, opts \\ []) do
    ScoreDistribution.plot(y_true, y_score, opts)
  end

  @doc group: :model_selection
  @doc """
  Plots training and validation score against how much data the model was
  trained on.

  Curves that meet at a low score mean the model is too simple for the problem;
  a gap that stays wide as data grows means it is overfitting.

  This does not train anything. Pass the scores you already measured, the same
  split `scikit-learn` draws between computing a learning curve and displaying
  one, so it works whatever you trained with. Scores may be one number per
  training size, or one per fold, in which case the mean is drawn with a band
  one standard deviation wide.

  To compare models, pass a list of `{label, train_scores, validation_scores}`
  in place of the two score arguments. Colour then carries the model and the
  dash pattern carries training against validation, so both readings survive.
  Consider `spread: false` there, since every model shades twice.

  ## Options

  #{NimbleOptions.docs(LearningCurve.schema())}

  ## Examples

      iex> sizes = [10, 20, 40]
      iex> train = [[0.99, 0.98], [0.95, 0.94], [0.92, 0.91]]
      iex> validation = [[0.70, 0.68], [0.80, 0.79], [0.88, 0.87]]
      iex> plot = Verdict.learning_curve(sizes, train, validation)
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["series"]))
      ...> |> Enum.uniq()
      ["Training", "Validation"]

      iex> sizes = [10, 20, 40]
      iex> plot =
      ...>   Verdict.learning_curve(sizes, [
      ...>     {"Linear", [0.9, 0.9, 0.9], [0.7, 0.8, 0.85]},
      ...>     {"Forest", [1.0, 1.0, 1.0], [0.6, 0.75, 0.88]}
      ...>   ])
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["model"]))
      ...> |> Enum.uniq()
      ["Linear", "Forest"]
  """
  def learning_curve(train_sizes, series_or_train, validation_or_opts \\ [], opts \\ [])

  def learning_curve(train_sizes, [{_, _, _} | _] = series, opts, _) when is_list(opts) do
    LearningCurve.plot(train_sizes, normalize_series(series), opts)
  end

  def learning_curve(train_sizes, train_scores, validation_scores, opts) do
    LearningCurve.plot(train_sizes, [{nil, train_scores, validation_scores}], opts)
  end

  @doc group: :model_selection
  @doc """
  Plots training and validation score against one hyperparameter.

  The learning curve asks whether more data would help. This asks the question
  you answer with the data you have: how far to turn one knob. The two curves
  meeting low means the setting is too restrictive, and a gap that widens as
  the parameter grows means it is fitting noise.

  Like `learning_curve/4` this trains nothing: pass the scores you measured.
  Parameter values may be numbers, in which case the axis is quantitative and
  can be logarithmic, or anything else, in which case it is nominal and keeps
  the order you gave.

  ## Options

  #{NimbleOptions.docs(ValidationCurve.schema())}

  ## Examples

      iex> alphas = [0.01, 0.1, 1.0, 10.0]
      iex> train = [0.99, 0.97, 0.90, 0.72]
      iex> validation = [0.80, 0.88, 0.86, 0.70]
      iex> plot = Verdict.validation_curve(alphas, train, validation, param_name: "alpha")
      iex> VegaLite.to_spec(plot)["title"]["subtitle"]
      "best 0.88 at alpha = 0.1"

      iex> alphas = [0.01, 0.1, 1.0, 10.0]
      iex> train = [0.99, 0.97, 0.90, 0.72]
      iex> validation = [0.80, 0.88, 0.86, 0.70]
      iex> plot = Verdict.validation_curve(alphas, train, validation, scale: :log)
      iex> VegaLite.to_spec(plot)["layer"]
      ...> |> List.last()
      ...> |> get_in(["encoding", "x", "scale", "type"])
      "log"
  """
  def validation_curve(param_values, train_scores, validation_scores, opts \\ []) do
    ValidationCurve.plot(param_values, train_scores, validation_scores, opts)
  end

  @doc group: :model_selection
  @doc """
  Plots a grid search as a heatmap, one cell per pair of hyperparameter values.

  Takes what `Scholar.ModelSelection.grid_search/5` returns: a list of
  `%{hyperparameters: keyword, score: tensor}`. Without `:x` and `:y` it uses
  the two hyperparameters that vary. When more than two vary, name the two to
  plot and the rest are collapsed by keeping the best score for each cell,
  which the subtitle says out loud.

  Darker always means better, so `best: :min` reverses the ramp rather than
  leaving you to invert it by eye.

  ## Options

  #{NimbleOptions.docs(GridSearch.schema())}

  ## Examples

      iex> results = [
      ...>   %{hyperparameters: [alpha: 0.0, iterations: 10], score: Nx.tensor([0.7])},
      ...>   %{hyperparameters: [alpha: 0.0, iterations: 50], score: Nx.tensor([0.8])},
      ...>   %{hyperparameters: [alpha: 1.0, iterations: 10], score: Nx.tensor([0.6])},
      ...>   %{hyperparameters: [alpha: 1.0, iterations: 50], score: Nx.tensor([0.9])}
      ...> ]
      iex> plot = Verdict.grid_search(results, x: :iterations, y: :alpha)
      iex> VegaLite.to_spec(plot)["title"]["subtitle"]
      "best 0.9 at iterations 50, alpha 1.0"
  """
  def grid_search(results, opts \\ []), do: GridSearch.plot(results, opts)

  @doc group: :model_selection
  @doc """
  Plots the score each fold got, with the mean they scatter around.

  Takes what `Scholar.ModelSelection.cross_validate/4` returns, a
  `{num_metrics, num_folds}` tensor, and draws a panel per metric. A single
  metric may be passed as a rank-1 tensor.

  A mean on its own says nothing about how much it moved between folds. This is
  that spread, stated in each panel's subtitle as well as drawn. The folds are
  not joined by a line: they are interchangeable, and joining them would show a
  trend across an order that carries no meaning.

  ## Options

  #{NimbleOptions.docs(FoldScores.schema())}

  ## Examples

      iex> scores = Nx.tensor([[0.80, 0.86, 0.84], [1.20, 1.10, 1.15]], type: :f64)
      iex> plot = Verdict.fold_scores(scores, metric_names: ["Accuracy", "Loss"])
      iex> VegaLite.to_spec(plot)["vconcat"] |> Enum.map(&(&1["title"]["text"]))
      ["Accuracy", "Loss"]

      iex> scores = Nx.tensor([0.80, 0.86, 0.84], type: :f64)
      iex> plot = Verdict.fold_scores(scores)
      iex> VegaLite.to_spec(plot)["title"]["subtitle"]
      "mean 0.8333 ± 0.0249"
  """
  def fold_scores(scores, opts \\ []), do: FoldScores.plot(scores, opts)

  @doc group: :screens
  @doc """
  Lays several plots out in a wrapping grid.

  Takes anything that returns a `VegaLite` specification, this library's
  functions included, so a screen of related plots is one value you can pass
  around and render.

  Vega-Lite allows `$schema`, `background`, `padding`, `autosize`, `config` and
  `usermeta` only on the outermost specification, so apply `VegaLite.config/2`
  to the grid rather than to the plots going into it.

  ## Options

  #{NimbleOptions.docs(Grid.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot =
      ...>   Verdict.grid([
      ...>     Verdict.roc_curve(y_true, scores),
      ...>     Verdict.precision_recall_curve(y_true, scores)
      ...>   ])
      iex> VegaLite.to_spec(plot)["columns"]
      2

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = Verdict.grid([Verdict.roc_curve(y_true, scores)], columns: 1)
      iex> VegaLite.to_spec(plot)["concat"] |> length()
      1
  """
  def grid(views, opts \\ []), do: Grid.plot(views, opts)

  @doc group: :screens
  @doc """
  Lets the reader drag and scroll a plot's axes.

  A scatter of thousands of points is a cloud until you can get inside it, and
  the interesting part of a ROC curve is a corner too small to read at full
  scale. This binds a drag and the scroll wheel to the axes, so the reader can
  go there without the plot being rebuilt at a different range.

  Takes any specification this library returns and gives back the same one,
  zoomable. A plot drawn from several views, `report/3` and `grid/2` included,
  has each of its views made zoomable in turn. Views that put their axes on
  categories, such as a confusion matrix, are left alone: Vega-Lite binds a
  drag to a scale only where that scale is continuous.

  ## Options

  #{NimbleOptions.docs(Zoom.schema())}

  ## Examples

      iex> embedding = Nx.tensor([[0.0, 0.0], [1.0, 0.5], [5.0, 5.0], [5.5, 4.8]])
      iex> plot = Verdict.projection(embedding) |> Verdict.zoomable()
      iex> VegaLite.to_spec(plot)["params"] |> Enum.map(&(&1["bind"]))
      ["scales"]

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = Verdict.roc_curve(y_true, scores) |> Verdict.zoomable(encodings: [:x])
      iex> [layer] = VegaLite.to_spec(plot)["layer"] |> Enum.filter(&(&1["params"]))
      iex> layer["params"] |> Enum.map(&(&1["select"]["encodings"]))
      [["x"]]
  """
  def zoomable(plot, opts \\ []), do: Zoom.apply(plot, opts)

  @doc group: :screens
  @doc """
  Draws a screen of the plots that go together for one model.

  For a binary classifier that is the confusion matrix, the ROC and
  precision-recall curves, and the scores split by true class. The confusion
  matrix and the score distribution read the same `:threshold`, so the report
  shows what one cut-off actually does.

  Pass `kind: :regression` and the second argument is read as predictions
  rather than scores, giving predicted against actual, the residuals, their
  distribution, and a normal Q-Q.

  This is a convenience over `grid/2`: build the plots yourself and lay them
  out with that when you want a different set.

  ## Options

  #{NimbleOptions.docs(Report.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1, 0, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
      iex> plot = Verdict.report(y_true, scores)
      iex> VegaLite.to_spec(plot)["concat"] |> length()
      4

      iex> y_true = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      iex> y_pred = Nx.tensor([1.2, 1.9, 3.3, 3.8])
      iex> plot = Verdict.report(y_true, y_pred, kind: :regression, columns: 4)
      iex> VegaLite.to_spec(plot)["columns"]
      4
  """
  def report(y_true, y_predicted, opts \\ []), do: Report.plot(y_true, y_predicted, opts)

  defp normalize_series(series) do
    Enum.map(series, fn
      {label, y_true, y_score} -> {to_string(label), y_true, y_score}
      other -> raise ArgumentError, "expected {label, y_true, y_score}, got: #{inspect(other)}"
    end)
  end

  defp normalize_panels(panels) do
    Enum.map(panels, fn
      {title, labels} -> {to_string(title), labels}
      other -> raise ArgumentError, "expected {title, labels}, got: #{inspect(other)}"
    end)
  end

  defp numbers(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp numbers(list) when is_list(list), do: list

  defp transform(%module{} = model, x) do
    if Code.ensure_loaded?(module) and function_exported?(module, :transform, 2) do
      module.transform(model, x)
    else
      raise ArgumentError,
            "#{inspect(module)} has no transform/2, so the projection cannot be derived " <>
              "from the model. Pass a {projection, loadings} pair instead."
    end
  end
end
