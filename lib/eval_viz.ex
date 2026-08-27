defmodule EvalViz do
  @moduledoc """
  Model evaluation plots for Nx tensors.

  Every function returns a `VegaLite` specification, which Livebook renders on
  its own and which you can keep customising through the `VegaLite` API:

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> y_pred = Nx.tensor([0, 1, 1, 1])
      iex> EvalViz.confusion_matrix(y_true, y_pred, num_classes: 2, title: "Validation set")
      ...> |> VegaLite.config(axis: [grid: false])
      ...> |> VegaLite.to_spec()
      ...> |> get_in(["config", "axis", "grid"])
      false

  Rendering them in Livebook needs `kino_vega_lite`: the `Kino.Render`
  implementation for `VegaLite` lives there rather than in `kino`.

  Nothing here assumes a particular modelling library. Pass the tensors your
  model produced and the plot follows, whether they came from Scholar, Axon or
  anywhere else.
  """

  alias EvalViz.Calibration
  alias EvalViz.ConfusionMatrix
  alias EvalViz.Curves
  alias EvalViz.Dendrogram
  alias EvalViz.LearningCurve
  alias EvalViz.Regression
  alias EvalViz.Scree
  alias EvalViz.Threshold
  alias EvalViz.Silhouette

  @doc """
  Plots a confusion matrix as a heatmap with the value written in each cell.

  `y_true` and `y_pred` are rank-1 tensors of class indices.

  ## Options

  #{NimbleOptions.docs(ConfusionMatrix.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2])
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2])
      iex> plot = EvalViz.confusion_matrix(y_true, y_pred, num_classes: 3)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> length()
      9

      iex> y_true = Nx.tensor([0, 0, 1, 1, 2, 2])
      iex> y_pred = Nx.tensor([0, 1, 0, 2, 2, 2])
      iex> plot =
      ...>   EvalViz.confusion_matrix(y_true, y_pred,
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

  @doc """
  Plots a ROC curve, with the area under it shown in the legend.

  `y_true` holds 0 and 1, `y_score` the score or probability the model gave the
  positive class. To compare models, pass a list of `{label, y_true, y_score}`
  instead and every curve is drawn on the same axes.

  For a multiclass model, pass `y_true` as class indices and `y_score` as a
  `{num_samples, num_classes}` matrix. That draws one one-vs-rest curve per
  class, and `:average` adds the micro or macro curve over them.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = EvalViz.roc_curve(y_true, scores)
      iex> VegaLite.to_spec(plot)["title"]["subtitle"]
      "AUC = 0.75"

      iex> y_true = Nx.tensor([0, 1, 2, 0, 1, 2])
      iex> scores =
      ...>   Nx.tensor([
      ...>     [0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6],
      ...>     [0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]
      ...>   ])
      iex> plot = EvalViz.roc_curve(y_true, scores, class_names: ["cat", "dog", "bird"])
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["series"]))
      ...> |> Enum.uniq()
      ["cat (AUC = 1.0)", "dog (AUC = 1.0)", "bird (AUC = 1.0)"]

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> logistic = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> naive_bayes = Nx.tensor([0.2, 0.3, 0.6, 0.9])
      iex> plot =
      ...>   EvalViz.roc_curve([
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

  @doc """
  Plots a precision-recall curve, with average precision shown in the legend.

  Takes the same arguments as `roc_curve/3`, multiclass included. The dashed
  baseline sits at the share of positives in `y_true`, which is what a no-skill
  classifier scores. One-vs-rest classes each have their own share, so the
  baseline is only drawn when every curve on the plot agrees on it.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = EvalViz.precision_recall_curve(y_true, scores)
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

  @doc """
  Plots a detection error tradeoff curve: false negative rate against false
  positive rate.

  Takes the same arguments as `roc_curve/3`, multiclass included.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = EvalViz.det_curve(y_true, scores)
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
      iex> plot = EvalViz.dendrogram(model)
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["link"]))
      ...> |> Enum.uniq()
      ...> |> length()
      4

      iex> data = Nx.tensor([[1.0], [1.5], [5.0], [5.4], [9.0]])
      iex> model = Scholar.Cluster.Hierarchical.fit(data, linkage: :single)
      iex> plot =
      ...>   EvalViz.dendrogram(model,
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
      iex> plot = EvalViz.silhouette(x, labels, num_clusters: 2)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> length()
      5
  """
  def silhouette(x, labels, opts \\ []) do
    Silhouette.plot(x, labels, opts)
  end

  @doc """
  Plots a scree chart of the variance each principal component explains, with
  the running total overlaid.

  Accepts a `Scholar.Decomposition.PCA` model or a rank-1 tensor of explained
  variance ratios.

  ## Options

  #{NimbleOptions.docs(Scree.schema())}

  ## Examples

      iex> plot = EvalViz.scree(Nx.tensor([0.6, 0.25, 0.1, 0.05]))
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&Float.round(&1["cumulative"], 4))
      [0.6, 0.85, 0.95, 1.0]
  """
  def scree(model_or_ratios, opts \\ [])

  def scree(%{explained_variance_ratio: ratios}, opts), do: Scree.plot(ratios, opts)
  def scree(ratios, opts), do: Scree.plot(ratios, opts)

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
      iex> plot = EvalViz.calibration_curve(y_true, probabilities, bins: 2)
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

  @doc """
  Plots predicted against actual values, with the diagonal a perfect model
  would sit on.

  Both axes share one range, so distance from the diagonal reads directly as
  error rather than being distorted by two different scales.

  ## Options

  #{NimbleOptions.docs(Regression.schema())}

  ## Examples

      iex> y_true = Nx.tensor([1.0, 2.0, 3.0])
      iex> y_pred = Nx.tensor([1.1, 1.9, 3.2])
      iex> plot = EvalViz.predicted_vs_actual(y_true, y_pred)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["actual"]))
      [1.0, 2.0, 3.0]
  """
  def predicted_vs_actual(y_true, y_pred, opts \\ []) do
    Regression.predicted_vs_actual(y_true, y_pred, opts)
  end

  @doc """
  Plots residuals against predicted values.

  Points scattered evenly around zero mean the model has no systematic bias
  left; a curve or a widening fan means it has.

  ## Options

  #{NimbleOptions.docs(Regression.schema())}

  ## Examples

      iex> y_true = Nx.tensor([1.0, 2.0, 3.0])
      iex> y_pred = Nx.tensor([1.5, 2.0, 2.0])
      iex> plot = EvalViz.residuals(y_true, y_pred)
      iex> VegaLite.to_spec(plot)["data"]["values"] |> Enum.map(&(&1["residual"]))
      [0.5, 0.0, -1.0]
  """
  def residuals(y_true, y_pred, opts \\ []) do
    Regression.residuals(y_true, y_pred, opts)
  end

  @doc """
  Plots precision, recall and F1 against the decision threshold.

  Every other classification plot here answers how good the ranking is. This one
  answers the question that follows: given that ranking, where do you cut? The
  dashed rule marks the threshold with the highest F1.

  A `{num_samples, num_classes}` `y_score` gives one one-vs-rest set of curves
  per class, with colour carrying the class and the dash pattern the metric. The
  best-F1 rule is left out there, since each class peaks somewhere different.

  ## Options

  #{NimbleOptions.docs(Threshold.schema())}

  ## Examples

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = EvalViz.threshold_curve(y_true, scores)
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["metric"]))
      ...> |> Enum.uniq()
      ["Precision", "Recall", "F1"]

      iex> y_true = Nx.tensor([0, 0, 1, 1])
      iex> scores = Nx.tensor([0.1, 0.4, 0.35, 0.8])
      iex> plot = EvalViz.threshold_curve(y_true, scores, metrics: [:precision, :recall])
      iex> VegaLite.to_spec(plot)["layer"] |> length()
      1
  """
  def threshold_curve(y_true, y_score, opts \\ []) do
    Threshold.plot(y_true, y_score, opts)
  end

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

  ## Options

  #{NimbleOptions.docs(LearningCurve.schema())}

  ## Examples

      iex> sizes = [10, 20, 40]
      iex> train = [[0.99, 0.98], [0.95, 0.94], [0.92, 0.91]]
      iex> validation = [[0.70, 0.68], [0.80, 0.79], [0.88, 0.87]]
      iex> plot = EvalViz.learning_curve(sizes, train, validation)
      iex> VegaLite.to_spec(plot)["data"]["values"]
      ...> |> Enum.map(&(&1["series"]))
      ...> |> Enum.uniq()
      ["Training", "Validation"]
  """
  def learning_curve(train_sizes, train_scores, validation_scores, opts \\ []) do
    LearningCurve.plot(train_sizes, train_scores, validation_scores, opts)
  end

  defp normalize_series(series) do
    Enum.map(series, fn
      {label, y_true, y_score} -> {to_string(label), y_true, y_score}
      other -> raise ArgumentError, "expected {label, y_true, y_score}, got: #{inspect(other)}"
    end)
  end
end
