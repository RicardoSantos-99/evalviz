defmodule EvalViz do
  @moduledoc """
  Model evaluation plots for Nx tensors.

  Every function returns a `VegaLite` specification, which Livebook renders on
  its own and which you can keep customising through the `VegaLite` API:

      EvalViz.confusion_matrix(y_true, y_pred, num_classes: 3, title: "Validation set")
      |> VegaLite.config(axis: [grid: false])

  Nothing here assumes a particular modelling library. Pass the tensors your
  model produced and the plot follows, whether they came from Scholar, Axon or
  anywhere else.
  """

  alias EvalViz.Calibration
  alias EvalViz.ConfusionMatrix
  alias EvalViz.Curves
  alias EvalViz.Dendrogram
  alias EvalViz.Regression
  alias EvalViz.Scree
  alias EvalViz.Silhouette

  @doc """
  Plots a confusion matrix as a heatmap with the value written in each cell.

  `y_true` and `y_pred` are rank-1 tensors of class indices.

  ## Options

  #{NimbleOptions.docs(ConfusionMatrix.schema())}

  ## Examples

      y_true = Nx.tensor([0, 0, 1, 1, 2, 2])
      y_pred = Nx.tensor([0, 1, 0, 2, 2, 2])
      EvalViz.confusion_matrix(y_true, y_pred, num_classes: 3)

      EvalViz.confusion_matrix(y_true, y_pred,
        num_classes: 3,
        class_names: ["cat", "dog", "bird"],
        normalize: :true_class
      )
  """
  def confusion_matrix(y_true, y_pred, opts \\ []) do
    ConfusionMatrix.plot(y_true, y_pred, opts)
  end

  @doc """
  Plots a ROC curve, with the area under it shown in the legend.

  `y_true` holds 0 and 1, `y_score` the score or probability the model gave the
  positive class. To compare models, pass a list of `{label, y_true, y_score}`
  instead and every curve is drawn on the same axes.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}

  ## Examples

      EvalViz.roc_curve(y_true, scores)

      EvalViz.roc_curve([
        {"Logistic", y_true, logistic_scores},
        {"Naive Bayes", y_true, nb_scores}
      ])
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

  Takes the same arguments as `roc_curve/3`. The dashed baseline sits at the
  share of positives in `y_true`, which is what a no-skill classifier scores.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}
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

  Takes the same arguments as `roc_curve/3`.

  ## Options

  #{NimbleOptions.docs(Curves.schema())}
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

      model = Scholar.Cluster.Hierarchical.fit(data)
      EvalViz.dendrogram(model)

      EvalViz.dendrogram(model,
        labels: ["a", "b", "c", "d", "e"],
        color_threshold: 2.0
      )
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

      EvalViz.silhouette(x, labels, num_clusters: 3)
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

      pca = Scholar.Decomposition.PCA.fit(x, num_components: 4)
      EvalViz.scree(pca)
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

  Unlike the ranking curves, this one needs real probabilities rather than
  arbitrary scores.

  ## Options

  #{NimbleOptions.docs(Calibration.schema())}

  ## Examples

      EvalViz.calibration_curve(y_true, probabilities, bins: 5)
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

      EvalViz.predicted_vs_actual(y_true, y_pred)
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

      EvalViz.residuals(y_true, y_pred)
  """
  def residuals(y_true, y_pred, opts \\ []) do
    Regression.residuals(y_true, y_pred, opts)
  end

  defp normalize_series(series) do
    Enum.map(series, fn
      {label, y_true, y_score} -> {to_string(label), y_true, y_score}
      other -> raise ArgumentError, "expected {label, y_true, y_score}, got: #{inspect(other)}"
    end)
  end
end
