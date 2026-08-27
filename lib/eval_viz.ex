defmodule EvalViz do
  @moduledoc """
  Model evaluation plots for Nx tensors.

  Every function returns a `VegaLite` specification, which Livebook renders on
  its own and which you can keep customising through the `VegaLite` API:

      EvalViz.confusion_matrix(y_true, y_pred, num_classes: 3)
      |> VegaLite.title("Validation set")

  Nothing here assumes a particular modelling library. Pass the tensors your
  model produced and the plot follows, whether they came from Scholar, Axon or
  anywhere else.
  """

  alias EvalViz.ConfusionMatrix
  alias EvalViz.Curves
  alias EvalViz.Dendrogram

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

  defp normalize_series(series) do
    Enum.map(series, fn
      {label, y_true, y_score} -> {to_string(label), y_true, y_score}
      other -> raise ArgumentError, "expected {label, y_true, y_score}, got: #{inspect(other)}"
    end)
  end
end
