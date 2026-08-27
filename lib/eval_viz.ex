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
end
