defmodule Verdict.Report do
  @moduledoc false

  alias Verdict.Grid

  @classification [:confusion_matrix, :roc, :precision_recall, :score_distribution]
  @regression [:predicted_vs_actual, :residuals, :residual_distribution, :qq]

  @opts_schema NimbleOptions.new!(
                 kind: [
                   type: {:in, [:classification, :regression]},
                   default: :classification,
                   doc: """
                   `:regression` reads the second argument as predictions rather
                   than scores, and draws the regression plots instead.
                   """
                 ],
                 plots: [
                   type: {:list, :atom},
                   doc: """
                   Which plots to draw, in order. Defaults to
                   `#{inspect(@classification)}` for classification and
                   `#{inspect(@regression)}` for regression.
                   """
                 ],
                 threshold: [
                   type: :float,
                   default: 0.5,
                   doc: """
                   The cut-off the confusion matrix and the score distribution
                   both read, so the report shows what one threshold does.
                   """
                 ],
                 class_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "Names for the two classes."
                 ],
                 columns: [type: :pos_integer, default: 2],
                 title: [type: :string, doc: "Title above the whole report."]
               )

  def schema, do: @opts_schema

  def plot(y_true, y_predicted, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    kind = opts[:kind]

    assert_supported!(y_predicted, kind)

    plots = opts[:plots] || default_plots(kind)

    plots
    |> Enum.map(&panel(&1, kind, y_true, y_predicted, opts))
    |> Grid.plot(grid_opts(opts))
  end

  defp default_plots(:classification), do: @classification
  defp default_plots(:regression), do: @regression

  # Every plot in the report reads a single column of scores. The one-vs-rest
  # plots take a matrix on their own, so a multiclass model is served by
  # calling them directly rather than by a second layout here.
  defp assert_supported!(y_predicted, :classification) do
    if Nx.rank(y_predicted) != 1 do
      raise ArgumentError,
            "expected a rank-1 y_score, got rank #{Nx.rank(y_predicted)}. " <>
              "For a multiclass model call roc_curve/3 and the others directly, " <>
              "which take a column per class."
    end

    :ok
  end

  defp assert_supported!(_y_predicted, :regression), do: :ok

  defp panel(:confusion_matrix, :classification, y_true, y_score, opts) do
    y_pred = Nx.select(Nx.greater_equal(y_score, opts[:threshold]), 1, 0)

    Verdict.confusion_matrix(
      y_true,
      y_pred,
      compact(
        num_classes: 2,
        class_names: opts[:class_names],
        title: "Confusion matrix at #{opts[:threshold]}",
        width: 260,
        height: 260
      )
    )
  end

  defp panel(:roc, :classification, y_true, y_score, _opts) do
    Verdict.roc_curve(y_true, y_score, title: "ROC", width: 300, height: 260)
  end

  defp panel(:precision_recall, :classification, y_true, y_score, _opts) do
    Verdict.precision_recall_curve(y_true, y_score,
      title: "Precision-recall",
      width: 300,
      height: 260
    )
  end

  defp panel(:det, :classification, y_true, y_score, _opts) do
    Verdict.det_curve(y_true, y_score, title: "DET", width: 300, height: 260)
  end

  defp panel(:threshold, :classification, y_true, y_score, _opts) do
    Verdict.threshold_curve(y_true, y_score, title: "Threshold", width: 320, height: 260)
  end

  defp panel(:calibration, :classification, y_true, y_score, _opts) do
    Verdict.calibration_curve(y_true, y_score, title: "Calibration", width: 300, height: 260)
  end

  defp panel(:score_distribution, :classification, y_true, y_score, opts) do
    Verdict.score_distribution(
      y_true,
      y_score,
      compact(
        class_names: opts[:class_names],
        threshold: opts[:threshold],
        title: "Scores by true class",
        width: 320,
        height: 240
      )
    )
  end

  defp panel(:predicted_vs_actual, :regression, y_true, y_pred, _opts) do
    Verdict.predicted_vs_actual(y_true, y_pred,
      title: "Predicted vs actual",
      width: 300,
      height: 280
    )
  end

  defp panel(:residuals, :regression, y_true, y_pred, _opts) do
    Verdict.residuals(y_true, y_pred, title: "Residuals", width: 300, height: 280)
  end

  defp panel(:residual_distribution, :regression, y_true, y_pred, _opts) do
    Verdict.residual_distribution(y_true, y_pred,
      title: "Residual distribution",
      width: 320,
      height: 240
    )
  end

  defp panel(:qq, :regression, y_true, y_pred, _opts) do
    Verdict.qq_plot(Nx.subtract(y_pred, y_true),
      title: "Normal Q-Q",
      width: 280,
      height: 280
    )
  end

  defp panel(plot, kind, _y_true, _y_predicted, _opts) do
    raise ArgumentError,
          "#{inspect(plot)} is not one of the #{kind} plots. " <>
            "Available: #{inspect(available(kind))}"
  end

  defp available(:classification) do
    [
      :confusion_matrix,
      :roc,
      :precision_recall,
      :det,
      :threshold,
      :calibration,
      :score_distribution
    ]
  end

  defp available(:regression), do: @regression

  defp grid_opts(opts) do
    base = [columns: opts[:columns]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end

  # An option declared but handed nil fails validation, so unset ones are
  # dropped rather than passed along empty.
  defp compact(opts), do: Enum.reject(opts, fn {_key, value} -> is_nil(value) end)
end
