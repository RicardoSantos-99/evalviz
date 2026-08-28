defmodule Verdict.Threshold do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Multiclass
  alias Verdict.Theme
  alias Scholar.Metrics.Classification, as: Metrics
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 metrics: [
                   type: {:list, {:in, [:precision, :recall, :f1]}},
                   default: [:precision, :recall, :f1],
                   doc: "Which curves to draw."
                 ],
                 class_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: """
                   Names for each class when `y_score` has one column per class.
                   Defaults to `0..num_classes - 1`.
                   """
                 ],
                 average: [
                   type: {:in, [:micro]},
                   doc: """
                   Set to `:micro` to add a set of curves pooling every
                   (sample, class) pair into one binary problem.
                   """
                 ],
                 sample_weights: Internal.weights_option(),
                 facet: [
                   type: :boolean,
                   default: false,
                   doc: """
                   Draws a panel per class instead of overlaying them. That also
                   frees colour to go back to the metric, so the dash pattern is
                   no longer needed to tell the two apart.
                   """
                 ],
                 columns: [
                   type: :pos_integer,
                   default: 3,
                   doc: "Panels per row when faceting."
                 ],
                 mark_best_f1: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Marks the threshold with the highest F1. Only drawn when `:f1`
                   is among the metrics and the input is binary, since one-vs-rest
                   classes each peak somewhere different.
                   """
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 500],
                 height: [type: :pos_integer, default: 380]
               )

  def schema, do: @opts_schema

  @names %{precision: "Precision", recall: "Recall", f1: "F1"}

  def plot(y_true, y_score, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    entry = {nil, y_true, y_score}
    sub_series = Multiclass.per_class(entry, opts) ++ Multiclass.micro(entry, opts)
    multi? = length(sub_series) > 1

    rows = Enum.flat_map(sub_series, &curve(&1, opts))
    best = best_f1(rows, opts, multi?)

    if opts[:facet] and multi? do
      facet(rows, opts)
    else
      Vl.new(vl_opts(opts, best))
      |> Vl.data_from_values(rows)
      |> Vl.layers(layers(rows, best, multi?, opts))
    end
  end

  # A panel per class leaves only the metrics inside it, so colour goes back to
  # carrying the metric and the dash pattern is not needed at all.
  defp facet(rows, opts) do
    child =
      Vl.new(width: opts[:width], height: opts[:height])
      |> Vl.layers(layers(rows, nil, false, opts))

    # `columns` belongs to the outer spec, not to the facet definition, and the
    # header would otherwise print the field's own name above the panels.
    Vl.new([columns: opts[:columns]] ++ title(opts[:title]))
    |> Vl.data_from_values(rows)
    |> Vl.facet([field: "class", type: :nominal, title: nil], child)
  end

  defp title(nil), do: []
  defp title(text), do: [title: text]

  # precision_recall_curve returns one more precision/recall than it does
  # thresholds: the final pair is the degenerate point where nothing is
  # predicted positive, which has no threshold to sit at.
  defp curve({class, y_true, y_score}, opts) do
    Internal.assert_paired!(y_true, y_score, "y_true", "y_score")
    assert_binary!(y_true)

    metrics = opts[:metrics]
    weights = Internal.weights(opts[:sample_weights], Nx.axis_size(y_true, 0))
    dvi = Metrics.distinct_value_indices(y_score)

    {precision, recall, thresholds} =
      Metrics.precision_recall_curve(y_true, y_score, dvi, weights)

    thresholds = Nx.to_flat_list(thresholds)
    precision = precision |> Nx.to_flat_list() |> Enum.take(length(thresholds))
    recall = recall |> Nx.to_flat_list() |> Enum.take(length(thresholds))

    [thresholds, precision, recall]
    |> Enum.zip()
    |> Enum.flat_map(fn {threshold, p, r} ->
      values = %{precision: p, recall: r, f1: f1(p, r)}

      Enum.map(metrics, fn metric ->
        row(threshold, Map.fetch!(values, metric), Map.fetch!(@names, metric), class)
      end)
    end)
  end

  defp row(threshold, value, metric, nil) do
    %{"threshold" => threshold, "value" => value, "metric" => metric}
  end

  defp row(threshold, value, metric, class) do
    %{"threshold" => threshold, "value" => value, "metric" => metric, "class" => class}
  end

  defp f1(precision, recall) when precision + recall == 0, do: 0.0
  defp f1(precision, recall), do: 2 * precision * recall / (precision + recall)

  defp best_f1(_rows, _opts, true), do: nil

  defp best_f1(rows, opts, false) do
    if opts[:mark_best_f1] and :f1 in opts[:metrics] do
      rows
      |> Enum.filter(&(&1["metric"] == "F1"))
      |> Enum.max_by(& &1["value"], fn -> nil end)
    end
  end

  defp layers(rows, best, multi?, opts) do
    metric_names = Enum.map(opts[:metrics], &Map.fetch!(@names, &1))

    lines =
      Vl.new()
      |> Vl.mark(:line, tooltip: true, point: length(rows) <= 60)
      |> Vl.encode_field(:x, "threshold", type: :quantitative, title: "Decision threshold")
      |> Vl.encode_field(:y, "value",
        type: :quantitative,
        title: nil,
        scale: [domain: [0, 1], nice: false]
      )
      |> encode_series(rows, metric_names, multi?)

    if best do
      [best_layer(best), lines]
    else
      [lines]
    end
  end

  defp encode_series(layer, _rows, metric_names, false) do
    Vl.encode_field(layer, :color, "metric",
      type: :nominal,
      title: nil,
      scale: [domain: metric_names, range: Theme.categorical(length(metric_names))]
    )
  end

  # With a class per curve there are two things to tell apart at once, so colour
  # carries the class and the dash pattern the metric.
  defp encode_series(layer, rows, metric_names, true) do
    classes = rows |> Enum.map(& &1["class"]) |> Enum.uniq()

    layer
    |> Vl.encode_field(:color, "class",
      type: :nominal,
      title: nil,
      scale: [domain: classes, range: Theme.categorical(length(classes))]
    )
    |> Vl.encode_field(:stroke_dash, "metric",
      type: :nominal,
      title: nil,
      legend: Theme.dash_legend(),
      scale: [domain: metric_names]
    )
  end

  defp best_layer(best) do
    Vl.new()
    |> Vl.data_from_values([%{"threshold" => best["threshold"]}])
    |> Vl.mark(:rule, Theme.reference_mark())
    |> Vl.encode_field(:x, "threshold", type: :quantitative)
  end

  defp assert_binary!(y_true) do
    distinct = y_true |> Nx.to_flat_list() |> Enum.uniq() |> Enum.sort()

    unless distinct == [0] or distinct == [1] or distinct == [0, 1] do
      raise ArgumentError,
            "expected y_true to hold only 0 and 1, got #{inspect(distinct)}. " <>
              "For a multiclass model pass a y_score with one column per class, " <>
              "and y_true as class indices."
    end

    :ok
  end

  defp vl_opts(opts, best) do
    base = [width: opts[:width], height: opts[:height]]

    subtitle =
      if best do
        "best F1 #{Float.round(best["value"], 3)} at threshold #{Float.round(best["threshold"], 3)}"
      end

    case {opts[:title], subtitle} do
      {nil, nil} -> base
      {title, nil} -> [{:title, title} | base]
      {nil, subtitle} -> [{:title, [text: "", subtitle: subtitle]} | base]
      {title, subtitle} -> [{:title, [text: title, subtitle: subtitle]} | base]
    end
  end
end
