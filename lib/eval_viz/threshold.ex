defmodule EvalViz.Threshold do
  @moduledoc false

  alias EvalViz.Internal
  alias EvalViz.Theme
  alias Scholar.Metrics.Classification, as: Metrics
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 metrics: [
                   type: {:list, {:in, [:precision, :recall, :f1]}},
                   default: [:precision, :recall, :f1],
                   doc: "Which curves to draw."
                 ],
                 mark_best_f1: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Marks the threshold with the highest F1. Only drawn when `:f1`
                   is among the metrics.
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
    Internal.assert_paired!(y_true, y_score, "y_true", "y_score")
    assert_binary!(y_true)

    rows = curve(y_true, y_score, opts[:metrics])
    best = best_f1(rows, opts)

    Vl.new(vl_opts(opts, best))
    |> Vl.data_from_values(rows)
    |> Vl.layers(layers(rows, best, opts))
  end

  # precision_recall_curve returns one more precision/recall than it does
  # thresholds: the final pair is the degenerate point where nothing is
  # predicted positive, which has no threshold to sit at.
  defp curve(y_true, y_score, metrics) do
    dvi = Metrics.distinct_value_indices(y_score)
    {precision, recall, thresholds} = Metrics.precision_recall_curve(y_true, y_score, dvi)

    thresholds = Nx.to_flat_list(thresholds)
    precision = precision |> Nx.to_flat_list() |> Enum.take(length(thresholds))
    recall = recall |> Nx.to_flat_list() |> Enum.take(length(thresholds))

    [thresholds, precision, recall]
    |> Enum.zip()
    |> Enum.flat_map(fn {threshold, p, r} ->
      values = %{precision: p, recall: r, f1: f1(p, r)}

      Enum.map(metrics, fn metric ->
        %{
          "threshold" => threshold,
          "value" => Map.fetch!(values, metric),
          "metric" => Map.fetch!(@names, metric)
        }
      end)
    end)
  end

  defp f1(precision, recall) when precision + recall == 0, do: 0.0
  defp f1(precision, recall), do: 2 * precision * recall / (precision + recall)

  defp best_f1(rows, opts) do
    if opts[:mark_best_f1] and :f1 in opts[:metrics] do
      rows
      |> Enum.filter(&(&1["metric"] == "F1"))
      |> Enum.max_by(& &1["value"], fn -> nil end)
    end
  end

  defp layers(rows, best, opts) do
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
      |> Vl.encode_field(:color, "metric",
        type: :nominal,
        title: nil,
        scale: [domain: metric_names, range: Theme.categorical(length(metric_names))]
      )

    if best do
      [best_layer(best), lines]
    else
      [lines]
    end
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
      raise ArgumentError, "expected y_true to hold only 0 and 1, got #{inspect(distinct)}"
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
