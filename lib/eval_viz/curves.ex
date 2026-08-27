defmodule EvalViz.Curves do
  @moduledoc false

  alias EvalViz.Internal
  alias EvalViz.Theme
  alias Scholar.Metrics.Classification, as: Metrics
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 title: [type: :string, doc: "Chart title."],
                 sample_weights: [
                   type: {:or, [{:list, {:or, [:float, :integer]}}, :any]},
                   doc: "Per-sample weights, as a list or rank-1 tensor."
                 ],
                 chance_line: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Draws the no-skill reference. For ROC that is the diagonal a
                   random ranker would trace; ignored by curves that have no
                   meaningful constant baseline.
                   """
                 ],
                 width: [type: :pos_integer, default: 450],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  # Each curve differs only in which pair of rates it draws and how it is
  # summarised in the legend.
  defp config(:roc) do
    %{
      x: {"fpr", "False positive rate"},
      y: {"tpr", "True positive rate"},
      compute: &Metrics.roc_curve/4,
      summary: {"AUC", &Metrics.roc_auc_score/4},
      chance: :diagonal,
      domain: [0, 1]
    }
  end

  defp config(:precision_recall) do
    %{
      x: {"recall", "Recall"},
      y: {"precision", "Precision"},
      compute: fn y_true, scores, dvi, weights ->
        {precision, recall, thresholds} =
          Metrics.precision_recall_curve(y_true, scores, dvi, weights)

        {recall, precision, thresholds}
      end,
      summary: {"AP", &Metrics.average_precision_score/4},
      chance: :positive_rate,
      domain: [0, 1]
    }
  end

  defp config(:det) do
    %{
      x: {"fpr", "False positive rate"},
      y: {"fnr", "False negative rate"},
      compute: &Metrics.det_curve/4,
      summary: nil,
      chance: :none,
      domain: nil
    }
  end

  def plot(kind, series, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    config = config(kind)

    curves = Enum.map(series, &curve_points(&1, config, opts))
    multi? = length(curves) > 1

    values = Enum.flat_map(curves, & &1.points)

    Vl.new(vl_opts(opts, curves, multi?))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(config, values, curves, multi?, opts))
  end

  defp curve_points({label, y_true, y_score}, config, opts) do
    Internal.assert_paired!(y_true, y_score, "y_true", "y_score")
    assert_binary!(y_true)

    weights = weights(opts[:sample_weights], Nx.axis_size(y_true, 0))
    dvi = Metrics.distinct_value_indices(y_score)

    {x, y, _thresholds} = config.compute.(y_true, y_score, dvi, weights)
    {x_field, _} = config.x
    {y_field, _} = config.y

    legend = legend_label(label, config, y_true, y_score, dvi, weights)

    points =
      x
      |> Internal.points(y, x_field, y_field)
      |> Enum.map(&Map.put(&1, "series", legend))

    %{points: points, legend: legend, y_true: y_true}
  end

  defp legend_label(label, %{summary: nil}, _y_true, _score, _dvi, _weights), do: label

  defp legend_label(label, %{summary: {name, fun}}, y_true, score, dvi, weights) do
    value = fun.(y_true, score, dvi, weights) |> Nx.to_number() |> Float.round(3)
    summary = "#{name} = #{value}"

    if label, do: "#{label} (#{summary})", else: summary
  end

  defp layers(config, values, curves, multi?, opts) do
    {x_field, x_title} = config.x
    {y_field, y_title} = config.y

    # Marking each threshold helps when there are few of them and turns into
    # noise when there are many.
    show_points = length(values) <= 25 * max(length(curves), 1)

    curve =
      Vl.new()
      |> Vl.mark(:line, point: show_points, tooltip: true, interpolate: "step-after")
      |> Vl.encode_field(:x, x_field, [type: :quantitative, title: x_title] ++ scale(config))
      |> Vl.encode_field(:y, y_field, [type: :quantitative, title: y_title] ++ scale(config))
      |> encode_series(multi?)

    case chance_layer(config.chance, curves, opts) do
      nil -> [curve]
      reference -> [reference, curve]
    end
  end

  # A single curve needs no legend, but the label still belongs somewhere, so it
  # rides along as the colour legend only when there is something to compare.
  defp encode_series(layer, false), do: layer

  defp encode_series(layer, true) do
    Vl.encode_field(layer, :color, "series", type: :nominal, title: nil)
  end

  defp scale(%{domain: nil}), do: []
  defp scale(%{domain: [min, max]}), do: [scale: [domain: [min, max], nice: false]]

  defp chance_layer(:none, _curves, _opts), do: nil

  defp chance_layer(kind, curves, opts) do
    if opts[:chance_line], do: build_chance_layer(kind, curves), else: nil
  end

  defp build_chance_layer(:diagonal, _curves) do
    reference_line([%{"x" => 0, "y" => 0}, %{"x" => 1, "y" => 1}])
  end

  # For precision-recall, a no-skill classifier sits at the share of positives,
  # so the baseline is a horizontal line rather than the diagonal.
  defp build_chance_layer(:positive_rate, curves) do
    rate = hd(curves).y_true |> Nx.mean() |> Nx.to_number()
    reference_line([%{"x" => 0, "y" => rate}, %{"x" => 1, "y" => rate}])
  end

  defp reference_line(points) do
    Vl.new()
    |> Vl.data_from_values(points)
    |> Vl.mark(:line, Theme.reference_mark())
    |> Vl.encode_field(:x, "x", type: :quantitative)
    |> Vl.encode_field(:y, "y", type: :quantitative)
  end

  defp weights(nil, _n), do: 1.0
  defp weights(list, _n) when is_list(list), do: Nx.tensor(list)
  defp weights(tensor, _n), do: tensor

  defp assert_binary!(y_true) do
    distinct = y_true |> Nx.to_flat_list() |> Enum.uniq() |> Enum.sort()

    unless distinct == [0] or distinct == [1] or distinct == [0, 1] do
      raise ArgumentError,
            "expected y_true to hold only 0 and 1, got #{inspect(distinct)}. " <>
              "These curves are defined for binary classification; for a multiclass " <>
              "model, plot one curve per class using a one-vs-rest y_true."
    end

    :ok
  end

  # A lone curve has no legend to carry its AUC, so it goes in the subtitle
  # instead of being computed and then never shown.
  defp vl_opts(opts, curves, multi?) do
    base = [width: opts[:width], height: opts[:height]]
    subtitle = if multi?, do: nil, else: hd(curves).legend

    case {opts[:title], subtitle} do
      {nil, nil} -> base
      {title, nil} -> [{:title, title} | base]
      {nil, subtitle} -> [{:title, [text: "", subtitle: subtitle]} | base]
      {title, subtitle} -> [{:title, [text: title, subtitle: subtitle]} | base]
    end
  end
end
