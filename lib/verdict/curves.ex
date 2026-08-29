defmodule Verdict.Curves do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Multiclass
  alias Verdict.Theme
  alias Scholar.Metrics.Classification, as: Metrics
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 title: [type: :string, doc: "Chart title."],
                 sample_weights: Internal.weights_option(),
                 class_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: """
                   Names for each class when `y_score` has one column per class.
                   Defaults to `0..num_classes - 1`.
                   """
                 ],
                 average: [
                   type: {:or, [{:in, [:micro, :macro]}, {:list, {:in, [:micro, :macro]}}]},
                   doc: """
                   Averaged curves to draw alongside the per-class ones, for a
                   `{n, num_classes}` score matrix. `:micro` pools every
                   (sample, class) pair into one binary problem; `:macro`
                   interpolates the per-class curves onto a shared grid and
                   averages them. Accepts either or both.
                   """
                 ],
                 facet: [
                   type: :boolean,
                   default: false,
                   doc: """
                   Draws a panel per curve instead of overlaying them.
                   One-vs-rest curves pile up fast, and past a handful of
                   classes the overlay stops being readable.
                   """
                 ],
                 columns: [
                   type: :pos_integer,
                   default: 3,
                   doc: "Panels per row when faceting."
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
      domains: %{x: [0, 1], y: [0, 1]}
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
      domains: %{x: [0, 1], y: [0, 1]}
    }
  end

  defp config(:det) do
    %{
      x: {"fpr", "False positive rate"},
      y: {"fnr", "False negative rate"},
      compute: &Metrics.det_curve/4,
      summary: nil,
      chance: :none,
      domains: %{x: nil, y: nil}
    }
  end

  # Both read off the ROC curve rather than being counted again. The share of
  # samples contacted is what the two rates mix to, and the share of positives
  # captured is the true positive rate itself, so the ties and the weighting
  # are handled once, in Scholar, and the two plots cannot drift from the ROC
  # of the same model.
  defp config(:cumulative_gain) do
    %{
      x: {"contacted", "Share of samples contacted"},
      y: {"captured", "Share of positives captured"},
      compute: &targeting(&1, &2, &3, &4, :gain),
      summary: nil,
      chance: :diagonal,
      domains: %{x: [0, 1], y: [0, 1]}
    }
  end

  defp config(:lift) do
    %{
      x: {"contacted", "Share of samples contacted"},
      y: {"lift", "Times better than random"},
      compute: &targeting(&1, &2, &3, &4, :lift),
      summary: nil,
      chance: :unit,
      domains: %{x: [0, 1], y: nil}
    }
  end

  # Contacting nobody captures nobody, which is a point on the gain curve and
  # a zero over zero on the lift, so lift starts at the first contact.
  defp targeting(y_true, y_score, dvi, weights, kind) do
    {fpr, tpr, thresholds} = Metrics.roc_curve(y_true, y_score, dvi, weights)

    # An absent weight arrives as the number one rather than as a tensor of
    # them, and the share is a weighted one either way.
    weights = Nx.broadcast(weights, {Nx.axis_size(y_true, 0)})
    positive_rate = Nx.divide(Nx.sum(Nx.multiply(y_true, weights)), Nx.sum(weights))

    contacted =
      Nx.add(
        Nx.multiply(tpr, positive_rate),
        Nx.multiply(fpr, Nx.subtract(1, positive_rate))
      )

    case kind do
      :gain ->
        {contacted, tpr, thresholds}

      :lift ->
        {contacted[1..-1//1], Nx.divide(tpr[1..-1//1], contacted[1..-1//1]), thresholds[1..-1//1]}
    end
  end

  def plot(kind, series, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    config = config(kind)

    curves = Enum.flat_map(series, &curves_for(&1, config, opts))
    multi? = length(curves) > 1

    values = Enum.flat_map(curves, & &1.points)

    if opts[:facet] and multi? do
      facet(config, values, curves, opts)
    else
      Vl.new(vl_opts(opts, curves, multi?))
      |> Vl.data_from_values(values)
      |> Vl.layers(layers(config, values, curves, multi?, false, opts))
    end
  end

  # Each panel already carries its curve's name, summary included, so the
  # colour legend has nothing left to say and the channel goes unused.
  defp facet(config, values, curves, opts) do
    child =
      Vl.new(width: opts[:width], height: opts[:height])
      |> Vl.layers(layers(config, values, curves, false, true, opts))

    # `columns` belongs to the outer spec, not to the facet definition, and the
    # header would otherwise print the field's own name above the panels.
    Vl.new([columns: opts[:columns]] ++ title(opts[:title]))
    |> Vl.data_from_values(values)
    |> Vl.facet([field: "series", type: :nominal, title: nil], child)
  end

  defp title(nil), do: []
  defp title(text), do: [title: text]

  defp curves_for(entry, config, opts) do
    per_class = entry |> Multiclass.per_class(opts) |> Enum.map(&curve_points(&1, config, opts))
    micro = entry |> Multiclass.micro(opts) |> Enum.map(&curve_points(&1, config, opts))

    per_class ++ micro ++ macro(entry, per_class, config, opts)
  end

  defp curve_points({label, y_true, y_score}, config, opts) do
    Internal.assert_paired!(y_true, y_score, "y_true", "y_score")
    assert_binary!(y_true)

    weights = Internal.weights(opts[:sample_weights], Nx.axis_size(y_true, 0))
    dvi = Metrics.distinct_value_indices(y_score)

    {x, y, _thresholds} = config.compute.(y_true, y_score, dvi, weights)
    {x_field, _} = config.x
    {y_field, _} = config.y

    {legend, summary} = legend_label(label, config, y_true, y_score, dvi, weights)

    points =
      x
      |> Internal.points(y, x_field, y_field)
      |> Enum.map(&Map.put(&1, "series", legend))
      |> with_baseline(config, y_true)

    %{points: points, legend: legend, y_true: y_true, summary: summary}
  end

  defp legend_label(label, %{summary: nil}, _y_true, _score, _dvi, _weights), do: {label, nil}

  defp legend_label(label, %{summary: {name, fun}}, y_true, score, dvi, weights) do
    value = fun.(y_true, score, dvi, weights) |> Nx.to_number() |> Float.round(3)
    summary = "#{name} = #{value}"

    {if(label, do: "#{label} (#{summary})", else: summary), value}
  end

  defp macro({label, _y_true, y_score}, per_class, config, opts) do
    if Multiclass.macro?(y_score, opts), do: [macro_curve(label, per_class, config)], else: []
  end

  # Every per-class curve is resampled onto the union of their x values and the
  # y values averaged there, which is how scikit-learn builds this curve.
  defp macro_curve(label, per_class, config) do
    {x_field, _} = config.x
    {y_field, _} = config.y

    curves = Enum.map(per_class, &as_curve(&1.points, x_field, y_field))
    grid = curves |> Enum.concat() |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> Enum.sort()
    legend = macro_legend(label, config, per_class)

    points =
      grid
      |> Enum.map(fn x ->
        %{
          x_field => x,
          y_field => mean(Enum.map(curves, &interpolate(&1, x))),
          "series" => legend
        }
      end)
      |> with_macro_baseline(per_class)

    %{points: points, legend: legend, y_true: nil, summary: nil}
  end

  # Stable, so points repeating an x keep the order the curve gave them.
  defp as_curve(points, x_field, y_field) do
    points
    |> Enum.map(&{&1[x_field], &1[y_field]})
    |> Enum.sort_by(&elem(&1, 0))
  end

  # These are step curves, so an x can carry several y values. Landing on one
  # takes the last, the top of the vertical segment; crossing it interpolates
  # from the first. That is what numpy's interp does, and departing from it
  # moves the averaged curve.
  defp interpolate(points, x) do
    {before, rest} = Enum.split_while(points, fn {px, _} -> px < x end)

    case {List.last(before), rest} do
      {_, [{^x, _} | _]} -> rest |> Enum.take_while(&(elem(&1, 0) == x)) |> List.last() |> elem(1)
      {nil, [{_, y} | _]} -> y
      {{_, y}, []} -> y
      {{x0, y0}, [{x1, y1} | _]} -> y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    end
  end

  defp macro_legend(label, %{summary: nil}, _per_class) do
    Multiclass.compose(label, "macro-average")
  end

  defp macro_legend(label, %{summary: {name, _fun}}, per_class) do
    value = per_class |> Enum.map(& &1.summary) |> mean() |> Float.round(3)
    "#{Multiclass.compose(label, "macro-average")} (#{name} = #{value})"
  end

  defp mean(values), do: Enum.sum(values) / length(values)

  # This curve averages the per-class ones, so its baseline is the average of
  # theirs. Rows also have to agree on their columns, since a ragged set is
  # rejected outright rather than read as a missing value.
  defp with_macro_baseline(points, per_class) do
    rates =
      per_class
      |> Enum.map(&(&1.points |> List.first(%{}) |> Map.get("baseline")))
      |> Enum.reject(&is_nil/1)

    case rates do
      [] -> points
      rates -> Enum.map(points, &Map.put(&1, "baseline", mean(rates)))
    end
  end

  # A curve's own share of positives, carried on its rows so a faceted panel can
  # draw the baseline that belongs to it.
  defp with_baseline(points, %{chance: :positive_rate}, y_true) do
    rate = y_true |> Nx.mean() |> Nx.to_number()
    Enum.map(points, &Map.put(&1, "baseline", rate))
  end

  defp with_baseline(points, _config, _y_true), do: points

  defp layers(config, values, curves, multi?, faceted?, opts) do
    {x_field, x_title} = config.x
    {y_field, y_title} = config.y

    # Marking each threshold helps when there are few of them and turns into
    # noise when there are many.
    show_points = length(values) <= 25 * max(length(curves), 1)

    curve =
      Vl.new()
      |> Vl.mark(:line, point: show_points, tooltip: true, interpolate: "step-after")
      |> Vl.encode_field(:x, x_field, [type: :quantitative, title: x_title] ++ scale(config, :x))
      |> Vl.encode_field(:y, y_field, [type: :quantitative, title: y_title] ++ scale(config, :y))
      |> encode_series(multi?)

    case chance_layer(config.chance, curves, faceted?, opts) do
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

  defp scale(%{domains: domains}, axis) do
    case domains[axis] do
      nil -> []
      [min, max] -> [scale: [domain: [min, max], nice: false]]
    end
  end

  defp chance_layer(:none, _curves, _faceted?, _opts), do: nil

  defp chance_layer(kind, curves, faceted?, opts) do
    if opts[:chance_line], do: build_chance_layer(kind, curves, faceted?), else: nil
  end

  defp build_chance_layer(:diagonal, _curves, _faceted?) do
    reference_line([%{"x" => 0, "y" => 0}, %{"x" => 1, "y" => 1}])
  end

  # Lift is already stated as a multiple of random, so no skill is a flat one
  # whatever the share of positives happens to be.
  defp build_chance_layer(:unit, _curves, _faceted?) do
    reference_line([%{"x" => 0, "y" => 1}, %{"x" => 1, "y" => 1}])
  end

  # A panel holds one class, so it can draw that class's own share rather than
  # going without because the classes disagree.
  defp build_chance_layer(:positive_rate, _curves, true) do
    Vl.new()
    |> Vl.mark(:rule, Theme.reference_mark())
    |> Vl.encode_field(:y, "baseline", type: :quantitative)
  end

  # For precision-recall, a no-skill classifier sits at the share of positives,
  # so the baseline is a horizontal line rather than the diagonal. One-vs-rest
  # classes each have their own share, and a single line would misread as
  # theirs, so it is only drawn when every curve agrees on it.
  defp build_chance_layer(:positive_rate, curves, false) do
    rates =
      curves
      |> Enum.filter(& &1.y_true)
      |> Enum.map(&(&1.y_true |> Nx.mean() |> Nx.to_number()))
      |> Enum.uniq()

    case rates do
      [rate] -> reference_line([%{"x" => 0, "y" => rate}, %{"x" => 1, "y" => rate}])
      _ -> nil
    end
  end

  defp reference_line(points) do
    Vl.new()
    |> Vl.data_from_values(points)
    |> Vl.mark(:line, Theme.reference_mark())
    |> Vl.encode_field(:x, "x", type: :quantitative)
    |> Vl.encode_field(:y, "y", type: :quantitative)
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
