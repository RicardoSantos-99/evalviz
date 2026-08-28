defmodule Verdict.GridSearch do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 x: [type: :atom, doc: "Hyperparameter on the x axis. Inferred when two vary."],
                 y: [type: :atom, doc: "Hyperparameter on the y axis. Inferred when two vary."],
                 metric: [
                   type: :non_neg_integer,
                   default: 0,
                   doc: "Which entry of each result's score to plot."
                 ],
                 metric_name: [type: :string, default: "Score"],
                 best: [
                   type: {:in, [:max, :min, :none]},
                   default: :max,
                   doc: """
                   Which end is better. It outlines the winning cell, and darker
                   always means better, so `:min` reverses the colour ramp
                   rather than leaving the reader to invert it.
                   """
                 ],
                 values: [type: :boolean, default: true, doc: "Writes each score in its cell."],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 460],
                 height: [type: :pos_integer, default: 380]
               )

  def schema, do: @opts_schema

  def plot(results, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if results == [] do
      raise ArgumentError, "expected at least one grid search result"
    end

    varying = varying(results)
    {x, y} = axes(varying, opts)
    collapsed = Enum.reject(varying, &(&1 in [x, y]))

    cells = cells(results, x, y, opts)
    best = best(cells, opts[:best])
    order = {axis_order(results, x), axis_order(results, y)}

    Vl.new(vl_opts(opts, {x, y}, collapsed, best))
    |> Vl.data_from_values(shade(cells, opts))
    |> Vl.layers(layers(order, x, y, best, opts))
  end

  # Sorted on the values themselves, not their labels: alphabetically "20.0"
  # comes before "5.0", which would put the axis out of order.
  defp axis_order(results, key) do
    results
    |> Enum.map(&Keyword.fetch!(&1.hyperparameters, key))
    |> Enum.uniq()
    |> Enum.sort()
    |> Enum.map(&to_string/1)
  end

  defp varying(results) do
    results
    |> Enum.flat_map(&Keyword.keys(&1.hyperparameters))
    |> Enum.uniq()
    |> Enum.filter(fn key ->
      results |> Enum.map(&Keyword.get(&1.hyperparameters, key)) |> Enum.uniq() |> length() > 1
    end)
  end

  defp axes(varying, opts) do
    case {opts[:x], opts[:y], varying} do
      {nil, nil, [x, y]} ->
        {x, y}

      {nil, nil, other} ->
        raise ArgumentError,
              "expected exactly two hyperparameters to vary, got #{inspect(other)}. " <>
                "Pass :x and :y to choose which two to plot."

      {x, y, _} when not is_nil(x) and not is_nil(y) ->
        {x, y}

      _ ->
        raise ArgumentError, "expected both :x and :y, or neither"
    end
  end

  # More than two parameters may vary, and a heatmap has room for two. The rest
  # are collapsed by keeping the best score for each cell, which the subtitle
  # says out loud rather than quietly dropping results.
  defp cells(results, x, y, opts) do
    keep = if opts[:best] == :min, do: &Enum.min_by/2, else: &Enum.max_by/2

    results
    |> Enum.group_by(fn result ->
      {Keyword.fetch!(result.hyperparameters, x), Keyword.fetch!(result.hyperparameters, y)}
    end)
    |> Enum.map(fn {{x_value, y_value}, group} ->
      scores = Enum.map(group, &score(&1, opts[:metric]))
      score = if opts[:best] == :none, do: hd(scores), else: keep.(scores, & &1)

      %{
        "x" => to_string(x_value),
        "y" => to_string(y_value),
        "score" => score,
        "label" => score |> Internal.round_to(3) |> to_string()
      }
    end)
    |> Enum.sort_by(&{&1["y"], &1["x"]})
  end

  defp score(%{score: score}, metric) do
    case Nx.rank(score) do
      0 -> Nx.to_number(score)
      1 -> score |> Nx.to_flat_list() |> fetch_metric(metric)
      rank -> raise ArgumentError, "expected a scalar or rank-1 score, got rank #{rank}"
    end
  end

  defp fetch_metric(scores, metric) do
    Enum.at(scores, metric) ||
      raise ArgumentError,
            "expected :metric to index the #{length(scores)} scores each result carries, " <>
              "got #{metric}"
  end

  defp best(_cells, :none), do: nil
  defp best(cells, :max), do: Enum.max_by(cells, & &1["score"])
  defp best(cells, :min), do: Enum.min_by(cells, & &1["score"])

  # Carried on the row rather than tested against the raw score, so the text
  # stays readable whichever way the ramp runs.
  defp shade(cells, opts) do
    scores = Enum.map(cells, & &1["score"])
    {min, max} = Enum.min_max(scores)
    span = max - min

    Enum.map(cells, fn cell ->
      shade = if span == 0, do: 1.0, else: (cell["score"] - min) / span
      Map.put(cell, "shade", if(opts[:best] == :min, do: 1 - shade, else: shade))
    end)
  end

  defp layers({xs, ys}, x, y, best, opts) do
    heatmap =
      Vl.new()
      |> Vl.mark(:rect, tooltip: true)
      |> Vl.encode_field(:x, "x",
        type: :nominal,
        sort: xs,
        title: to_string(x),
        axis: [label_angle: 0]
      )
      |> Vl.encode_field(:y, "y", type: :nominal, sort: ys, title: to_string(y))
      |> Vl.encode_field(:color, "score",
        type: :quantitative,
        title: opts[:metric_name],
        scale: [scheme: Theme.heatmap_scheme(), reverse: opts[:best] == :min]
      )

    text = if opts[:values], do: [text_layer(xs, ys)], else: []
    marker = if best, do: [best_layer(best, xs, ys)], else: []

    [heatmap] ++ text ++ marker
  end

  defp text_layer(xs, ys) do
    Vl.new()
    |> Vl.mark(:text, font_size: 11)
    |> Vl.encode_field(:x, "x", type: :nominal, sort: xs)
    |> Vl.encode_field(:y, "y", type: :nominal, sort: ys)
    |> Vl.encode_field(:text, "label", type: :nominal)
    |> Vl.encode(:color,
      condition: [test: "datum.shade > 0.65", value: "white"],
      value: "black"
    )
  end

  defp best_layer(best, xs, ys) do
    Vl.new()
    |> Vl.data_from_values([%{"x" => best["x"], "y" => best["y"]}])
    |> Vl.mark(:rect, fill: nil, stroke: Theme.secondary(), stroke_width: 3)
    |> Vl.encode_field(:x, "x", type: :nominal, sort: xs)
    |> Vl.encode_field(:y, "y", type: :nominal, sort: ys)
  end

  defp vl_opts(opts, axes, collapsed, best) do
    base = [width: opts[:width], height: opts[:height]]

    subtitle =
      [best_note(best, axes), collapsed_note(collapsed, opts)]
      |> Enum.reject(&is_nil/1)
      |> Enum.join(", ")
      |> case do
        "" -> nil
        text -> text
      end

    case {opts[:title], subtitle} do
      {nil, nil} -> base
      {title, nil} -> [{:title, title} | base]
      {nil, subtitle} -> [{:title, [text: "", subtitle: subtitle]} | base]
      {title, subtitle} -> [{:title, [text: title, subtitle: subtitle]} | base]
    end
  end

  defp best_note(nil, _axes), do: nil

  defp best_note(best, {x, y}) do
    "best #{Internal.round_to(best["score"], 4)} at #{x} #{best["x"]}, #{y} #{best["y"]}"
  end

  defp collapsed_note([], _opts), do: nil

  defp collapsed_note(collapsed, opts) do
    direction = if opts[:best] == :min, do: "lowest", else: "highest"
    "#{direction} over #{Enum.map_join(collapsed, " and ", &to_string/1)}"
  end
end
