defmodule EvalViz.FoldScores do
  @moduledoc false

  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 metric_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "One name per metric. Defaults to `Metric 0`, `Metric 1`, ..."
                 ],
                 mean_line: [
                   type: :boolean,
                   default: true,
                   doc: "Draws the mean each fold is scattered around."
                 ],
                 point_size: [type: :pos_integer, default: 70],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 440],
                 height: [type: :pos_integer, default: 200]
               )

  def schema, do: @opts_schema

  def plot(scores, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    metrics = metrics(scores)
    names = metric_names(opts[:metric_names], length(metrics))

    case Enum.zip(names, metrics) do
      [single] -> panel(single, size(opts) ++ title(opts[:title]), opts)
      many -> concat(many, opts)
    end
  end

  defp concat(panels, opts) do
    views = Enum.map(panels, &panel(&1, size(opts) ++ title(elem(&1, 0)), opts))

    Vl.new(title(opts[:title]))
    |> Vl.concat(views, :vertical)
  end

  defp panel({name, values}, new_opts, opts) do
    rows =
      values
      |> Enum.with_index()
      |> Enum.map(fn {score, fold} -> %{"fold" => fold, "score" => score, "metric" => name} end)

    Vl.new(with_subtitle(new_opts, values))
    |> Vl.data_from_values(rows)
    |> Vl.layers(layers(values, opts))
  end

  # Points, not a line: folds are interchangeable, and joining them would draw
  # a trend across an order that carries no meaning.
  defp layers(values, opts) do
    points =
      Vl.new()
      |> Vl.mark(:point,
        filled: true,
        size: opts[:point_size],
        opacity: 0.85,
        color: Theme.primary(),
        tooltip: true
      )
      |> Vl.encode_field(:x, "fold",
        type: :quantitative,
        title: "Fold",
        axis: [tick_min_step: 1, format: "d"]
      )
      |> Vl.encode_field(:y, "score", type: :quantitative, title: nil, scale: [zero: false])

    if opts[:mean_line], do: [mean_layer(values), points], else: [points]
  end

  defp mean_layer(values) do
    Vl.new()
    |> Vl.data_from_values([%{"mean" => mean(values)}])
    |> Vl.mark(:rule, Theme.reference_mark())
    |> Vl.encode_field(:y, "mean", type: :quantitative)
  end

  defp metrics(%Nx.Tensor{} = scores) do
    case Nx.rank(scores) do
      1 -> [Nx.to_flat_list(scores)]
      2 -> Nx.to_list(scores)
      rank -> raise ArgumentError, "expected scores to be rank 1 or 2, got rank #{rank}"
    end
  end

  defp metrics(list) when is_list(list) do
    case list do
      [head | _] when is_list(head) -> list
      _ -> [list]
    end
  end

  defp metric_names(nil, n), do: Enum.map(0..(n - 1), &"Metric #{&1}")

  defp metric_names(names, n) when length(names) == n, do: Enum.map(names, &to_string/1)

  defp metric_names(names, n) do
    raise ArgumentError,
          "expected :metric_names to have one entry per metric, " <>
            "got #{length(names)} names for #{n} metrics"
  end

  # The spread is the point of the plot, so it is stated as well as drawn.
  defp with_subtitle(base, values) do
    mean = mean(values)
    text = "mean #{Float.round(mean, 4)} ± #{Float.round(deviation(values, mean), 4)}"

    Keyword.put(base, :title, text: Keyword.get(base, :title, ""), subtitle: text)
  end

  defp mean(values), do: Enum.sum(values) / length(values)

  defp deviation(values, _mean) when length(values) < 2, do: 0.0

  defp deviation(values, mean) do
    variance =
      values
      |> Enum.map(&((&1 - mean) * (&1 - mean)))
      |> Enum.sum()
      |> Kernel./(length(values))

    :math.sqrt(variance)
  end

  defp size(opts), do: [width: opts[:width], height: opts[:height]]

  defp title(nil), do: []
  defp title(text), do: [title: text]
end
