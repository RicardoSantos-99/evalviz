defmodule EvalViz.LearningCurve do
  @moduledoc false

  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 score_title: [
                   type: :string,
                   default: "Score",
                   doc: "Label for the score axis."
                 ],
                 size_title: [
                   type: :string,
                   default: "Training examples",
                   doc: "Label for the x axis."
                 ],
                 spread: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Shades one standard deviation around each mean. Needs a score
                   per fold; with a single score per size there is no spread to
                   draw.
                   """
                 ],
                 labels: [
                   type: {:list, :string},
                   default: ["Training", "Validation"],
                   doc: "Names for the two curves."
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 500],
                 height: [type: :pos_integer, default: 380]
               )

  def schema, do: @opts_schema

  def plot(train_sizes, train_scores, validation_scores, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    sizes = to_list(train_sizes)
    train = to_folds(train_scores, length(sizes), "train_scores")
    validation = to_folds(validation_scores, length(sizes), "validation_scores")

    [train_label, validation_label] = opts[:labels]

    values =
      series(sizes, train, train_label) ++ series(sizes, validation, validation_label)

    folded? = train |> hd() |> length() > 1

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(opts, folded?))
  end

  defp series(sizes, folds, label) do
    Enum.zip_with(sizes, folds, fn size, scores ->
      mean = Enum.sum(scores) / length(scores)
      deviation = standard_deviation(scores, mean)

      %{
        "size" => size,
        "score" => mean,
        "lower" => mean - deviation,
        "upper" => mean + deviation,
        "series" => label
      }
    end)
  end

  defp standard_deviation(scores, _mean) when length(scores) < 2, do: 0.0

  defp standard_deviation(scores, mean) do
    variance =
      scores
      |> Enum.map(&((&1 - mean) * (&1 - mean)))
      |> Enum.sum()
      |> Kernel./(length(scores))

    :math.sqrt(variance)
  end

  defp layers(opts, folded?) do
    line =
      Vl.new()
      |> Vl.mark(:line, point: true, tooltip: true)
      |> Vl.encode_field(:x, "size", type: :quantitative, title: opts[:size_title])
      |> Vl.encode_field(:y, "score",
        type: :quantitative,
        title: opts[:score_title],
        scale: [zero: false]
      )
      |> Vl.encode_field(:color, "series",
        type: :nominal,
        title: nil,
        scale: [domain: opts[:labels], range: Theme.categorical(2)]
      )

    if opts[:spread] and folded? do
      [band(opts), line]
    else
      [line]
    end
  end

  # Drawn under the lines so the means stay legible on top of it.
  defp band(opts) do
    Vl.new()
    |> Vl.mark(:area, opacity: 0.2)
    |> Vl.encode_field(:x, "size", type: :quantitative)
    |> Vl.encode_field(:y, "lower", type: :quantitative, title: opts[:score_title])
    |> Vl.encode_field(:y2, "upper")
    |> Vl.encode_field(:color, "series",
      type: :nominal,
      legend: nil,
      scale: [domain: opts[:labels], range: Theme.categorical(2)]
    )
  end

  defp to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp to_list(list) when is_list(list), do: list

  # A score per size, or a score per fold per size.
  defp to_folds(%Nx.Tensor{} = tensor, expected, name) do
    case Nx.rank(tensor) do
      1 -> tensor |> Nx.to_flat_list() |> Enum.map(&[&1])
      2 -> Nx.to_list(tensor)
      rank -> raise ArgumentError, "expected #{name} to be rank 1 or 2, got rank #{rank}"
    end
    |> check_length(expected, name)
  end

  defp to_folds(list, expected, name) when is_list(list) do
    list
    |> Enum.map(fn
      scores when is_list(scores) -> scores
      score -> [score]
    end)
    |> check_length(expected, name)
  end

  defp check_length(folds, expected, name) do
    if length(folds) != expected do
      raise ArgumentError,
            "expected #{name} to have one entry per training size, " <>
              "got #{length(folds)} for #{expected} sizes"
    end

    folds
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
