defmodule EvalViz.LearningCurve do
  @moduledoc false

  alias EvalViz.Folds
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
    train = Folds.normalize(train_scores, length(sizes), "train_scores", "training size")

    validation =
      Folds.normalize(validation_scores, length(sizes), "validation_scores", "training size")

    [train_label, validation_label] = opts[:labels]

    values =
      Folds.rows(sizes, train, train_label, "size") ++
        Folds.rows(sizes, validation, validation_label, "size")

    folded? = Folds.folded?(train)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(opts, folded?))
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
    # No `legend: nil` here: the band shares this field with the line, and
    # suppressing it on one layer suppresses the merged legend for both, which
    # leaves two coloured curves with nothing saying which is which.
    |> Vl.encode_field(:color, "series",
      type: :nominal,
      title: nil,
      scale: [domain: opts[:labels], range: Theme.categorical(2)]
    )
  end

  defp to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp to_list(list) when is_list(list), do: list

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
