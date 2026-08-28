defmodule Verdict.LearningCurve do
  @moduledoc false

  alias Verdict.Folds
  alias Verdict.Theme
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
                   draw. Worth turning off when comparing several models, since
                   every one of them shades twice.
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

  def plot(train_sizes, series, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    sizes = to_list(train_sizes)
    [train_label, validation_label] = opts[:labels]

    {values, folded?} =
      Enum.map_reduce(series, false, fn {model, train_scores, validation_scores}, folded? ->
        train = Folds.normalize(train_scores, length(sizes), "train_scores", "training size")

        validation =
          Folds.normalize(validation_scores, length(sizes), "validation_scores", "training size")

        rows =
          Folds.rows(sizes, train, train_label, "size") ++
            Folds.rows(sizes, validation, validation_label, "size")

        {tag(rows, model), folded? or Folds.folded?(train)}
      end)

    values = List.flatten(values)
    models = models(values)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(models, folded?, opts))
  end

  defp tag(rows, nil), do: rows
  defp tag(rows, model), do: Enum.map(rows, &Map.put(&1, "model", model))

  defp models(values) do
    values |> Enum.map(& &1["model"]) |> Enum.uniq() |> Enum.reject(&is_nil/1)
  end

  defp layers(models, folded?, opts) do
    line =
      Vl.new()
      |> Vl.mark(:line, point: true, tooltip: true)
      |> Vl.encode_field(:x, "size", type: :quantitative, title: opts[:size_title])
      |> Vl.encode_field(:y, "score",
        type: :quantitative,
        title: opts[:score_title],
        scale: [zero: false]
      )
      |> encode_series(models, opts)

    if opts[:spread] and folded? do
      [band(models, opts), line]
    else
      [line]
    end
  end

  # With one model colour is free to carry training against validation. With
  # several it has to carry the model, and the dash pattern takes over the
  # split, so the two questions stay separable.
  defp encode_series(layer, [], opts) do
    Vl.encode_field(layer, :color, "series",
      type: :nominal,
      title: nil,
      scale: [domain: opts[:labels], range: Theme.categorical(2)]
    )
  end

  defp encode_series(layer, models, opts) do
    layer
    |> Vl.encode_field(:color, "model",
      type: :nominal,
      title: nil,
      scale: [domain: models, range: Theme.categorical(length(models))]
    )
    |> Vl.encode_field(:stroke_dash, "series",
      type: :nominal,
      title: nil,
      legend: Theme.dash_legend(),
      scale: [domain: opts[:labels]]
    )
  end

  # Drawn under the lines so the means stay legible on top of it.
  defp band(models, opts) do
    Vl.new()
    |> Vl.mark(:area, opacity: 0.2)
    |> Vl.encode_field(:x, "size", type: :quantitative)
    |> Vl.encode_field(:y, "lower", type: :quantitative, title: opts[:score_title])
    |> Vl.encode_field(:y2, "upper")
    |> encode_band_series(models, opts)
  end

  # No `legend: nil` here: the band shares this field with the line, and
  # suppressing it on one layer suppresses the merged legend for both, which
  # leaves two coloured curves with nothing saying which is which.
  defp encode_band_series(layer, [], opts) do
    Vl.encode_field(layer, :color, "series",
      type: :nominal,
      title: nil,
      scale: [domain: opts[:labels], range: Theme.categorical(2)]
    )
  end

  # An area takes no dash pattern, so the split rides on `detail`, which keeps
  # the two bands of a model apart without spending another visual channel.
  defp encode_band_series(layer, models, _opts) do
    layer
    |> Vl.encode_field(:color, "model",
      type: :nominal,
      title: nil,
      scale: [domain: models, range: Theme.categorical(length(models))]
    )
    |> Vl.encode_field(:detail, "series", type: :nominal)
  end

  defp to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp to_list(list) when is_list(list), do: list

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
