defmodule Verdict.Regression do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 reference_line: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Draws the line a perfect model would sit on: the diagonal for
                   predicted against actual, zero for residuals.
                   """
                 ],
                 point_size: [type: :pos_integer, default: 45],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 450],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  def predicted_vs_actual(series, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    values = rows(series, &Internal.points(&1, &2, "actual", "predicted"))

    # Both axes carry the same quantity, so they share one range: otherwise the
    # diagonal stops meaning "perfect" and the eye reads the spread wrong.
    domain = shared_domain(values, ["actual", "predicted"])

    scatter =
      values
      |> scatter(opts)
      |> Vl.encode_field(:x, "actual",
        type: :quantitative,
        title: "Actual",
        scale: [domain: domain, nice: false]
      )
      |> Vl.encode_field(:y, "predicted",
        type: :quantitative,
        title: "Predicted",
        scale: [domain: domain, nice: false]
      )

    layers = if opts[:reference_line], do: [diagonal(domain), scatter], else: [scatter]

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers)
  end

  def residuals(series, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    values =
      rows(series, fn y_true, y_pred ->
        Internal.points(y_pred, Nx.subtract(y_pred, y_true), "predicted", "residual")
      end)

    scatter =
      values
      |> scatter(opts)
      |> Vl.encode_field(:x, "predicted", type: :quantitative, title: "Predicted")
      |> Vl.encode_field(:y, "residual", type: :quantitative, title: "Residual")

    layers = if opts[:reference_line], do: [zero_line(), scatter], else: [scatter]

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers)
  end

  # One flat set of rows over every model, so they share the axes and the eye
  # compares distance from the same reference line.
  defp rows(series, build) do
    Enum.flat_map(series, fn {label, y_true, y_pred} ->
      Internal.assert_paired!(y_true, y_pred, "y_true", "y_pred")
      points = build.(y_true, y_pred)

      if label, do: Enum.map(points, &Map.put(&1, "series", label)), else: points
    end)
  end

  defp scatter(values, opts) do
    layer =
      Vl.new()
      |> Vl.mark(:point, filled: true, opacity: 0.6, size: opts[:point_size], tooltip: true)

    case labels(values) do
      [] -> layer
      names -> encode_series(layer, names)
    end
  end

  defp labels(values),
    do: values |> Enum.map(& &1["series"]) |> Enum.uniq() |> Enum.reject(&is_nil/1)

  defp encode_series(layer, names) do
    Vl.encode_field(layer, :color, "series",
      type: :nominal,
      title: nil,
      scale: [domain: names, range: Theme.categorical(length(names))]
    )
  end

  defp diagonal([min, max]) do
    Vl.new()
    |> Vl.data_from_values([%{"x" => min, "y" => min}, %{"x" => max, "y" => max}])
    |> Vl.mark(:line, Theme.reference_mark())
    |> Vl.encode_field(:x, "x", type: :quantitative)
    |> Vl.encode_field(:y, "y", type: :quantitative)
  end

  defp zero_line do
    Vl.new()
    |> Vl.data_from_values([%{"zero" => 0}])
    |> Vl.mark(:rule, Theme.reference_mark())
    |> Vl.encode_field(:y, "zero", type: :quantitative)
  end

  defp shared_domain(values, fields) do
    all = Enum.flat_map(values, fn row -> Enum.map(fields, &Map.fetch!(row, &1)) end)
    {min, max} = Enum.min_max(all)
    pad = max((max - min) * 0.05, 1.0e-9)
    [min - pad, max + pad]
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
