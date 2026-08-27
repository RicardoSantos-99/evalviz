defmodule EvalViz.Regression do
  @moduledoc false

  alias EvalViz.Internal
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

  def predicted_vs_actual(y_true, y_pred, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    Internal.assert_paired!(y_true, y_pred, "y_true", "y_pred")

    values = Internal.points(y_true, y_pred, "actual", "predicted")

    # Both axes carry the same quantity, so they share one range: otherwise the
    # diagonal stops meaning "perfect" and the eye reads the spread wrong.
    domain = shared_domain(values, ["actual", "predicted"])

    scatter =
      Vl.new()
      |> Vl.mark(:point, filled: true, opacity: 0.6, size: opts[:point_size], tooltip: true)
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

    layers =
      if opts[:reference_line] do
        [diagonal(domain), scatter]
      else
        [scatter]
      end

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers)
  end

  def residuals(y_true, y_pred, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    Internal.assert_paired!(y_true, y_pred, "y_true", "y_pred")

    residuals = Nx.subtract(y_pred, y_true)
    values = Internal.points(y_pred, residuals, "predicted", "residual")

    scatter =
      Vl.new()
      |> Vl.mark(:point, filled: true, opacity: 0.6, size: opts[:point_size], tooltip: true)
      |> Vl.encode_field(:x, "predicted", type: :quantitative, title: "Predicted")
      |> Vl.encode_field(:y, "residual", type: :quantitative, title: "Residual")

    layers =
      if opts[:reference_line] do
        [zero_line(), scatter]
      else
        [scatter]
      end

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers)
  end

  defp diagonal([min, max]) do
    Vl.new()
    |> Vl.data_from_values([%{"x" => min, "y" => min}, %{"x" => max, "y" => max}])
    |> Vl.mark(:line, stroke_dash: [4, 4], color: "#999", size: 1)
    |> Vl.encode_field(:x, "x", type: :quantitative)
    |> Vl.encode_field(:y, "y", type: :quantitative)
  end

  defp zero_line do
    Vl.new()
    |> Vl.data_from_values([%{"zero" => 0}])
    |> Vl.mark(:rule, stroke_dash: [4, 4], color: "#999", size: 1)
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
