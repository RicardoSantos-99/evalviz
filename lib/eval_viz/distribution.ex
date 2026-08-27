defmodule EvalViz.Distribution do
  @moduledoc false

  alias EvalViz.Internal
  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @histogram_schema NimbleOptions.new!(
                      bins: [type: :pos_integer, default: 30, doc: "Number of bins."],
                      zero_line: [
                        type: :boolean,
                        default: true,
                        doc: "Draws the zero a residual with no bias would centre on."
                      ],
                      value_title: [type: :string, default: "Residual"],
                      title: [type: :string, doc: "Chart title."],
                      width: [type: :pos_integer, default: 480],
                      height: [type: :pos_integer, default: 320]
                    )

  @qq_schema NimbleOptions.new!(
               reference_line: [
                 type: :boolean,
                 default: true,
                 doc: """
                 Draws the least-squares fit through the points, which is the
                 line they would lie on were the sample exactly normal.
                 """
               ],
               point_size: [type: :pos_integer, default: 45],
               sample_title: [type: :string, default: "Sample quantiles"],
               title: [type: :string, doc: "Chart title."],
               width: [type: :pos_integer, default: 440],
               height: [type: :pos_integer, default: 400]
             )

  def histogram_schema, do: @histogram_schema
  def qq_schema, do: @qq_schema

  def histogram(values, opts) do
    opts = NimbleOptions.validate!(opts, @histogram_schema)
    values = Nx.to_flat_list(values)

    span = Internal.bin_span(values, opts[:bins])

    rows =
      values
      |> Internal.bin_counts(span, opts[:bins])
      |> Enum.reject(fn {_lower, _upper, count} -> count == 0 end)
      |> Enum.map(fn {lower, upper, count} ->
        %{"lower" => lower, "upper" => upper, "zero" => 0, "count" => count}
      end)

    tallest = rows |> Enum.map(& &1["count"]) |> Enum.max(fn -> 1 end)

    # Both axes carry a quantity, so the bar needs its extent on each: x across
    # the bin, y up from zero. Given only x2 it draws as a dash at the count.
    bars =
      Vl.new()
      |> Vl.mark(:bar, tooltip: true, color: Theme.primary())
      |> Vl.encode_field(:x, "lower",
        type: :quantitative,
        title: opts[:value_title],
        scale: [zero: false]
      )
      |> Vl.encode_field(:x2, "upper")
      |> Vl.encode_field(:y, "zero",
        type: :quantitative,
        title: "Count",
        scale: [domain: [0, tallest * 1.05], nice: false]
      )
      |> Vl.encode_field(:y2, "count")

    layers = if opts[:zero_line], do: [bars, zero_rule()], else: [bars]

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(rows)
    |> Vl.layers(layers)
  end

  def qq(values, opts) do
    opts = NimbleOptions.validate!(opts, @qq_schema)

    sample = values |> Nx.to_flat_list() |> Enum.sort()
    count = length(sample)

    if count < 2 do
      raise ArgumentError, "expected at least two values to plot, got #{count}"
    end

    theoretical = Enum.map(plotting_positions(count), &Internal.normal_quantile/1)
    rows = Enum.zip_with(theoretical, sample, &%{"theoretical" => &1, "sample" => &2})

    points =
      Vl.new()
      |> Vl.mark(:point,
        filled: true,
        opacity: 0.7,
        size: opts[:point_size],
        color: Theme.primary(),
        tooltip: true
      )
      |> Vl.encode_field(:x, "theoretical",
        type: :quantitative,
        title: "Theoretical quantiles",
        scale: [zero: false]
      )
      |> Vl.encode_field(:y, "sample",
        type: :quantitative,
        title: opts[:sample_title],
        scale: [zero: false]
      )

    layers =
      if opts[:reference_line] do
        [fit_layer(theoretical, sample), points]
      else
        [points]
      end

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(rows)
    |> Vl.layers(layers)
  end

  # Filliben's estimates of the uniform order statistic medians, the positions
  # scipy's probplot places the ordered sample at.
  defp plotting_positions(1), do: [0.5]

  defp plotting_positions(count) do
    last = :math.pow(0.5, 1 / count)
    middle = for i <- 2..(count - 1)//1, do: (i - 0.3175) / (count + 0.365)

    [1 - last] ++ middle ++ [last]
  end

  defp fit_layer(theoretical, sample) do
    {slope, intercept} = least_squares(theoretical, sample)
    {min, max} = Enum.min_max(theoretical)

    Vl.new()
    |> Vl.data_from_values([
      %{"x" => min, "y" => slope * min + intercept},
      %{"x" => max, "y" => slope * max + intercept}
    ])
    |> Vl.mark(:line, Theme.reference_mark())
    |> Vl.encode_field(:x, "x", type: :quantitative)
    |> Vl.encode_field(:y, "y", type: :quantitative)
  end

  defp least_squares(xs, ys) do
    count = length(xs)
    mean_x = Enum.sum(xs) / count
    mean_y = Enum.sum(ys) / count

    covariance =
      Enum.zip_with(xs, ys, fn x, y -> (x - mean_x) * (y - mean_y) end) |> Enum.sum()

    variance = xs |> Enum.map(&((&1 - mean_x) * (&1 - mean_x))) |> Enum.sum()

    slope = if variance == 0, do: 0.0, else: covariance / variance
    {slope, mean_y - slope * mean_x}
  end

  defp zero_rule do
    Vl.new()
    |> Vl.data_from_values([%{"zero" => 0}])
    |> Vl.mark(:rule, Theme.reference_mark())
    |> Vl.encode_field(:x, "zero", type: :quantitative)
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
