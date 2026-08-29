defmodule Verdict.ScoreDistribution do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 bins: [
                   type: :pos_integer,
                   default: 30,
                   doc: "Number of bins, shared by every class so the bars line up."
                 ],
                 sample_weights: Internal.weights_option(),
                 normalize: [
                   type: {:in, [:class, :none]},
                   default: :class,
                   doc: """
                   `:class` makes each class's bars sum to one, which is what
                   keeps a rare class visible next to a common one. `:none`
                   shows raw counts.
                   """
                 ],
                 class_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "Names for the classes in `y_true`, indexed from zero."
                 ],
                 threshold: [
                   type: :float,
                   doc: "Draws a rule at this score, to see what a cut-off would separate."
                 ],
                 opacity: [type: :float, default: 0.65],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 500],
                 height: [type: :pos_integer, default: 340]
               )

  def schema, do: @opts_schema

  def plot(y_true, y_score, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    Internal.assert_paired!(y_true, y_score, "y_true", "y_score")

    weights = Internal.weight_list(opts[:sample_weights], Nx.axis_size(y_true, 0))

    {values, ordered} =
      Internal.class_histogram(
        Nx.to_flat_list(y_score),
        Nx.to_flat_list(y_true),
        weights,
        opts[:bins],
        opts[:normalize],
        opts[:class_names]
      )

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(ordered, values, opts))
  end

  defp layers(ordered, values, opts) do
    # Both axes carry a quantity, so the bar has to be told its extent on each:
    # x across the bin, y up from zero. Given only x2 it would draw as a dash
    # floating at the count. Every bar starting at zero is also what overlays
    # the classes instead of stacking them, which is what the plot is read for.
    bars =
      Vl.new()
      |> Vl.mark(:bar, opacity: opts[:opacity], tooltip: true)
      |> Vl.encode_field(:x, "lower", type: :quantitative, title: "Score", scale: [zero: false])
      |> Vl.encode_field(:x2, "upper")
      |> Vl.encode_field(:y, "zero",
        type: :quantitative,
        title: y_title(opts),
        scale: [domain: [0, headroom(values)], nice: false]
      )
      |> Vl.encode_field(:y2, "value")
      |> Vl.encode_field(:color, "class",
        type: :nominal,
        title: "True class",
        scale: [domain: ordered, range: Theme.categorical(length(ordered))]
      )

    case opts[:threshold] do
      nil -> [bars]
      threshold -> [bars, threshold_layer(threshold)]
    end
  end

  defp threshold_layer(threshold) do
    Vl.new()
    |> Vl.data_from_values([%{"threshold" => threshold}])
    |> Vl.mark(:rule, Theme.reference_mark())
    |> Vl.encode_field(:x, "threshold", type: :quantitative)
  end

  defp y_title(opts) do
    if opts[:normalize] == :class, do: "Share of class", else: "Count"
  end

  defp headroom(values) do
    values |> Enum.map(& &1["value"]) |> Enum.max(fn -> 1 end) |> Kernel.*(1.05)
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
