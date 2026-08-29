defmodule Verdict.FeatureDistribution do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 bins: [
                   type: :pos_integer,
                   default: 20,
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
                   doc: "Names for the classes in `labels`, indexed from zero."
                 ],
                 feature_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "Names for the columns of `x`. Defaults to the column index."
                 ],
                 columns: [
                   type: :pos_integer,
                   default: 3,
                   doc: "Panels per row before wrapping."
                 ],
                 opacity: [type: :float, default: 0.65],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 220],
                 height: [type: :pos_integer, default: 160]
               )

  def schema, do: @opts_schema

  def plot(x, labels, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    assert_matrix!(x)

    num_samples = Nx.axis_size(x, 0)
    num_features = Nx.axis_size(x, 1)
    assert_paired!(labels, num_samples)

    label_values = Nx.to_flat_list(labels)
    weights = Internal.weight_list(opts[:sample_weights], num_samples)
    feature_names = feature_names(opts[:feature_names], num_features)

    panels =
      for column <- 0..(num_features - 1) do
        values = x[[.., column]] |> Nx.to_flat_list()
        {bars, ordered} = histogram(values, label_values, weights, opts)
        panel(bars, ordered, Enum.at(feature_names, column), opts)
      end

    Vl.new(vl_opts(opts))
    |> Vl.concat(panels, :wrappable)
    |> Vl.resolve(:legend, color: :shared)
  end

  defp histogram(values, label_values, weights, opts) do
    Internal.class_histogram(
      values,
      label_values,
      weights,
      opts[:bins],
      opts[:normalize],
      opts[:class_names]
    )
  end

  # Both axes carry a quantity, so the bar has to be told its extent on each: x
  # across the bin, y up from zero. Every bar starting at zero is also what
  # overlays the classes instead of stacking them, which is what the plot is
  # read for.
  defp panel(bars, ordered, name, opts) do
    Vl.new(width: opts[:width], height: opts[:height], title: name)
    |> Vl.data_from_values(bars)
    |> Vl.mark(:bar, opacity: opts[:opacity], tooltip: true)
    |> Vl.encode_field(:x, "lower", type: :quantitative, title: nil, scale: [zero: false])
    |> Vl.encode_field(:x2, "upper")
    |> Vl.encode_field(:y, "zero",
      type: :quantitative,
      title: y_title(opts),
      scale: [domain: [0, headroom(bars)], nice: false]
    )
    |> Vl.encode_field(:y2, "value")
    |> Vl.encode_field(:color, "class",
      type: :nominal,
      title: "True class",
      scale: [domain: ordered, range: Theme.categorical(length(ordered))]
    )
  end

  defp y_title(opts), do: if(opts[:normalize] == :class, do: "Share of class", else: "Count")

  defp headroom(bars),
    do: bars |> Enum.map(& &1["value"]) |> Enum.max(fn -> 1 end) |> Kernel.*(1.05)

  defp feature_names(nil, n), do: Enum.map(0..(n - 1), &Integer.to_string/1)

  defp feature_names(names, n) when length(names) == n do
    Enum.map(names, &to_string/1)
  end

  defp feature_names(names, n) do
    raise ArgumentError,
          "expected :feature_names to have one entry per column of x, " <>
            "got #{length(names)} names for #{n} columns"
  end

  defp assert_matrix!(x) do
    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected x to be a rank-2 tensor of {num_samples, num_features}, " <>
              "got rank #{Nx.rank(x)}"
    end
  end

  defp assert_paired!(labels, num_samples) do
    if Nx.axis_size(labels, 0) != num_samples do
      raise ArgumentError,
            "expected labels to have one entry per row of x, " <>
              "got #{Nx.axis_size(labels, 0)} for #{num_samples} rows"
    end
  end

  defp vl_opts(opts) do
    base = [columns: opts[:columns]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
