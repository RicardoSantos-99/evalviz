defmodule Verdict.Calibration do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Multiclass
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 class_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: """
                   Names for each class when `y_prob` has one column per class.
                   Defaults to `0..num_classes - 1`.
                   """
                 ],
                 average: [
                   type: {:in, [:micro]},
                   doc: """
                   Set to `:micro` to add a curve pooling every (sample, class)
                   pair into one binary problem. There is no macro option: the
                   per-class curves land on different bins, so averaging them
                   would compare probabilities that were never comparable.
                   """
                 ],
                 sample_weights: Internal.weights_option(),
                 bins: [
                   type: :pos_integer,
                   default: 10,
                   doc: "Number of bins to group the predicted probabilities into."
                 ],
                 strategy: [
                   type: {:in, [:uniform, :quantile]},
                   default: :uniform,
                   doc: """
                   How to place the bin edges. `:uniform` splits the 0..1 range
                   evenly; `:quantile` gives every bin the same number of points,
                   which keeps the curve steady when predictions cluster.
                   """
                 ],
                 facet: [
                   type: :boolean,
                   default: false,
                   doc: "Draws a panel per curve instead of overlaying them."
                 ],
                 columns: [
                   type: :pos_integer,
                   default: 3,
                   doc: "Panels per row when faceting."
                 ],
                 perfect_line: [
                   type: :boolean,
                   default: true,
                   doc: "Draws the diagonal a perfectly calibrated model would trace."
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 450],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  def plot(series, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    curves = series |> Enum.flat_map(&expand(&1, opts)) |> Enum.map(&curve(&1, opts))
    multi? = length(curves) > 1
    values = Enum.flat_map(curves, & &1.points)

    if opts[:facet] and multi? do
      facet(values, opts)
    else
      Vl.new(vl_opts(opts))
      |> Vl.data_from_values(values)
      |> Vl.layers(layers(multi?, opts))
    end
  end

  # Each panel names its own curve, so the colour legend has nothing to add.
  defp facet(values, opts) do
    child =
      Vl.new(width: opts[:width], height: opts[:height])
      |> Vl.layers(layers(false, opts))

    # `columns` belongs to the outer spec, not to the facet definition, and the
    # header would otherwise print the field's own name above the panels.
    Vl.new([columns: opts[:columns]] ++ title(opts[:title]))
    |> Vl.data_from_values(values)
    |> Vl.facet([field: "series", type: :nominal, title: nil], child)
  end

  defp title(nil), do: []
  defp title(text), do: [title: text]

  defp expand(entry, opts) do
    Multiclass.per_class(entry, opts) ++ Multiclass.micro(entry, opts)
  end

  defp curve({label, y_true, y_prob}, opts) do
    Internal.assert_paired!(y_true, y_prob, "y_true", "y_prob")

    count = Nx.axis_size(y_true, 0)
    y_true = Nx.to_flat_list(y_true)
    y_prob = Nx.to_flat_list(y_prob)
    weights = Internal.weight_list(opts[:sample_weights], count)

    assert_binary!(y_true)
    assert_probabilities!(y_prob)

    bin_edges = edges(y_prob, opts[:bins], opts[:strategy])

    # Every mean here is weighted, so a weight of two counts a sample twice
    # exactly as duplicating its row would.
    points =
      [y_prob, y_true, weights]
      |> Enum.zip()
      |> Enum.group_by(fn {prob, _, _} -> bin_of(prob, bin_edges) end)
      |> Enum.sort_by(fn {bin, _} -> bin end)
      |> Enum.map(fn {_bin, triples} ->
        total = triples |> Enum.map(&elem(&1, 2)) |> Enum.sum()

        %{
          "predicted" => weighted_mean(triples, 0, total),
          "observed" => weighted_mean(triples, 1, total),
          "count" => total,
          "series" => legend(label, y_true, y_prob, weights)
        }
      end)

    %{points: points}
  end

  # Interior edges only: a value lands in the bin holding every edge below it,
  # which is what numpy's searchsorted does inside sklearn. A probability
  # sitting exactly on an edge is decided by the input's precision, since an
  # f32 0.2 is a hair above the f64 edge and falls to the bin above.
  defp bin_of(value, edges), do: Enum.count(edges, &(&1 < value))

  defp edges(_y_prob, bins, :uniform) do
    for i <- 1..(bins - 1)//1, do: i / bins
  end

  defp edges(y_prob, bins, :quantile) do
    sorted = Enum.sort(y_prob)
    for i <- 1..(bins - 1)//1, do: percentile(sorted, i / bins)
  end

  # Linear interpolation between the two neighbouring ranks, matching numpy's
  # default percentile method.
  defp percentile(sorted, q) do
    n = length(sorted)
    index = q * (n - 1)
    lower = trunc(Float.floor(index))
    upper = min(lower + 1, n - 1)
    weight = index - lower

    Enum.at(sorted, lower) * (1 - weight) + Enum.at(sorted, upper) * weight
  end

  defp weighted_mean(triples, index, total) do
    triples
    |> Enum.map(fn triple -> elem(triple, index) * elem(triple, 2) end)
    |> Enum.sum()
    |> Kernel./(total)
  end

  defp legend(nil, _y_true, _y_prob, _weights), do: nil

  defp legend(label, y_true, y_prob, weights) do
    brier =
      [y_prob, y_true, weights]
      |> Enum.zip()
      |> Enum.map(fn {prob, actual, weight} -> (prob - actual) * (prob - actual) * weight end)
      |> Enum.sum()
      |> Kernel./(Enum.sum(weights))
      |> Float.round(4)

    "#{label} (Brier = #{brier})"
  end

  defp layers(multi?, opts) do
    curve =
      Vl.new()
      |> Vl.mark(:line, point: true, tooltip: true)
      |> Vl.encode_field(:x, "predicted",
        type: :quantitative,
        title: "Mean predicted probability",
        scale: [domain: [0, 1], nice: false]
      )
      |> Vl.encode_field(:y, "observed",
        type: :quantitative,
        title: "Observed frequency",
        scale: [domain: [0, 1], nice: false]
      )

    curve =
      if multi? do
        Vl.encode_field(curve, :color, "series", type: :nominal, title: nil)
      else
        curve
      end

    if opts[:perfect_line], do: [perfect_layer(), curve], else: [curve]
  end

  defp perfect_layer do
    Vl.new()
    |> Vl.data_from_values([%{"x" => 0, "y" => 0}, %{"x" => 1, "y" => 1}])
    |> Vl.mark(:line, Theme.reference_mark())
    |> Vl.encode_field(:x, "x", type: :quantitative)
    |> Vl.encode_field(:y, "y", type: :quantitative)
  end

  defp assert_binary!(y_true) do
    distinct = y_true |> Enum.uniq() |> Enum.sort()

    unless distinct == [0] or distinct == [1] or distinct == [0, 1] do
      raise ArgumentError,
            "expected y_true to hold only 0 and 1, got #{inspect(distinct)}"
    end

    :ok
  end

  defp assert_probabilities!(y_prob) do
    unless Enum.all?(y_prob, &(&1 >= 0 and &1 <= 1)) do
      raise ArgumentError,
            "expected y_prob to hold probabilities between 0 and 1. " <>
              "Calibration compares predicted probability against observed " <>
              "frequency, so raw scores have to be converted first."
    end

    :ok
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
