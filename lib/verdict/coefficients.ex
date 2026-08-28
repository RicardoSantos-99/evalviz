defmodule Verdict.Coefficients do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 feature_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "One name per feature. Defaults to the index."
                 ],
                 class_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "One name per class, when the model fits a coefficient per class."
                 ],
                 sort: [
                   type: {:in, [:magnitude, :value, :none]},
                   default: :magnitude,
                   doc: """
                   `:magnitude` puts the features that move the prediction most
                   at the top, whichever way they push. `:value` orders by the
                   signed coefficient, `:none` keeps the feature order.
                   """
                 ],
                 top: [
                   type: :pos_integer,
                   doc:
                     "Keep only this many features, by magnitude. Everything is kept by default."
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 460],
                 height: [type: :pos_integer, default: 360]
               )

  def schema, do: @opts_schema

  def plot(coefficients, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    {rows, classes} = rows(coefficients, opts)
    ordered = order(rows, opts)
    kept = Enum.filter(rows, &(&1["feature"] in ordered))

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(kept)
    |> Vl.layers(layers(ordered, classes))
  end

  defp rows(coefficients, opts) do
    case Nx.rank(coefficients) do
      1 ->
        names = Internal.class_labels(opts[:feature_names], Nx.axis_size(coefficients, 0))

        rows =
          coefficients
          |> Nx.to_flat_list()
          |> Enum.zip(names)
          |> Enum.map(fn {value, name} -> %{"feature" => name, "value" => value} end)

        {rows, nil}

      2 ->
        per_class(coefficients, opts)

      rank ->
        raise ArgumentError,
              "expected the coefficients to be rank 1 or 2, got rank #{rank}"
    end
  end

  # A classifier fits one coefficient per feature per class, arriving as
  # {num_features, num_classes}.
  defp per_class(coefficients, opts) do
    features = Internal.class_labels(opts[:feature_names], Nx.axis_size(coefficients, 0))
    classes = class_names(opts[:class_names], Nx.axis_size(coefficients, 1))

    rows =
      for {row, feature} <- Enum.with_index(Nx.to_list(coefficients)),
          {value, class} <- Enum.with_index(row) do
        %{
          "feature" => Enum.at(features, feature),
          "value" => value,
          "class" => Enum.at(classes, class)
        }
      end

    {rows, classes}
  end

  defp class_names(nil, n), do: Enum.map(0..(n - 1), &"Class #{&1}")

  defp class_names(names, n) when length(names) == n, do: Enum.map(names, &to_string/1)

  defp class_names(names, n) do
    raise ArgumentError,
          "expected :class_names to have one entry per class, " <>
            "got #{length(names)} names for #{n} classes"
  end

  # A feature with a coefficient per class is ranked by its strongest one.
  defp order(rows, opts) do
    by_feature =
      rows
      |> Enum.group_by(& &1["feature"], & &1["value"])
      |> Enum.map(fn {feature, values} ->
        {feature, Enum.max_by(values, &abs/1)}
      end)

    sorted =
      case opts[:sort] do
        :magnitude -> Enum.sort_by(by_feature, fn {_, value} -> -abs(value) end)
        :value -> Enum.sort_by(by_feature, fn {_, value} -> -value end)
        :none -> by_feature
      end

    names = Enum.map(sorted, &elem(&1, 0))
    if opts[:top], do: Enum.take(names, opts[:top]), else: names
  end

  defp layers(ordered, classes) do
    bars =
      Vl.new()
      |> Vl.mark(:bar, tooltip: true)
      |> Vl.encode_field(:y, "feature", type: :nominal, sort: ordered, title: nil)
      |> Vl.encode_field(:x, "value", type: :quantitative, title: "Coefficient")
      |> encode_class(classes)

    [bars, zero_layer()]
  end

  defp encode_class(layer, nil), do: Vl.encode(layer, :color, value: Theme.primary())

  defp encode_class(layer, classes) do
    layer
    |> Vl.encode_field(:color, "class",
      type: :nominal,
      title: nil,
      scale: [domain: classes, range: Theme.categorical(length(classes))]
    )
    |> Vl.encode_field(:y_offset, "class", type: :nominal)
  end

  # Coefficients are signed, so the bars need the line they are measured from.
  defp zero_layer do
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
