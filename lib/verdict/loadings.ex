defmodule Verdict.Loadings do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 feature_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "One name per feature. Defaults to the index."
                 ],
                 component_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "One name per component. Defaults to `Component 0`, `Component 1`, ..."
                 ],
                 values: [
                   type: :boolean,
                   default: true,
                   doc: "Writes each loading in its cell."
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 480],
                 height: [type: :pos_integer, default: 260]
               )

  def schema, do: @opts_schema

  def plot(components, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if Nx.rank(components) != 2 do
      raise ArgumentError,
            "expected the loadings to be a rank-2 tensor of " <>
              "{num_components, num_features}, got rank #{Nx.rank(components)}"
    end

    rows = Nx.to_list(components)
    features = Internal.class_labels(opts[:feature_names], Nx.axis_size(components, 1))
    names = component_names(opts[:component_names], length(rows))

    values = to_values(rows, features, names)
    extent = extent(rows)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(features, names, extent, opts))
  end

  defp to_values(rows, features, names) do
    for {row, component} <- Enum.with_index(rows),
        {value, feature} <- Enum.with_index(row) do
      %{
        "component" => Enum.at(names, component),
        "feature" => Enum.at(features, feature),
        "value" => value,
        "label" => value |> Internal.round_to(2) |> to_string()
      }
    end
  end

  # A loading's sign says which way the feature pushes, so the scale has to be
  # centred on zero rather than running from the smallest value up.
  defp extent(rows) do
    rows |> List.flatten() |> Enum.map(&abs/1) |> Enum.max() |> max(1.0e-9)
  end

  defp layers(features, names, extent, opts) do
    heatmap =
      Vl.new()
      |> Vl.mark(:rect, tooltip: true)
      |> Vl.encode_field(:x, "feature",
        type: :nominal,
        sort: features,
        title: "Feature",
        axis: [label_angle: 0]
      )
      |> Vl.encode_field(:y, "component", type: :nominal, sort: names, title: nil)
      |> Vl.encode_field(:color, "value",
        type: :quantitative,
        title: "Loading",
        scale: [scheme: Theme.diverging_scheme(), domain: [-extent, extent]]
      )

    if opts[:values], do: [heatmap, text_layer(features, names, extent)], else: [heatmap]
  end

  defp text_layer(features, names, extent) do
    Vl.new()
    |> Vl.mark(:text, font_size: 12)
    |> Vl.encode_field(:x, "feature", type: :nominal, sort: features)
    |> Vl.encode_field(:y, "component", type: :nominal, sort: names)
    |> Vl.encode_field(:text, "label", type: :nominal)
    # both ends of a diverging scheme go dark, so the test is on magnitude
    |> Vl.encode(:color,
      condition: [test: "abs(datum.value) > #{extent * 0.6}", value: "white"],
      value: "black"
    )
  end

  defp component_names(nil, n), do: Enum.map(0..(n - 1), &"Component #{&1}")

  defp component_names(names, n) when length(names) == n, do: Enum.map(names, &to_string/1)

  defp component_names(names, n) do
    raise ArgumentError,
          "expected :component_names to have one entry per component, " <>
            "got #{length(names)} names for #{n} components"
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
