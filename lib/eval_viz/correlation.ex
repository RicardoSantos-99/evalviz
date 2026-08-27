defmodule EvalViz.Correlation do
  @moduledoc false

  alias EvalViz.Internal
  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 feature_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "One name per column. Defaults to the index."
                 ],
                 values: [
                   type: :boolean,
                   default: true,
                   doc: "Writes each coefficient in its cell."
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 420],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  def plot(x, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if Nx.rank(x) != 2 do
      raise ArgumentError,
            "expected the data to be a rank-2 tensor of {num_samples, num_features}, " <>
              "got rank #{Nx.rank(x)}"
    end

    matrix = Scholar.Stats.correlation_matrix(x)
    names = Internal.class_labels(opts[:feature_names], Nx.axis_size(x, 1))
    values = to_values(Nx.to_list(matrix), names)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(names, opts))
  end

  defp to_values(rows, names) do
    for {row, i} <- Enum.with_index(rows),
        {value, j} <- Enum.with_index(row) do
      %{
        "row" => Enum.at(names, i),
        "column" => Enum.at(names, j),
        "value" => value,
        "label" => value |> Internal.round_to(2) |> to_string()
      }
    end
  end

  defp layers(names, opts) do
    # A correlation already spans a fixed -1..1, so the scale is pinned rather
    # than fitted: otherwise a matrix of weak correlations would colour like a
    # matrix of strong ones.
    heatmap =
      Vl.new()
      |> Vl.mark(:rect, tooltip: true)
      |> Vl.encode_field(:x, "column",
        type: :nominal,
        sort: names,
        title: nil,
        axis: [label_angle: 0]
      )
      |> Vl.encode_field(:y, "row", type: :nominal, sort: names, title: nil)
      |> Vl.encode_field(:color, "value",
        type: :quantitative,
        title: "Correlation",
        scale: [scheme: Theme.diverging_scheme(), domain: [-1, 1]]
      )

    if opts[:values], do: [heatmap, text_layer(names)], else: [heatmap]
  end

  defp text_layer(names) do
    Vl.new()
    |> Vl.mark(:text, font_size: 11)
    |> Vl.encode_field(:x, "column", type: :nominal, sort: names)
    |> Vl.encode_field(:y, "row", type: :nominal, sort: names)
    |> Vl.encode_field(:text, "label", type: :nominal)
    # both ends of a diverging scheme go dark, so the test is on magnitude
    |> Vl.encode(:color,
      condition: [test: "abs(datum.value) > 0.6", value: "white"],
      value: "black"
    )
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
