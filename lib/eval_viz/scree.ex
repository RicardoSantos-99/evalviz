defmodule EvalViz.Scree do
  @moduledoc false

  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 cumulative: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Overlays the running total, which is the line you read to pick
                   how many components to keep.
                   """
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 500],
                 height: [type: :pos_integer, default: 350]
               )

  def schema, do: @opts_schema

  def plot(ratios, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    ratios = Nx.to_flat_list(ratios)

    values =
      ratios
      |> Enum.scan(0.0, &(&1 + &2))
      |> Enum.zip(ratios)
      |> Enum.with_index(1)
      |> Enum.map(fn {{cumulative, ratio}, component} ->
        %{
          "component" => component,
          "ratio" => ratio,
          "cumulative" => cumulative
        }
      end)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(opts))
    |> Vl.resolve(:scale, y: :shared)
  end

  defp layers(opts) do
    bars =
      Vl.new()
      |> Vl.mark(:bar, tooltip: true, color: Theme.primary())
      |> Vl.encode_field(:x, "component",
        type: :ordinal,
        title: "Component",
        axis: [label_angle: 0]
      )
      |> Vl.encode_field(:y, "ratio",
        type: :quantitative,
        title: "Explained variance",
        axis: [format: ".0%"]
      )

    if opts[:cumulative] do
      [bars, cumulative_layer()]
    else
      [bars]
    end
  end

  defp cumulative_layer do
    Vl.new()
    |> Vl.mark(:line, point: true, color: Theme.secondary(), tooltip: true)
    |> Vl.encode_field(:x, "component", type: :ordinal)
    |> Vl.encode_field(:y, "cumulative", type: :quantitative)
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
