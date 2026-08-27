defmodule EvalViz.Biplot do
  @moduledoc false

  alias EvalViz.Internal
  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 feature_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "One name per feature, labelling the arrows. Defaults to the index."
                 ],
                 labels: [
                   type: :any,
                   doc: "Rank-1 tensor colouring the points, such as the true class."
                 ],
                 label_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "Names for the values in `:labels`, indexed from zero."
                 ],
                 components: [
                   type: {:list, :non_neg_integer},
                   default: [0, 1],
                   doc: "Which two components to draw, as `[x, y]`."
                 ],
                 arrow_scale: [
                   type: :float,
                   default: 0.9,
                   doc: """
                   How far the longest arrow reaches, as a share of the widest
                   point. Loadings and scores have unrelated units, so an arrow's
                   length is only meaningful next to the other arrows.
                   """
                 ],
                 opacity: [type: :float, default: 0.6],
                 point_size: [type: :pos_integer, default: 55],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 460],
                 height: [type: :pos_integer, default: 440]
               )

  def schema, do: @opts_schema

  def plot(projection, loadings, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    [a, b] = components(opts[:components])

    points = points(projection, a, b, opts)
    arrows = arrows(loadings, a, b, points, opts)
    domain = domain(points, arrows)

    Vl.new(vl_opts(opts))
    |> Vl.layers([
      point_layer(points, domain, a, b, opts),
      arrow_layer(arrows, domain),
      arrow_label_layer(arrows, domain)
    ])
  end

  defp points(projection, a, b, opts) do
    xs = column(projection, a, "the projection")
    ys = column(projection, b, "the projection")
    labels = label_values(opts[:labels], length(xs), opts[:label_names])

    [xs, ys, labels]
    |> Enum.zip()
    |> Enum.map(fn {x, y, label} -> %{"x" => x, "y" => y, "label" => label} end)
  end

  defp label_values(nil, count, _names), do: List.duplicate(nil, count)

  defp label_values(labels, count, names) do
    values = Nx.to_flat_list(labels)

    if length(values) != count do
      raise ArgumentError,
            "expected :labels to have one entry per row of the projection, " <>
              "got #{length(values)} for #{count} rows"
    end

    Enum.map(values, fn label ->
      if names, do: names |> Enum.at(trunc(label), label) |> to_string(), else: to_string(label)
    end)
  end

  # Loadings are unit-length and scores are not, so the arrows are stretched to
  # sit against the cloud. Only their directions and relative lengths mean
  # anything.
  defp arrows(loadings, a, b, points, opts) do
    xs = component_row(loadings, a)
    ys = component_row(loadings, b)
    names = Internal.class_labels(opts[:feature_names], length(xs))

    scale = arrow_scale(xs, ys, points, opts[:arrow_scale])

    [xs, ys, names]
    |> Enum.zip()
    |> Enum.map(fn {x, y, name} ->
      %{
        "zero" => 0,
        "x" => x * scale,
        "y" => y * scale,
        "text_x" => x * scale * 1.1,
        "text_y" => y * scale * 1.1,
        "feature" => name
      }
    end)
  end

  defp arrow_scale(xs, ys, points, share) do
    longest_arrow = xs |> Enum.zip(ys) |> Enum.map(&radius/1) |> Enum.max()
    widest_point = points |> Enum.map(&{&1["x"], &1["y"]}) |> Enum.map(&radius/1) |> Enum.max()

    if longest_arrow == 0.0, do: 1.0, else: share * widest_point / longest_arrow
  end

  defp radius({x, y}), do: :math.sqrt(x * x + y * y)

  # Loadings run either side of zero, so the arrows only read correctly when the
  # origin sits in the middle and both axes share a scale.
  defp domain(points, arrows) do
    coordinates =
      Enum.flat_map(points, &[&1["x"], &1["y"]]) ++
        Enum.flat_map(arrows, &[&1["text_x"], &1["text_y"]])

    extent = coordinates |> Enum.map(&abs/1) |> Enum.max() |> Kernel.*(1.05)
    [-extent, extent]
  end

  defp point_layer(points, domain, a, b, opts) do
    layer =
      Vl.new()
      |> Vl.data_from_values(points)
      |> Vl.mark(:point,
        filled: true,
        size: opts[:point_size],
        opacity: opts[:opacity],
        tooltip: true
      )
      |> Vl.encode_field(:x, "x",
        type: :quantitative,
        title: "Component #{a}",
        scale: [domain: domain, nice: false]
      )
      |> Vl.encode_field(:y, "y",
        type: :quantitative,
        title: "Component #{b}",
        scale: [domain: domain, nice: false]
      )

    if opts[:labels], do: encode_labels(layer, points), else: layer
  end

  defp encode_labels(layer, points) do
    names = points |> Enum.map(& &1["label"]) |> Enum.uniq()

    Vl.encode_field(layer, :color, "label",
      type: :nominal,
      title: nil,
      scale: [domain: names, range: Theme.categorical(length(names))]
    )
  end

  defp arrow_layer(arrows, domain) do
    Vl.new()
    |> Vl.data_from_values(arrows)
    |> Vl.mark(:rule, color: Theme.secondary(), size: 1.5, opacity: 0.9)
    |> Vl.encode_field(:x, "zero", type: :quantitative, scale: [domain: domain, nice: false])
    |> Vl.encode_field(:y, "zero", type: :quantitative, scale: [domain: domain, nice: false])
    |> Vl.encode_field(:x2, "x")
    |> Vl.encode_field(:y2, "y")
  end

  defp arrow_label_layer(arrows, domain) do
    Vl.new()
    |> Vl.data_from_values(arrows)
    |> Vl.mark(:text, color: Theme.secondary(), font_size: 11, font_weight: "bold")
    |> Vl.encode_field(:x, "text_x", type: :quantitative, scale: [domain: domain, nice: false])
    |> Vl.encode_field(:y, "text_y", type: :quantitative, scale: [domain: domain, nice: false])
    |> Vl.encode_field(:text, "feature", type: :nominal)
  end

  defp column(tensor, index, what) do
    assert_rank_2!(tensor, what)
    available = Nx.axis_size(tensor, 1)

    if index >= available do
      raise ArgumentError,
            "expected :components to index #{what}'s #{available} columns, got #{index}"
    end

    Internal.to_list(tensor[[.., index]])
  end

  # Loadings arrive as {num_components, num_features}, so a component is a row.
  defp component_row(loadings, index) do
    assert_rank_2!(loadings, "the loadings")
    available = Nx.axis_size(loadings, 0)

    if index >= available do
      raise ArgumentError,
            "expected :components to index the loadings' #{available} rows, got #{index}"
    end

    Internal.to_list(loadings[index])
  end

  defp assert_rank_2!(tensor, what) do
    if Nx.rank(tensor) != 2 do
      raise ArgumentError,
            "expected #{what} to be a rank-2 tensor, got rank #{Nx.rank(tensor)}"
    end

    :ok
  end

  defp components([a, b]), do: [a, b]

  defp components(other) do
    raise ArgumentError, "expected :components to name two columns, got #{inspect(other)}"
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
