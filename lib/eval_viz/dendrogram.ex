defmodule EvalViz.Dendrogram do
  @moduledoc false

  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 labels: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "Leaf labels, one per point, in data order. Defaults to the index."
                 ],
                 color_threshold: [
                   type: {:or, [:float, :integer]},
                   doc: """
                   Colours each subtree that merges below this height as its own
                   cluster, leaving the links above it grey. This is the height
                   you would cut the tree at to read clusters off the plot.
                   """
                 ],
                 title: [type: :string, doc: "Chart title."],
                 height_title: [
                   type: :string,
                   default: "Dissimilarity",
                   doc: "Label for the height axis."
                 ],
                 width: [type: :pos_integer, default: 600],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  def plot(clades, heights, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    clades = Nx.to_list(clades)
    heights = Nx.to_flat_list(heights)
    num_merges = length(clades)
    n = num_merges + 1

    validate!(clades, heights, n, opts[:labels])

    order = leaf_order(clades, n)
    labels = leaf_labels(opts[:labels], order)
    colors = cluster_colors(clades, heights, n, opts[:color_threshold])

    values = links(clades, heights, n, order, colors)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.mark(:line, stroke_width: 1.5, tooltip: true)
    |> Vl.encode_field(:x, "x",
      type: :quantitative,
      title: nil,
      axis: [
        values: Enum.to_list(0..(n - 1)),
        label_expr: label_expr(labels),
        grid: false,
        label_angle: leaf_label_angle(labels)
      ],
      scale: [domain: [-0.6, n - 0.4], nice: false]
    )
    |> Vl.encode_field(:y, "y",
      type: :quantitative,
      title: opts[:height_title],
      scale: [domain_min: 0, nice: true]
    )
    # Each bracket is its own line: without splitting on the link, Vega-Lite
    # joins every point into a single stroke.
    |> Vl.encode_field(:detail, "link", type: :nominal)
    |> encode_color(colors)
  end

  # Leaves are laid out in the order a depth-first walk of the merge tree
  # reaches them, which is what keeps the brackets from crossing.
  defp leaf_order(clades, n) do
    clades
    |> collect_leaves(2 * n - 2, n)
    |> Enum.reverse()
  end

  defp collect_leaves(clades, id, n, acc \\ [])
  defp collect_leaves(_clades, id, n, acc) when id < n, do: [id | acc]

  defp collect_leaves(clades, id, n, acc) do
    [left, right] = Enum.at(clades, id - n)

    acc = collect_leaves(clades, left, n, acc)
    collect_leaves(clades, right, n, acc)
  end

  defp links(clades, heights, n, order, colors) do
    x_of_leaf = order |> Enum.with_index() |> Map.new()

    initial =
      Map.new(x_of_leaf, fn {leaf, x} -> {leaf, {x / 1, 0.0}} end)

    {_positions, values} =
      clades
      |> Enum.zip(heights)
      |> Enum.with_index()
      |> Enum.reduce({initial, []}, fn {{[left, right], height}, k}, {positions, acc} ->
        {x_left, y_left} = Map.fetch!(positions, left)
        {x_right, y_right} = Map.fetch!(positions, right)

        # the classic bracket: up from one child, across, back down to the other
        bracket =
          [{x_left, y_left}, {x_left, height}, {x_right, height}, {x_right, y_right}]
          |> Enum.with_index()
          |> Enum.map(fn {{x, y}, order_in_link} ->
            %{
              "x" => x,
              "y" => y * 1.0,
              "link" => k,
              "order" => order_in_link,
              "cluster" => Map.get(colors, k, above_cut())
            }
          end)

        parent = {(x_left + x_right) / 2, height * 1.0}
        {Map.put(positions, k + n, parent), acc ++ bracket}
      end)

    values
  end

  # A cluster is the largest subtree whose merges all happen below the cut, so
  # the roots are the clades under the threshold whose parent is above it.
  defp cluster_colors(_clades, _heights, _n, nil), do: %{}

  defp cluster_colors(clades, heights, n, threshold) do
    parents =
      clades
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {[left, right], k}, acc ->
        acc |> Map.put(left, k + n) |> Map.put(right, k + n)
      end)

    height_of = heights |> Enum.with_index() |> Map.new(fn {h, k} -> {k + n, h} end)

    roots =
      heights
      |> Enum.with_index()
      |> Enum.filter(fn {height, k} ->
        parent = Map.get(parents, k + n)
        height < threshold and (parent == nil or Map.fetch!(height_of, parent) >= threshold)
      end)
      |> Enum.map(fn {_height, k} -> k end)

    roots
    |> Enum.with_index(1)
    |> Enum.reduce(%{}, fn {root, cluster}, acc ->
      label = "Cluster #{cluster}"

      clades
      |> descendant_merges(root, n)
      |> Enum.reduce(acc, &Map.put(&2, &1, label))
    end)
  end

  defp descendant_merges(clades, k, n) do
    [left, right] = Enum.at(clades, k)

    [k] ++
      if(left >= n, do: descendant_merges(clades, left - n, n), else: []) ++
      if(right >= n, do: descendant_merges(clades, right - n, n), else: [])
  end

  defp above_cut, do: "Above cut"

  defp encode_color(plot, colors) when map_size(colors) == 0 do
    Vl.encode(plot, :color, value: Theme.primary())
  end

  defp encode_color(plot, colors) do
    names = colors |> Map.values() |> Enum.uniq() |> Enum.sort_by(&cluster_number/1)

    # The links above the cut are not a cluster, so they get a neutral grey
    # rather than the next colour in the palette.
    domain = names ++ [above_cut()]
    range = Theme.categorical(length(names)) ++ [Theme.muted()]

    Vl.encode_field(plot, :color, "cluster",
      type: :nominal,
      title: nil,
      scale: [domain: domain, range: range]
    )
  end

  defp cluster_number("Cluster " <> number), do: String.to_integer(number)
  defp cluster_number(_), do: 0

  defp leaf_labels(nil, order), do: Enum.map(order, &Integer.to_string/1)

  defp leaf_labels(labels, order) do
    Enum.map(order, fn leaf -> labels |> Enum.at(leaf) |> to_string() end)
  end

  # Vega-Lite has no way to map tick values to labels directly, so the mapping
  # goes in as an expression indexing a literal array.
  defp label_expr(labels) do
    array = labels |> Enum.map(&"'#{escape(&1)}'") |> Enum.join(",")
    "[#{array}][datum.value]"
  end

  defp escape(label), do: String.replace(label, "'", "\\'")

  defp leaf_label_angle(labels) do
    if Enum.any?(labels, &(String.length(&1) > 3)), do: -45, else: 0
  end

  defp validate!(clades, heights, n, labels) do
    if length(heights) != length(clades) do
      raise ArgumentError,
            "expected one height per merge, got #{length(heights)} heights " <>
              "for #{length(clades)} merges"
    end

    if n < 3 do
      raise ArgumentError, "a dendrogram needs at least 3 points, got #{n}"
    end

    if labels && length(labels) != n do
      raise ArgumentError,
            "expected :labels to have one entry per point, " <>
              "got #{length(labels)} labels for #{n} points"
    end

    :ok
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
