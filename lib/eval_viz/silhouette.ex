defmodule EvalViz.Silhouette do
  @moduledoc false

  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 num_clusters: [
                   type: :pos_integer,
                   required: true,
                   doc: "Number of clusters in `labels`."
                 ],
                 cluster_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "Names for each cluster. Defaults to the label index."
                 ],
                 average_line: [
                   type: :boolean,
                   default: true,
                   doc: "Draws the overall mean silhouette score as a reference."
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 500],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  def plot(x, labels, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    scores =
      Scholar.Metrics.Clustering.silhouette_samples(x, labels, num_clusters: opts[:num_clusters])

    labels = Nx.to_flat_list(labels)
    scores = Nx.to_flat_list(scores)
    names = cluster_names(opts[:cluster_names], opts[:num_clusters])

    values = bars(labels, scores, names)
    mean = Enum.sum(scores) / length(scores)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers(layers(values, names, mean, opts))
  end

  # Within a cluster the bars are sorted by score, which is what makes the
  # familiar knife shape readable: a cluster that tapers early is one whose
  # points sit close to a neighbouring cluster.
  defp bars(labels, scores, names) do
    labels
    |> Enum.zip(scores)
    |> Enum.group_by(fn {label, _} -> label end, fn {_, score} -> score end)
    |> Enum.sort_by(fn {label, _} -> label end)
    |> Enum.flat_map_reduce(0, fn {label, cluster_scores}, offset ->
      sorted = Enum.sort(cluster_scores, :desc)

      rows =
        sorted
        |> Enum.with_index(offset)
        |> Enum.map(fn {score, position} ->
          %{
            "position" => position,
            "position_end" => position + 1,
            "zero" => 0,
            "score" => score,
            "cluster" => Enum.at(names, label, to_string(label))
          }
        end)

      # a gap between clusters keeps the blocks visually separate
      {rows, offset + length(sorted) + 2}
    end)
    |> elem(0)
  end

  defp layers(values, names, mean, opts) do
    max_position = values |> Enum.map(& &1["position"]) |> Enum.max(fn -> 0 end)

    # Both axes are quantitative, so the bar's extent has to be given on both:
    # x from zero out to the score, y spanning the one row the point occupies.
    bars =
      Vl.new()
      |> Vl.mark(:bar, tooltip: true)
      |> Vl.encode_field(:y, "position",
        type: :quantitative,
        title: nil,
        axis: nil,
        scale: [domain: [max_position + 1, -1], nice: false]
      )
      |> Vl.encode_field(:y2, "position_end")
      |> Vl.encode_field(:x, "zero",
        type: :quantitative,
        title: "Silhouette score",
        scale: [domain: score_domain(values), nice: false]
      )
      |> Vl.encode_field(:x2, "score")
      |> Vl.encode_field(:color, "cluster",
        type: :nominal,
        title: "Cluster",
        scale: [domain: names]
      )

    if opts[:average_line] do
      [bars, average_layer(mean)]
    else
      [bars]
    end
  end

  defp average_layer(mean) do
    Vl.new()
    |> Vl.data_from_values([%{"mean" => mean}])
    |> Vl.mark(:rule, stroke_dash: [4, 4], color: "#333", size: 1)
    |> Vl.encode_field(:x, "mean", type: :quantitative)
  end

  # A negative score means the point is closer to another cluster, so the axis
  # has to keep room for it rather than clipping at zero.
  defp score_domain(values) do
    scores = Enum.map(values, & &1["score"])
    min = scores |> Enum.min(fn -> 0.0 end) |> min(0.0)
    [Float.round(min - 0.05, 2), 1.0]
  end

  defp cluster_names(nil, n), do: Enum.map(0..(n - 1), &Integer.to_string/1)

  defp cluster_names(names, n) when length(names) == n, do: Enum.map(names, &to_string/1)

  defp cluster_names(names, n) do
    raise ArgumentError,
          "expected :cluster_names to have one entry per cluster, " <>
            "got #{length(names)} names for #{n} clusters"
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
