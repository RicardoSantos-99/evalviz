defmodule EvalViz.Elbow do
  @moduledoc false

  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 mark_elbow: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Marks the k furthest from the line joining the first and last
                   points, which is the corner the plot is read for. Needs at
                   least three points.
                   """
                 ],
                 metric_title: [
                   type: :string,
                   default: "Inertia",
                   doc: "Label for the y axis."
                 ],
                 k_title: [type: :string, default: "Clusters"],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 460],
                 height: [type: :pos_integer, default: 340]
               )

  def schema, do: @opts_schema

  def plot(ks, values, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if length(ks) != length(values) do
      raise ArgumentError,
            "expected one score per k, got #{length(values)} for #{length(ks)} values of k"
    end

    if ks == [] do
      raise ArgumentError, "expected at least one k to plot"
    end

    points = Enum.zip_with(ks, values, &%{"k" => &1, "value" => &2})
    elbow = if opts[:mark_elbow], do: elbow(ks, values)

    Vl.new(vl_opts(opts, elbow))
    |> Vl.data_from_values(points)
    |> Vl.layers(layers(elbow, opts))
  end

  # Distance to the chord, on axes rescaled to 0..1 first: k spans single digits
  # and inertia can span thousands, so raw distance would only ever measure the
  # larger of the two.
  defp elbow(ks, _values) when length(ks) < 3, do: nil

  defp elbow(ks, values) do
    xs = rescale(ks)
    ys = rescale(values)

    {x1, y1} = {hd(xs), hd(ys)}
    {x2, y2} = {List.last(xs), List.last(ys)}
    {dx, dy} = {x2 - x1, y2 - y1}
    chord = :math.sqrt(dx * dx + dy * dy)

    if chord == 0.0, do: nil, else: furthest(xs, ys, ks, {x1, y1, x2, y2, dx, dy, chord})
  end

  defp furthest(xs, ys, ks, {x1, y1, x2, y2, dx, dy, chord}) do
    {distance, k} =
      xs
      |> Enum.zip(ys)
      |> Enum.zip(ks)
      |> Enum.map(fn {{x, y}, k} ->
        {abs(dy * x - dx * y + x2 * y1 - y2 * x1) / chord, k}
      end)
      |> Enum.max_by(&elem(&1, 0))

    # Every point on the chord means there is no corner, which is what a
    # straight run of scores is. Naming one anyway would invent a result.
    if distance < 1.0e-9, do: nil, else: k
  end

  defp rescale(values) do
    {min, max} = Enum.min_max(values)
    span = max - min

    if span == 0,
      do: Enum.map(values, fn _ -> 0.0 end),
      else: Enum.map(values, &((&1 - min) / span))
  end

  defp layers(elbow, opts) do
    curve =
      Vl.new()
      |> Vl.mark(:line, point: true, tooltip: true, color: Theme.primary())
      # k is a count, so the axis must not offer 2.5 clusters
      |> Vl.encode_field(:x, "k",
        type: :quantitative,
        title: opts[:k_title],
        axis: [tick_min_step: 1, format: "d"]
      )
      |> Vl.encode_field(:y, "value",
        type: :quantitative,
        title: opts[:metric_title],
        scale: [zero: false]
      )

    if elbow, do: [elbow_layer(elbow), curve], else: [curve]
  end

  defp elbow_layer(elbow) do
    Vl.new()
    |> Vl.data_from_values([%{"k" => elbow}])
    |> Vl.mark(:rule, Theme.reference_mark())
    |> Vl.encode_field(:x, "k", type: :quantitative)
  end

  defp vl_opts(opts, elbow) do
    base = [width: opts[:width], height: opts[:height]]
    subtitle = if elbow, do: "elbow at k = #{elbow}"

    case {opts[:title], subtitle} do
      {nil, nil} -> base
      {title, nil} -> [{:title, title} | base]
      {nil, subtitle} -> [{:title, [text: "", subtitle: subtitle]} | base]
      {title, subtitle} -> [{:title, [text: title, subtitle: subtitle]} | base]
    end
  end
end
