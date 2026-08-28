defmodule Verdict.Projection do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 label_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: """
                   Names for the values in `labels`, indexed from zero. Defaults
                   to the label itself.
                   """
                 ],
                 components: [
                   type: {:list, :non_neg_integer},
                   default: [0, 1],
                   doc: "Which two columns of the embedding to draw, as `[x, y]`."
                 ],
                 equal_axes: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Gives both axes the same range. The two directions of an
                   embedding measure the same thing, so letting them scale apart
                   would stretch the shape of the data.
                   """
                 ],
                 opacity: [
                   type: :float,
                   default: 0.7,
                   doc: "Point opacity, which is what keeps dense regions readable."
                 ],
                 point_size: [type: :pos_integer, default: 60],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 420],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  def plot(embedding, panels, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    {xs, ys} = coordinates(embedding, opts[:components])
    domain = domain(xs, ys, opts[:equal_axes])

    case panels do
      [] ->
        panel(size(opts) ++ title(opts[:title]), xs, ys, nil, domain, opts)

      [{panel_title, labels}] ->
        panel(size(opts) ++ title(opts[:title] || panel_title), xs, ys, labels, domain, opts)

      many ->
        concat(many, xs, ys, domain, opts)
    end
  end

  # Each panel builds its scale from its own labels, so a clustering that found
  # a different number of groups still renders instead of being squeezed into
  # the other panel's domain. The axes stay shared, which is what lets the two
  # clouds be compared at all.
  defp concat(panels, xs, ys, domain, opts) do
    views =
      Enum.map(panels, fn {panel_title, labels} ->
        panel(size(opts) ++ title(panel_title), xs, ys, labels, domain, opts)
      end)

    Vl.new(title(opts[:title]))
    |> Vl.concat(views, :horizontal)
  end

  defp panel(new_opts, xs, ys, labels, domain, opts) do
    rows = rows(xs, ys, labels, opts)

    layer =
      Vl.new(new_opts)
      |> Vl.data_from_values(rows)
      |> Vl.mark(:point,
        filled: true,
        size: opts[:point_size],
        opacity: opts[:opacity],
        tooltip: true
      )
      |> Vl.encode_field(
        :x,
        "x",
        [type: :quantitative, title: axis_title(opts, 0)] ++ scale(domain)
      )
      |> Vl.encode_field(
        :y,
        "y",
        [type: :quantitative, title: axis_title(opts, 1)] ++ scale(domain)
      )

    if labels, do: encode_labels(layer, rows), else: layer
  end

  defp encode_labels(layer, rows) do
    names = rows |> Enum.map(& &1["label"]) |> Enum.uniq()

    Vl.encode_field(layer, :color, "label",
      type: :nominal,
      title: nil,
      scale: [domain: names, range: Theme.categorical(length(names))]
    )
  end

  defp rows(xs, ys, nil, _opts), do: Enum.zip_with(xs, ys, &%{"x" => &1, "y" => &2})

  defp rows(xs, ys, labels, opts) do
    values = Nx.to_flat_list(labels)

    if length(values) != length(xs) do
      raise ArgumentError,
            "expected labels to have one entry per row of the embedding, " <>
              "got #{length(values)} for #{length(xs)} rows"
    end

    [xs, ys, values]
    |> Enum.zip()
    |> Enum.map(fn {x, y, label} ->
      %{"x" => x, "y" => y, "label" => name_of(label, opts[:label_names])}
    end)
  end

  defp name_of(label, nil), do: to_string(label)
  defp name_of(label, names), do: names |> Enum.at(trunc(label), label) |> to_string()

  defp coordinates(%{embedding: tensor}, components), do: coordinates(tensor, components)

  defp coordinates(tensor, [x, y]) do
    if Nx.rank(tensor) != 2 do
      raise ArgumentError,
            "expected the embedding to be a rank-2 tensor of " <>
              "{num_samples, num_components}, got rank #{Nx.rank(tensor)}"
    end

    available = Nx.axis_size(tensor, 1)

    if x >= available or y >= available do
      raise ArgumentError,
            "expected :components to index the embedding's #{available} " <>
              "columns, got #{inspect([x, y])}"
    end

    {Internal.to_list(tensor[[.., x]]), Internal.to_list(tensor[[.., y]])}
  end

  defp coordinates(_tensor, components) do
    raise ArgumentError, "expected :components to name two columns, got #{inspect(components)}"
  end

  defp domain(_xs, _ys, false), do: nil

  defp domain(xs, ys, true) do
    {min, max} = Enum.min_max(xs ++ ys)
    padding = max((max - min) * 0.05, 1.0e-6)
    [min - padding, max + padding]
  end

  defp scale(nil), do: [scale: [zero: false]]
  defp scale(domain), do: [scale: [domain: domain, nice: false]]

  defp axis_title(opts, index), do: "Component #{Enum.at(opts[:components], index)}"

  defp size(opts), do: [width: opts[:width], height: opts[:height]]

  defp title(nil), do: []
  defp title(text), do: [title: text]
end
