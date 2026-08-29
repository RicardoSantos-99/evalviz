defmodule Verdict.Reachability do
  @moduledoc false

  alias Verdict.Internal
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 label_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: """
                   Names for the clusters, indexed from zero. Points OPTICS left
                   out are named "noise" whatever this says, since they are not
                   a cluster.
                   """
                 ],
                 cut_off: [
                   type: {:or, [:boolean, :float, :integer]},
                   default: true,
                   doc: """
                   Draws the distance the clusters were extracted at. `true`
                   takes it from the model, a float overrides it, which is how
                   a different cut-off is read off the plot before being fitted
                   again, and `false` leaves it out.
                   """
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 560],
                 height: [type: :pos_integer, default: 300]
               )

  def schema, do: @opts_schema

  def plot(model, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    {reachability, ordering, labels} = fields(model)

    rows = rows(reachability, ordering, labels, opts[:label_names])
    {finite, walls} = Enum.split_with(rows, & &1["reachability"])

    Vl.new(vl_opts(opts))
    |> Vl.layers(layers(finite, walls, domain(finite), cut_off(model, opts[:cut_off]), rows))
  end

  # The reachability of a point says nothing on its own: what makes the plot
  # readable is that neighbours in the cluster ordering sit next to each other,
  # so both arrays are read through `ordering` rather than as they are stored.
  defp rows(reachability, ordering, labels, names) do
    reachability = Internal.to_list(reachability)
    labels = Nx.to_flat_list(labels)

    ordering
    |> Nx.to_flat_list()
    |> Enum.with_index()
    |> Enum.map(fn {point, position} ->
      %{
        "position" => position,
        "position_end" => position + 1,
        "centre" => position + 0.5,
        "zero" => 0,
        "point" => point,
        "reachability" => finite(Enum.at(reachability, point)),
        "label" => Enum.at(labels, point),
        "cluster" => name_of(Enum.at(labels, point), names)
      }
    end)
  end

  # Nx hands infinity back as :infinity rather than as a float, and it reaches
  # Vega-Lite as the string "infinity" if it is left alone.
  defp finite(value) when is_number(value), do: value
  defp finite(_), do: nil

  # OPTICS marks a point it could not reach at all with an infinite distance,
  # which is not a value the axis can carry. Drawing those as full-height rules
  # keeps the wall between two clusters visible without inventing a number.
  #
  # Both axes are quantitative, so the bar's extent has to be given on both: x
  # across the one slot the point occupies, y from zero up to its distance.
  defp layers(finite, walls, domain, cut_off, rows) do
    names = names(rows)
    positions = length(rows)

    bars =
      Vl.new()
      |> Vl.data_from_values(finite)
      |> Vl.mark(:bar, tooltip: true)
      |> Vl.encode_field(:x, "position",
        type: :quantitative,
        title: "Cluster order",
        scale: [domain: [0, positions], nice: false]
      )
      |> Vl.encode_field(:x2, "position_end")
      |> Vl.encode_field(:y, "zero",
        type: :quantitative,
        title: "Reachability distance",
        scale: [domain: domain, nice: false]
      )
      |> Vl.encode_field(:y2, "reachability")
      |> Vl.encode_field(:color, "cluster",
        type: :nominal,
        title: "Cluster",
        scale: [domain: names, range: colours(names)]
      )

    [bars] ++ wall_layer(walls) ++ cut_off_layer(cut_off)
  end

  # Ordered by cluster rather than by where each one first turns up, and noise
  # last, since it is what everything else was picked out of.
  defp names(rows) do
    rows
    |> Enum.uniq_by(& &1["label"])
    |> Enum.sort_by(fn %{"label" => label} -> if label == -1, do: :infinity, else: label end)
    |> Enum.map(& &1["cluster"])
  end

  defp wall_layer([]), do: []

  defp wall_layer(walls) do
    [
      Vl.new()
      |> Vl.data_from_values(walls)
      |> Vl.mark(:rule, Theme.reference_mark() ++ [tooltip: true])
      |> Vl.encode_field(:x, "centre", type: :quantitative)
    ]
  end

  defp cut_off_layer(nil), do: []

  defp cut_off_layer(value) do
    [
      Vl.new()
      |> Vl.data_from_values([%{"cut_off" => value}])
      |> Vl.mark(:rule, Theme.reference_mark())
      |> Vl.encode_field(:y, "cut_off", type: :quantitative)
    ]
  end

  defp cut_off(_model, false), do: nil
  defp cut_off(_model, value) when is_number(value), do: value

  defp cut_off(model, true) do
    case model do
      %{eps: eps} when not is_nil(eps) -> eps |> Nx.to_number() |> finite()
      _ -> nil
    end
  end

  # A model nothing was reachable in still has to draw, so the axis falls back
  # to a unit range rather than to no range at all.
  defp domain([]), do: [0, 1]

  defp domain(finite) do
    max = finite |> Enum.map(& &1["reachability"]) |> Enum.max()
    [0, max * 1.05]
  end

  # Noise is not a cluster and does not take a colour from the sequence, so a
  # model that reached nothing still has a scale to draw against.
  defp colours(names) do
    palette =
      case Enum.reject(names, &(&1 == "noise")) do
        [] -> %{}
        clusters -> clusters |> Enum.zip(Theme.categorical(length(clusters))) |> Map.new()
      end

    Enum.map(names, &Map.get(palette, &1, Theme.muted()))
  end

  defp name_of(-1, _names), do: "noise"
  defp name_of(label, nil), do: to_string(label)
  defp name_of(label, names), do: names |> Enum.at(label, label) |> to_string()

  defp fields(%{reachability: reachability, ordering: ordering, labels: labels})
       when not is_nil(reachability) and not is_nil(ordering) do
    {reachability, ordering, labels}
  end

  defp fields(other) do
    raise ArgumentError,
          "expected a fitted Scholar.Cluster.OPTICS model, which carries " <>
            ":reachability, :ordering and :labels, got: #{inspect(other)}"
  end

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
