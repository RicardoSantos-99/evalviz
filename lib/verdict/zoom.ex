defmodule Verdict.Zoom do
  @moduledoc false

  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 encodings: [
                   type: {:list, {:in, [:x, :y]}},
                   default: [:x, :y],
                   doc: """
                   Which axes the drag and the scroll wheel move. Naming one
                   axis pins the other, which is what you want when the range
                   being read against is the point, such as a curve that has to
                   stay against its diagonal.
                   """
                 ]
               )

  @param "verdict_zoom"

  # Vega-Lite binds a selection to a scale only where that scale is continuous.
  @continuous ~w(quantitative temporal)

  def schema, do: @opts_schema

  def apply(%Vl{spec: spec} = plot, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    encodings = Enum.map(opts[:encodings], &Atom.to_string/1)

    if encodings == [] do
      raise ArgumentError, "expected :encodings to name at least one axis"
    end

    if zoomable?(spec) do
      raise ArgumentError, "this plot is already zoomable"
    end

    case walk(spec, %{}, encodings) do
      {spec, true} ->
        %{plot | spec: spec}

      {_spec, false} ->
        raise ArgumentError,
              "no view in this plot can be zoomed on #{Enum.join(encodings, " and ")}: " <>
                "Vega-Lite binds a drag to a scale only where that scale is continuous, " <>
                "and a matrix or a bar chart puts its categories on a discrete one"
    end
  end

  defp zoomable?(%{} = spec) do
    Enum.any?(spec["params"] || [], &(&1["name"] == @param)) or
      spec |> children() |> Enum.any?(&zoomable?/1)
  end

  defp zoomable?(_), do: false

  defp walk(%{"mark" => _} = spec, inherited, encodings) do
    encoding = inherit(inherited, spec)

    if Enum.all?(encodings, &continuous?(encoding, &1)) do
      {add_param(spec, encodings), true}
    else
      {spec, false}
    end
  end

  # Layers share one pair of scales, so the parameter belongs to a single one of
  # them. Repeating it, or hoisting it to the parent, collides on the signal the
  # binding generates.
  defp walk(%{"layer" => layers} = spec, inherited, encodings) do
    inherited = inherit(inherited, spec)

    {layers, zoomed?} =
      Enum.map_reduce(layers, false, fn
        layer, false -> walk(layer, inherited, encodings)
        layer, true -> {layer, true}
      end)

    {Map.put(spec, "layer", layers), zoomed?}
  end

  defp walk(%{} = spec, inherited, encodings) do
    inherited = inherit(inherited, spec)

    # "spec" is how facet and repeat wrap the view they multiply.
    Enum.reduce(~w(concat hconcat vconcat spec), {spec, false}, fn key, {spec, zoomed?} ->
      case spec do
        %{^key => views} when is_list(views) ->
          {views, any?} = map_reduce_views(views, inherited, encodings)
          {Map.put(spec, key, views), zoomed? or any?}

        %{^key => view} when is_map(view) ->
          {view, any?} = walk(view, inherited, encodings)
          {Map.put(spec, key, view), zoomed? or any?}

        _ ->
          {spec, zoomed?}
      end
    end)
  end

  defp walk(spec, _inherited, _encodings), do: {spec, false}

  defp map_reduce_views(views, inherited, encodings) do
    Enum.map_reduce(views, false, fn view, any? ->
      {view, this?} = walk(view, inherited, encodings)
      {view, any? or this?}
    end)
  end

  defp children(%{} = spec) do
    Enum.flat_map(~w(layer concat hconcat vconcat spec), fn key ->
      case spec[key] do
        views when is_list(views) -> views
        view when is_map(view) -> [view]
        nil -> []
      end
    end)
  end

  defp continuous?(encoding, axis), do: encoding[axis]["type"] in @continuous

  defp inherit(inherited, spec), do: Map.merge(inherited, spec["encoding"] || %{})

  defp add_param(spec, encodings) do
    %Vl{spec: spec}
    |> Vl.param(@param, select: [type: :interval, encodings: encodings], bind: :scales)
    |> Map.fetch!(:spec)
  end
end
