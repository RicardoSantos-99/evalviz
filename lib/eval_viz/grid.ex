defmodule EvalViz.Grid do
  @moduledoc false

  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 columns: [
                   type: :pos_integer,
                   default: 2,
                   doc: "How many plots to fit on a row before wrapping."
                 ],
                 title: [type: :string, doc: "Title above the whole grid."]
               )

  def schema, do: @opts_schema

  # These belong to the outer spec, and Vega-Lite refuses them on a child.
  # `VegaLite.config/2` sets one, so it has to be applied to the grid rather
  # than to the plots going into it. `$schema` is not among them: every spec
  # carries one and it is stripped on the way in.
  @top_level ~w(background padding autosize config usermeta)

  def plot(views, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    if views == [] do
      raise ArgumentError, "expected at least one plot to lay out"
    end

    Enum.each(views, &assert_view!/1)

    Vl.new(vl_opts(opts))
    |> Vl.concat(views, :wrappable)
  end

  defp assert_view!(%Vl{spec: spec}) do
    case Enum.find(@top_level, &Map.has_key?(spec, &1)) do
      nil ->
        :ok

      key ->
        raise ArgumentError,
              "a plot in the grid carries the top-level key #{inspect(key)}, which " <>
                "Vega-Lite only allows on the outermost spec. Apply it to the grid instead."
    end
  end

  defp assert_view!(other) do
    raise ArgumentError, "expected a VegaLite specification, got: #{inspect(other)}"
  end

  defp vl_opts(opts) do
    base = [columns: opts[:columns]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
