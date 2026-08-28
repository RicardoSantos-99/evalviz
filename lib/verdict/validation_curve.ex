defmodule Verdict.ValidationCurve do
  @moduledoc false

  alias Verdict.Folds
  alias Verdict.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 param_name: [
                   type: :string,
                   default: "Parameter value",
                   doc: "Label for the x axis, normally the hyperparameter's name."
                 ],
                 scale: [
                   type: {:in, [:linear, :log]},
                   default: :linear,
                   doc: """
                   `:log` for a parameter searched over orders of magnitude,
                   which is how regularisation strengths are usually swept.
                   Needs every value to be positive.
                   """
                 ],
                 score_title: [type: :string, default: "Score"],
                 best: [
                   type: {:in, [:max, :min, :none]},
                   default: :max,
                   doc: """
                   Marks the parameter with the best validation score. Use
                   `:min` when the score is an error, `:none` to mark nothing.
                   """
                 ],
                 spread: [
                   type: :boolean,
                   default: true,
                   doc: """
                   Shades one standard deviation around each mean. Needs a score
                   per fold.
                   """
                 ],
                 labels: [
                   type: {:list, :string},
                   default: ["Training", "Validation"],
                   doc: "Names for the two curves."
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 500],
                 height: [type: :pos_integer, default: 380]
               )

  def schema, do: @opts_schema

  def plot(param_values, train_scores, validation_scores, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)

    values = to_list(param_values)
    count = length(values)

    train = Folds.normalize(train_scores, count, "train_scores", "parameter value")

    validation =
      Folds.normalize(validation_scores, count, "validation_scores", "parameter value")

    numeric? = numeric?(values)
    assert_loggable!(values, numeric?, opts[:scale])

    [train_label, validation_label] = opts[:labels]
    positions = Enum.map(values, &position(&1, numeric?))

    rows =
      Folds.rows(positions, train, train_label, "param") ++
        Folds.rows(positions, validation, validation_label, "param")

    best = best(rows, validation_label, opts[:best])

    Vl.new(vl_opts(opts, best))
    |> Vl.data_from_values(rows)
    |> Vl.layers(layers(numeric?, positions, best, Folds.folded?(train), opts))
  end

  # A parameter can be a number, but it can just as well be a boolean or an
  # atom, and those have to sit on a nominal axis.
  defp numeric?(values), do: Enum.all?(values, &is_number/1)

  defp position(value, true), do: value
  defp position(value, false), do: to_string(value)

  defp assert_loggable!(values, numeric?, :log) do
    cond do
      not numeric? ->
        raise ArgumentError, "expected numeric parameter values for a log scale"

      Enum.any?(values, &(&1 <= 0)) ->
        raise ArgumentError,
              "expected every parameter value to be positive for a log scale, " <>
                "got #{inspect(Enum.filter(values, &(&1 <= 0)))}"

      true ->
        :ok
    end
  end

  defp assert_loggable!(_values, _numeric?, :linear), do: :ok

  defp best(_rows, _label, :none), do: nil

  defp best(rows, label, direction) do
    picker = if direction == :max, do: &Enum.max_by/2, else: &Enum.min_by/2

    rows
    |> Enum.filter(&(&1["series"] == label))
    |> picker.(& &1["score"])
  end

  defp layers(numeric?, positions, best, folded?, opts) do
    line =
      Vl.new()
      |> Vl.mark(:line, point: true, tooltip: true)
      |> encode_param(numeric?, positions, opts[:scale], title: opts[:param_name])
      |> Vl.encode_field(:y, "score",
        type: :quantitative,
        title: opts[:score_title],
        scale: [zero: false]
      )
      |> Vl.encode_field(:color, "series",
        type: :nominal,
        title: nil,
        scale: [domain: opts[:labels], range: Theme.categorical(2)]
      )

    band = if opts[:spread] and folded?, do: [band(numeric?, positions, opts)], else: []
    marker = if best, do: [best_layer(best, numeric?, positions, opts[:scale])], else: []

    band ++ marker ++ [line]
  end

  # Drawn under the lines so the means stay legible on top of it.
  defp band(numeric?, positions, opts) do
    Vl.new()
    |> Vl.mark(:area, opacity: 0.2)
    |> encode_param(numeric?, positions, opts[:scale], [])
    |> Vl.encode_field(:y, "lower", type: :quantitative, title: opts[:score_title])
    |> Vl.encode_field(:y2, "upper")
    # No `legend: nil` here: the band shares this field with the line, and
    # suppressing it on one layer suppresses the merged legend for both.
    |> Vl.encode_field(:color, "series",
      type: :nominal,
      title: nil,
      scale: [domain: opts[:labels], range: Theme.categorical(2)]
    )
  end

  defp best_layer(best, numeric?, positions, scale) do
    Vl.new()
    |> Vl.data_from_values([%{"param" => best["param"]}])
    |> Vl.mark(:rule, Theme.reference_mark())
    |> encode_param(numeric?, positions, scale, [])
  end

  defp encode_param(layer, true, _positions, scale, extra) do
    scale_opts = if scale == :log, do: [scale: [type: :log]], else: []
    Vl.encode_field(layer, :x, "param", [type: :quantitative] ++ scale_opts ++ extra)
  end

  # A non-numeric parameter sits on a nominal axis, sorted the way it was given
  # rather than alphabetically.
  defp encode_param(layer, false, positions, _scale, extra) do
    Vl.encode_field(layer, :x, "param", [type: :nominal, sort: positions] ++ extra)
  end

  defp to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp to_list(list) when is_list(list), do: list

  defp vl_opts(opts, best) do
    base = [width: opts[:width], height: opts[:height]]

    subtitle =
      if best do
        "best #{Float.round(best["score"], 4)} at #{opts[:param_name]} = #{best["param"]}"
      end

    case {opts[:title], subtitle} do
      {nil, nil} -> base
      {title, nil} -> [{:title, title} | base]
      {nil, subtitle} -> [{:title, [text: "", subtitle: subtitle]} | base]
      {title, subtitle} -> [{:title, [text: title, subtitle: subtitle]} | base]
    end
  end
end
