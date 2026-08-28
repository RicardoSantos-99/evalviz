defmodule EvalViz.ConfusionMatrix do
  @moduledoc false

  alias EvalViz.Internal
  alias EvalViz.Theme
  alias VegaLite, as: Vl

  @opts_schema NimbleOptions.new!(
                 num_classes: [
                   type: :pos_integer,
                   required: true,
                   doc: "Number of distinct classes."
                 ],
                 class_names: [
                   type: {:list, {:or, [:string, :atom, :integer]}},
                   doc: "Axis labels, one per class. Defaults to `0..num_classes - 1`."
                 ],
                 sample_weights: Internal.weights_option(),
                 normalize: [
                   type: {:in, [:none, :true_class, :predicted, :all]},
                   default: :none,
                   doc: """
                   Whether to show counts (`:none`) or proportions. `:true_class`
                   divides each row by its total, which is the usual choice for
                   reading per-class recall off the diagonal.
                   """
                 ],
                 title: [type: :string, doc: "Chart title."],
                 width: [type: :pos_integer, default: 400],
                 height: [type: :pos_integer, default: 400]
               )

  def schema, do: @opts_schema

  def plot(y_true, y_pred, opts) do
    opts = NimbleOptions.validate!(opts, @opts_schema)
    Internal.assert_paired!(y_true, y_pred, "y_true", "y_pred")

    num_classes = opts[:num_classes]
    labels = Internal.class_labels(opts[:class_names], num_classes)

    matrix = compute(y_true, y_pred, num_classes, opts)
    values = to_values(matrix, labels, opts[:normalize])

    {value_title, format} = value_encoding(opts[:normalize])
    light_text_above = light_text_threshold(matrix)

    Vl.new(vl_opts(opts))
    |> Vl.data_from_values(values)
    |> Vl.layers([
      Vl.new()
      |> Vl.mark(:rect, tooltip: true)
      |> Vl.encode_field(:x, "predicted",
        type: :nominal,
        sort: labels,
        title: "Predicted",
        axis: [label_angle: 0]
      )
      |> Vl.encode_field(:y, "actual", type: :nominal, sort: labels, title: "Actual")
      |> Vl.encode_field(:color, "value",
        type: :quantitative,
        title: value_title,
        scale: [scheme: Theme.heatmap_scheme()],
        legend: [format: format]
      ),
      Vl.new()
      |> Vl.mark(:text, font_size: 13)
      |> Vl.encode_field(:x, "predicted", type: :nominal, sort: labels)
      |> Vl.encode_field(:y, "actual", type: :nominal, sort: labels)
      |> Vl.encode_field(:text, "label", type: :nominal)
      # dark cells need light text to stay readable
      |> Vl.encode(:color,
        condition: [test: "datum.value > #{light_text_above}", value: "white"],
        value: "black"
      )
    ])
  end

  # The blues scheme gets dark enough to swallow black text past roughly two
  # thirds of the range.
  defp light_text_threshold(matrix) do
    matrix |> Nx.reduce_max() |> Nx.to_number() |> Kernel.*(0.65)
  end

  defp compute(y_true, y_pred, num_classes, opts) do
    # Scholar takes no :normalize key at all for raw counts, and names the
    # row-wise mode `true`, which reads as a boolean at the call site.
    computed =
      case opts[:normalize] do
        :none -> [num_classes: num_classes]
        :true_class -> [num_classes: num_classes, normalize: true]
        other -> [num_classes: num_classes, normalize: other]
      end

    # Passed as a list, which every Scholar version accepts.
    computed =
      case opts[:sample_weights] do
        nil ->
          computed

        weights ->
          [{:sample_weights, Internal.weight_list(weights, Nx.axis_size(y_true, 0))} | computed]
      end

    Scholar.Metrics.Classification.confusion_matrix(y_true, y_pred, computed)
  end

  defp to_values(matrix, labels, normalize) do
    rows = Nx.to_list(matrix)

    for {row, actual} <- Enum.with_index(rows),
        {value, predicted} <- Enum.with_index(row) do
      %{
        "actual" => Enum.at(labels, actual),
        "predicted" => Enum.at(labels, predicted),
        "value" => value,
        "label" => cell_label(value, normalize)
      }
    end
  end

  defp cell_label(value, :none), do: to_string(value)

  defp cell_label(value, _normalized) do
    value |> Internal.round_to(3) |> to_string()
  end

  defp value_encoding(:none), do: {"Count", "d"}
  defp value_encoding(_), do: {"Proportion", ".2f"}

  defp vl_opts(opts) do
    base = [width: opts[:width], height: opts[:height]]
    if opts[:title], do: [{:title, opts[:title]} | base], else: base
  end
end
