defmodule EvalViz.ConfusionMatrixTest do
  use ExUnit.Case, async: true

  # y_true/y_pred below produce this matrix, which is the one used in
  # Scholar.Metrics.Classification.confusion_matrix/3's own doctest:
  #
  #   [[1, 1, 0],
  #    [1, 0, 1],
  #    [0, 0, 2]]
  defp y_true, do: Nx.tensor([0, 0, 1, 1, 2, 2])
  defp y_pred, do: Nx.tensor([0, 1, 0, 2, 2, 2])

  defp values(spec), do: VegaLite.to_spec(spec)["data"]["values"]

  defp cell(spec, actual, predicted) do
    Enum.find(values(spec), &(&1["actual"] == actual and &1["predicted"] == predicted))
  end

  describe "confusion_matrix/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3)
    end

    test "encodes one datum per cell, with counts matching the matrix" do
      spec = EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3)

      assert length(values(spec)) == 9

      assert cell(spec, "0", "0")["value"] == 1
      assert cell(spec, "0", "1")["value"] == 1
      assert cell(spec, "0", "2")["value"] == 0
      assert cell(spec, "1", "0")["value"] == 1
      assert cell(spec, "1", "1")["value"] == 0
      assert cell(spec, "1", "2")["value"] == 1
      assert cell(spec, "2", "0")["value"] == 0
      assert cell(spec, "2", "1")["value"] == 0
      assert cell(spec, "2", "2")["value"] == 2
    end

    test "writes the count into each cell as a label" do
      spec = EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3)

      assert cell(spec, "2", "2")["label"] == "2"
      assert cell(spec, "0", "2")["label"] == "0"
    end

    test "normalize: :true_class makes each row sum to 1" do
      spec = EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3, normalize: :true_class)

      for actual <- ["0", "1", "2"] do
        row_total =
          values(spec)
          |> Enum.filter(&(&1["actual"] == actual))
          |> Enum.map(& &1["value"])
          |> Enum.sum()

        assert_in_delta row_total, 1.0, 1.0e-6
      end
    end

    test "normalize: :all makes the whole matrix sum to 1" do
      spec = EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3, normalize: :all)

      total = values(spec) |> Enum.map(& &1["value"]) |> Enum.sum()
      assert_in_delta total, 1.0, 1.0e-6
    end

    test "labels axes with class_names, keeping the given order" do
      spec =
        EvalViz.confusion_matrix(y_true(), y_pred(),
          num_classes: 3,
          class_names: ["cat", "dog", "bird"]
        )

      assert cell(spec, "cat", "cat")["value"] == 1
      assert cell(spec, "bird", "bird")["value"] == 2

      # nominal axes sort alphabetically unless told otherwise, which would put
      # bird first and silently transpose the reader's mental model
      for layer <- VegaLite.to_spec(spec)["layer"] do
        assert layer["encoding"]["x"]["sort"] == ["cat", "dog", "bird"]
        assert layer["encoding"]["y"]["sort"] == ["cat", "dog", "bird"]
      end
    end

    test "draws the value on top of the heatmap" do
      spec = EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3)
      [heatmap, text] = VegaLite.to_spec(spec)["layer"]

      assert heatmap["mark"]["type"] == "rect"
      assert text["mark"]["type"] == "text"
      assert text["encoding"]["text"]["field"] == "label"
    end

    test "switches to light text on the darkest cells" do
      spec = EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3)
      [_heatmap, text] = VegaLite.to_spec(spec)["layer"]

      assert text["encoding"]["color"]["value"] == "black"
      assert text["encoding"]["color"]["condition"]["value"] == "white"
    end

    test "rejects class_names that do not cover every class" do
      assert_raise ArgumentError, ~r/one entry per class/, fn ->
        EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 3, class_names: ["cat", "dog"])
      end
    end

    test "rejects inputs of different lengths" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        EvalViz.confusion_matrix(y_true(), Nx.tensor([0, 1]), num_classes: 3)
      end
    end

    test "rejects inputs that are not rank-1" do
      assert_raise ArgumentError, ~r/rank-1/, fn ->
        EvalViz.confusion_matrix(Nx.tensor([[0, 1], [1, 0]]), y_pred(), num_classes: 3)
      end
    end

    test "requires num_classes" do
      assert_raise NimbleOptions.ValidationError, fn ->
        EvalViz.confusion_matrix(y_true(), y_pred())
      end
    end
  end
end
