defmodule EvalViz.CurvesTest do
  use ExUnit.Case, async: true

  # Same fixture sklearn's own roc_curve docs use, so the expected rates below
  # are the ones scikit-learn returns:
  #   fpr [0, 0, 0.5, 0.5, 1], tpr [0, 0.5, 0.5, 1, 1], auc 0.75
  defp y_true, do: Nx.tensor([0, 0, 1, 1])
  defp scores, do: Nx.tensor([0.1, 0.4, 0.35, 0.8])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]

  defp curve_layer(plot), do: spec(plot)["layer"] |> List.last()

  describe "roc_curve/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.roc_curve(y_true(), scores())
    end

    test "plots the rates scikit-learn computes" do
      points = values(EvalViz.roc_curve(y_true(), scores()))

      assert Enum.map(points, & &1["fpr"]) == [0.0, 0.0, 0.5, 0.5, 1.0]
      assert Enum.map(points, & &1["tpr"]) == [0.0, 0.5, 0.5, 1.0, 1.0]
    end

    test "puts AUC in the series label" do
      plot = EvalViz.roc_curve([{"Logistic", y_true(), scores()}])
      assert [%{"series" => label} | _] = values(plot)
      assert label == "Logistic (AUC = 0.75)"
    end

    test "draws the chance diagonal by default" do
      [reference, _curve] = spec(EvalViz.roc_curve(y_true(), scores()))["layer"]

      assert reference["data"]["values"] == [
               %{"x" => 0, "y" => 0},
               %{"x" => 1, "y" => 1}
             ]
    end

    test "chance_line: false drops the reference" do
      layers = spec(EvalViz.roc_curve(y_true(), scores(), chance_line: false))["layer"]
      assert length(layers) == 1
    end

    test "a single curve carries no colour legend" do
      refute curve_layer(EvalViz.roc_curve(y_true(), scores()))["encoding"]["color"]
    end

    test "comparing models colours by series" do
      plot =
        EvalViz.roc_curve([
          {"A", y_true(), scores()},
          {"B", y_true(), Nx.tensor([0.2, 0.3, 0.6, 0.9])}
        ])

      assert curve_layer(plot)["encoding"]["color"]["field"] == "series"

      labels = plot |> values() |> Enum.map(& &1["series"]) |> Enum.uniq()
      assert length(labels) == 2
    end

    test "shows markers for few thresholds and hides them for many" do
      assert curve_layer(EvalViz.roc_curve(y_true(), scores()))["mark"]["point"] == true

      many_true = Nx.tensor(Enum.map(1..200, &rem(&1, 2)))
      many_scores = Nx.tensor(Enum.map(1..200, &(&1 / 200)))

      assert curve_layer(EvalViz.roc_curve(many_true, many_scores))["mark"]["point"] == false
    end

    test "steps between thresholds instead of interpolating" do
      assert curve_layer(EvalViz.roc_curve(y_true(), scores()))["mark"]["interpolate"] ==
               "step-after"
    end

    test "pins both axes to 0..1" do
      layer = curve_layer(EvalViz.roc_curve(y_true(), scores()))
      assert layer["encoding"]["x"]["scale"]["domain"] == [0, 1]
      assert layer["encoding"]["y"]["scale"]["domain"] == [0, 1]
    end

    test "rejects a y_true that is not binary" do
      assert_raise ArgumentError, ~r/only 0 and 1/, fn ->
        EvalViz.roc_curve(Nx.tensor([0, 1, 2, 1]), scores())
      end
    end

    test "rejects inputs of different lengths" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        EvalViz.roc_curve(y_true(), Nx.tensor([0.1, 0.2]))
      end
    end
  end

  describe "precision_recall_curve/3" do
    test "plots the values scikit-learn computes" do
      points = values(EvalViz.precision_recall_curve(y_true(), scores()))

      assert Enum.map(points, & &1["recall"]) == [1.0, 1.0, 0.5, 0.5, 0.0]

      assert points
             |> Enum.map(& &1["precision"])
             |> Enum.zip([0.5, 0.6666667, 0.5, 1.0, 1.0])
             |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "puts average precision in the series label" do
      plot = EvalViz.precision_recall_curve([{"Logistic", y_true(), scores()}])
      assert [%{"series" => label} | _] = values(plot)
      assert label =~ "AP = "
    end

    test "baselines at the share of positives, not the diagonal" do
      # two of four labels are positive
      [reference, _curve] = spec(EvalViz.precision_recall_curve(y_true(), scores()))["layer"]

      assert [%{"y" => start_rate}, %{"y" => end_rate}] = reference["data"]["values"]
      assert start_rate == end_rate
      assert_in_delta start_rate, 0.5, 1.0e-6
    end
  end

  describe "det_curve/3" do
    test "plots false negative against false positive rate" do
      plot = EvalViz.det_curve(y_true(), scores())
      layer = curve_layer(plot)

      assert layer["encoding"]["x"]["field"] == "fpr"
      assert layer["encoding"]["y"]["field"] == "fnr"
      assert Enum.map(values(plot), & &1["fnr"]) == [0.0, 0.0, 0.5, 0.5]
    end

    test "has no chance reference" do
      assert length(spec(EvalViz.det_curve(y_true(), scores()))["layer"]) == 1
    end

    test "leaves the axes unpinned, since rates here are not bounded to a corner" do
      layer = curve_layer(EvalViz.det_curve(y_true(), scores()))
      refute layer["encoding"]["x"]["scale"]["domain"]
    end
  end
end
