defmodule EvalViz.RegressionTest do
  use ExUnit.Case, async: true

  defp y_true, do: Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
  defp y_pred, do: Nx.tensor([1.2, 1.8, 3.5, 3.9, 5.4])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp scatter(plot), do: spec(plot)["layer"] |> List.last()

  describe "predicted_vs_actual/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.predicted_vs_actual(y_true(), y_pred())
    end

    test "pairs each actual with its prediction" do
      points = values(EvalViz.predicted_vs_actual(y_true(), y_pred()))

      assert Enum.map(points, & &1["actual"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

      points
      |> Enum.map(& &1["predicted"])
      |> Enum.zip([1.2, 1.8, 3.5, 3.9, 5.4])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "gives both axes the same range so the diagonal means what it looks like" do
      layer = scatter(EvalViz.predicted_vs_actual(y_true(), y_pred()))

      assert layer["encoding"]["x"]["scale"]["domain"] ==
               layer["encoding"]["y"]["scale"]["domain"]
    end

    test "draws the diagonal across the shared range" do
      plot = EvalViz.predicted_vs_actual(y_true(), y_pred())
      [reference, scatter] = spec(plot)["layer"]

      [min, max] = scatter["encoding"]["x"]["scale"]["domain"]

      assert reference["data"]["values"] == [
               %{"x" => min, "y" => min},
               %{"x" => max, "y" => max}
             ]
    end

    test "reference_line: false drops the diagonal" do
      plot = EvalViz.predicted_vs_actual(y_true(), y_pred(), reference_line: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "rejects inputs of different lengths" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        EvalViz.predicted_vs_actual(y_true(), Nx.tensor([1.0]))
      end
    end
  end

  describe "residuals/3" do
    test "plots prediction error against the prediction" do
      points = values(EvalViz.residuals(y_true(), y_pred()))

      assert Enum.map(points, & &1["predicted"])
             |> Enum.zip([1.2, 1.8, 3.5, 3.9, 5.4])
             |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)

      # residual is predicted minus actual, so an over-prediction reads positive
      points
      |> Enum.map(& &1["residual"])
      |> Enum.zip([0.2, -0.2, 0.5, -0.1, 0.4])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-5 end)
    end

    test "draws the zero line by default" do
      [reference, _scatter] = spec(EvalViz.residuals(y_true(), y_pred()))["layer"]

      assert reference["data"]["values"] == [%{"zero" => 0}]
      assert reference["mark"]["type"] == "rule"
    end

    test "reference_line: false drops the zero line" do
      plot = EvalViz.residuals(y_true(), y_pred(), reference_line: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "leaves the residual axis free, since it is not on the target's scale" do
      layer = scatter(EvalViz.residuals(y_true(), y_pred()))
      refute layer["encoding"]["y"]["scale"]
    end
  end
end
