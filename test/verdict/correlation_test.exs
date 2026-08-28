defmodule Verdict.CorrelationTest do
  use ExUnit.Case, async: true

  # numpy.corrcoef on this data gives row 0 as [1.0, 0.8, -0.9797959].
  defp x do
    Nx.tensor(
      [[1.0, 2.0, 5.0], [2.0, 1.0, 4.0], [3.0, 5.0, 2.0], [4.0, 3.0, 1.0], [5.0, 9.0, 0.5]],
      type: :f64
    )
  end

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp heatmap(plot), do: spec(plot)["layer"] |> hd()

  defp cell(plot, row, column) do
    plot
    |> values()
    |> Enum.find(&(&1["row"] == row and &1["column"] == column))
    |> Map.fetch!("value")
  end

  describe "correlation/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.correlation(x())
    end

    test "draws a cell for every pair of columns" do
      assert length(values(Verdict.correlation(x()))) == 9
    end

    test "matches what numpy computes" do
      plot = Verdict.correlation(x())

      assert_in_delta cell(plot, "0", "0"), 1.0, 1.0e-6
      assert_in_delta cell(plot, "0", "1"), 0.8, 1.0e-6
      assert_in_delta cell(plot, "0", "2"), -0.9797959, 1.0e-6
    end

    test "is symmetric" do
      plot = Verdict.correlation(x())
      assert_in_delta cell(plot, "1", "2"), cell(plot, "2", "1"), 1.0e-9
    end

    # A matrix of weak correlations would otherwise colour like a matrix of
    # strong ones, since the scale would stretch to fit whatever it was given.
    test "pins the colour scale to -1..1 rather than fitting it" do
      assert heatmap(Verdict.correlation(x()))["encoding"]["color"]["scale"]["domain"] == [-1, 1]
    end

    test "uses a diverging scheme" do
      assert heatmap(Verdict.correlation(x()))["encoding"]["color"]["scale"]["scheme"] ==
               Verdict.Theme.diverging_scheme()
    end

    test "names the features on both axes" do
      plot = Verdict.correlation(x(), feature_names: ["a", "b", "c"])

      assert values(plot) |> Enum.map(& &1["row"]) |> Enum.uniq() == ["a", "b", "c"]
      assert values(plot) |> Enum.map(& &1["column"]) |> Enum.uniq() == ["a", "b", "c"]
    end

    test "writes the coefficient in each cell" do
      assert length(spec(Verdict.correlation(x()))["layer"]) == 2
      assert Verdict.correlation(x()) |> values() |> hd() |> Map.fetch!("label") == "1.0"
    end

    test "values: false leaves the numbers out" do
      assert length(spec(Verdict.correlation(x(), values: false))["layer"]) == 1
    end

    test "rejects data that is not a matrix" do
      assert_raise ArgumentError, ~r/rank-2 tensor/, fn ->
        Verdict.correlation(Nx.tensor([1.0, 2.0, 3.0]))
      end
    end

    test "rejects the wrong number of feature names" do
      assert_raise ArgumentError, ~r/one entry per class/, fn ->
        Verdict.correlation(x(), feature_names: ["a"])
      end
    end
  end
end
