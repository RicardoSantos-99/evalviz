defmodule Verdict.FeatureDistributionTest do
  use ExUnit.Case, async: true

  defp x, do: Nx.tensor([[1.0, 5.0], [2.0, 5.5], [8.0, 1.0], [9.0, 1.5]])
  defp labels, do: Nx.tensor([0, 0, 1, 1])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp panels(plot), do: spec(plot)["concat"]
  defp values(panel), do: panel["data"]["values"]

  defp of_class(panel, name) do
    panel |> values() |> Enum.filter(&(&1["class"] == name)) |> Enum.map(& &1["value"])
  end

  describe "feature_distribution/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.feature_distribution(x(), labels())
    end

    test "draws one panel per column of x" do
      assert length(panels(Verdict.feature_distribution(x(), labels()))) == 2
    end

    test "names each panel after its feature" do
      plot = Verdict.feature_distribution(x(), labels(), feature_names: ["income", "age"])
      assert panels(plot) |> Enum.map(& &1["title"]) == ["income", "age"]
    end

    test "names panels by column index when not given names" do
      assert panels(Verdict.feature_distribution(x(), labels())) |> Enum.map(& &1["title"]) ==
               ["0", "1"]
    end

    # income (column 0) is 1, 2 for class 0 and 8, 9 for class 1: with 5 bins
    # over the shared span 1..9, class 0 falls in the first bin and class 1 in
    # the last, so the two humps sit apart with nothing between them.
    test "puts each class's rows in the bin the shared span gives them" do
      [income | _] = panels(Verdict.feature_distribution(x(), labels(), bins: 5))
      rows = values(income)

      Enum.each(rows, fn row -> assert row["upper"] - row["lower"] > 0 end)

      class_0 = Enum.filter(rows, &(&1["class"] == "0"))
      class_1 = Enum.filter(rows, &(&1["class"] == "1"))

      assert Enum.max(Enum.map(class_0, & &1["upper"])) <=
               Enum.min(Enum.map(class_1, & &1["lower"]))
    end

    test "makes each class sum to one by default" do
      [income | _] = panels(Verdict.feature_distribution(x(), labels(), bins: 2))

      assert_in_delta income |> of_class("0") |> Enum.sum(), 1.0, 1.0e-9
      assert_in_delta income |> of_class("1") |> Enum.sum(), 1.0, 1.0e-9
    end

    test "normalize: :none shows the raw counts" do
      [income | _] =
        panels(Verdict.feature_distribution(x(), labels(), bins: 2, normalize: :none))

      assert income |> of_class("0") |> Enum.sum() == 2
      assert income |> of_class("1") |> Enum.sum() == 2
    end

    test "names the classes when asked" do
      plot = Verdict.feature_distribution(x(), labels(), class_names: ["neg", "pos"])
      [income | _] = panels(plot)

      assert values(income) |> Enum.map(& &1["class"]) |> Enum.uniq() |> Enum.sort() ==
               ["neg", "pos"]
    end

    # A weight of two has to say the same thing as the row written twice.
    test "counts a weighted sample the way it counts a repeated one" do
      weighted =
        Verdict.feature_distribution(x(), labels(), sample_weights: [1, 1, 2, 1], bins: 2)

      repeated =
        Verdict.feature_distribution(
          Nx.tensor([[1.0, 5.0], [2.0, 5.5], [8.0, 1.0], [8.0, 1.0], [9.0, 1.5]]),
          Nx.tensor([0, 0, 1, 1, 1]),
          bins: 2
        )

      [w | _] = panels(weighted)
      [r | _] = panels(repeated)

      assert of_class(w, "1") == of_class(r, "1")
    end

    test "gives every panel the same colour scale, one class per colour" do
      plot = Verdict.feature_distribution(x(), labels())

      panels(plot)
      |> Enum.each(fn panel ->
        scale = panel["encoding"]["color"]["scale"]
        assert scale["domain"] == ["0", "1"]
        assert length(scale["range"]) == 2
      end)
    end

    test "shares one legend instead of repeating it per panel" do
      assert spec(Verdict.feature_distribution(x(), labels()))["resolve"] == %{
               "legend" => %{"color" => "shared"}
             }
    end

    test "wraps panels at the given column count" do
      assert spec(Verdict.feature_distribution(x(), labels(), columns: 1))["columns"] == 1
    end

    test "rejects x that is not a matrix" do
      assert_raise ArgumentError, ~r/rank-2 tensor/, fn ->
        Verdict.feature_distribution(Nx.tensor([1.0, 2.0]), labels())
      end
    end

    test "rejects labels that do not line up with the rows" do
      assert_raise ArgumentError, ~r/one entry per row of x/, fn ->
        Verdict.feature_distribution(x(), Nx.tensor([0, 1]))
      end
    end

    test "rejects the wrong number of feature names" do
      assert_raise ArgumentError, ~r/one entry per column of x/, fn ->
        Verdict.feature_distribution(x(), labels(), feature_names: ["only_one"])
      end
    end
  end
end
