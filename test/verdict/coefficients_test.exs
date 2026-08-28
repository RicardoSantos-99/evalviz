defmodule Verdict.CoefficientsTest do
  use ExUnit.Case, async: true

  defp flat, do: Nx.tensor([0.2, -1.5, 0.9, -0.1], type: :f64)
  defp per_class, do: Nx.tensor([[0.2, -0.2], [-1.5, 1.5], [0.9, -0.9]], type: :f64)

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp bars(plot), do: spec(plot)["layer"] |> hd()
  defp order(plot), do: bars(plot)["encoding"]["y"]["sort"]

  describe "coefficients/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.coefficients(flat())
    end

    test "draws one bar per feature" do
      assert length(values(Verdict.coefficients(flat()))) == 4
    end

    # What matters is how far a coefficient moves the prediction, not which way.
    test "orders by magnitude by default" do
      assert order(Verdict.coefficients(flat())) == ["1", "2", "0", "3"]
    end

    test "sort: :value orders by the signed coefficient" do
      assert order(Verdict.coefficients(flat(), sort: :value)) == ["2", "0", "3", "1"]
    end

    test "sort: :none keeps the feature order" do
      assert order(Verdict.coefficients(flat(), sort: :none)) == ["0", "1", "2", "3"]
    end

    test "names the features" do
      plot = Verdict.coefficients(flat(), feature_names: ["a", "b", "c", "d"])
      assert order(plot) == ["b", "c", "a", "d"]
    end

    test "top: n keeps only the strongest, and drops their rows too" do
      plot = Verdict.coefficients(flat(), top: 2)

      assert order(plot) == ["1", "2"]
      assert values(plot) |> Enum.map(& &1["feature"]) == ["1", "2"]
    end

    # Coefficients are signed, so the bars need the line they grow from.
    test "draws the zero line" do
      [_bars, zero] = spec(Verdict.coefficients(flat()))["layer"]

      assert zero["mark"]["type"] == "rule"
      assert zero["data"]["values"] == [%{"zero" => 0}]
    end

    test "reads a model that exposes its coefficients" do
      x = Nx.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
      y = Nx.tensor([2.0, 4.0, 6.0, 8.0])
      model = Scholar.Linear.LinearRegression.fit(x, y)

      assert length(values(Verdict.coefficients(model))) == 2
    end

    test "rejects coefficients of the wrong rank" do
      assert_raise ArgumentError, ~r/rank 1 or 2/, fn ->
        Verdict.coefficients(Nx.broadcast(0.5, {2, 2, 2}))
      end
    end
  end

  describe "a coefficient per class" do
    test "draws a bar per feature per class" do
      assert length(values(Verdict.coefficients(per_class()))) == 6
    end

    test "names the classes" do
      plot = Verdict.coefficients(per_class())
      assert values(plot) |> Enum.map(& &1["class"]) |> Enum.uniq() == ["Class 0", "Class 1"]

      named = Verdict.coefficients(per_class(), class_names: ["cat", "dog"])
      assert values(named) |> Enum.map(& &1["class"]) |> Enum.uniq() == ["cat", "dog"]
    end

    test "groups the bars rather than overlaying them" do
      assert bars(Verdict.coefficients(per_class()))["encoding"]["yOffset"]["field"] == "class"
    end

    # A feature is only as important as its strongest class.
    test "ranks a feature by its largest coefficient across the classes" do
      assert order(Verdict.coefficients(per_class())) == ["1", "2", "0"]
    end

    test "rejects the wrong number of class names" do
      assert_raise ArgumentError, ~r/one entry per class/, fn ->
        Verdict.coefficients(per_class(), class_names: ["only one"])
      end
    end
  end
end
