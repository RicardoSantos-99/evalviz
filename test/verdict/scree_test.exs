defmodule Verdict.ScreeTest do
  use ExUnit.Case, async: true

  defp ratios, do: Nx.tensor([0.6, 0.25, 0.1, 0.05])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]

  describe "scree/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.scree(ratios())
    end

    test "numbers components from one" do
      assert values(Verdict.scree(ratios())) |> Enum.map(& &1["component"]) == [1, 2, 3, 4]
    end

    test "keeps each component's own share" do
      assert values(Verdict.scree(ratios()))
             |> Enum.map(& &1["ratio"])
             |> Enum.zip([0.6, 0.25, 0.1, 0.05])
             |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "accumulates the running total up to one" do
      cumulative = values(Verdict.scree(ratios())) |> Enum.map(& &1["cumulative"])

      Enum.zip(cumulative, [0.6, 0.85, 0.95, 1.0])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "accepts a PCA model as well as a tensor" do
      model = %{explained_variance_ratio: ratios()}
      assert values(Verdict.scree(model)) == values(Verdict.scree(ratios()))
    end

    test "layers the running total over the bars" do
      [bars, line] = spec(Verdict.scree(ratios()))["layer"]

      assert bars["mark"]["type"] == "bar"
      assert bars["encoding"]["y"]["field"] == "ratio"
      assert line["mark"]["type"] == "line"
      assert line["encoding"]["y"]["field"] == "cumulative"
    end

    test "cumulative: false leaves only the bars" do
      assert length(spec(Verdict.scree(ratios(), cumulative: false))["layer"]) == 1
    end

    test "labels the variance axis as a percentage" do
      [bars, _line] = spec(Verdict.scree(ratios()))["layer"]
      assert bars["encoding"]["y"]["axis"]["format"] == ".0%"
    end

    test "handles a single component" do
      plot = Verdict.scree(Nx.tensor([1.0]))

      assert [%{"component" => 1, "cumulative" => 1.0, "ratio" => 1.0}] = values(plot)
    end
  end
end
