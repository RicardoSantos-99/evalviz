defmodule EvalViz.LoadingsTest do
  use ExUnit.Case, async: true

  defp components, do: Nx.tensor([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0]])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp heatmap(plot), do: spec(plot)["layer"] |> hd()

  describe "loadings/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.loadings(components())
    end

    test "draws one cell per component and feature" do
      assert length(values(EvalViz.loadings(components()))) == 6
    end

    test "lays components out as rows and features as columns" do
      plot = EvalViz.loadings(components())

      assert values(plot) |> Enum.map(& &1["component"]) |> Enum.uniq() ==
               ["Component 0", "Component 1"]

      assert values(plot) |> Enum.map(& &1["feature"]) |> Enum.uniq() == ["0", "1", "2"]
    end

    test "keeps each loading with its component" do
      first =
        EvalViz.loadings(components())
        |> values()
        |> Enum.filter(&(&1["component"] == "Component 0"))
        |> Enum.map(& &1["value"])

      Enum.zip(first, [0.6, -0.8, 0.0])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    # A loading's sign says which way the feature pushes, so an off-centre scale
    # would colour a mild negative like a strong positive.
    test "centres the colour scale on zero" do
      [min, max] = heatmap(EvalViz.loadings(components()))["encoding"]["color"]["scale"]["domain"]

      assert_in_delta min, -0.8, 1.0e-6
      assert_in_delta max, 0.8, 1.0e-6
    end

    test "uses a diverging scheme, not a sequential one" do
      assert heatmap(EvalViz.loadings(components()))["encoding"]["color"]["scale"]["scheme"] ==
               EvalViz.Theme.diverging_scheme()
    end

    test "names features and components when asked" do
      plot =
        EvalViz.loadings(components(),
          feature_names: ["height", "weight", "age"],
          component_names: ["size", "shape"]
        )

      assert values(plot) |> Enum.map(& &1["feature"]) |> Enum.uniq() ==
               ["height", "weight", "age"]

      assert values(plot) |> Enum.map(& &1["component"]) |> Enum.uniq() == ["size", "shape"]
    end

    test "writes the value in each cell" do
      assert EvalViz.loadings(components()) |> values() |> Enum.map(& &1["label"]) |> hd() ==
               "0.6"

      assert length(spec(EvalViz.loadings(components()))["layer"]) == 2
    end

    test "values: false leaves the numbers out" do
      assert length(spec(EvalViz.loadings(components(), values: false))["layer"]) == 1
    end

    test "reads a model that exposes its components" do
      x = Nx.tensor([[1.0, 0.0], [3.0, 1.0], [5.0, 0.0], [7.0, 1.0]])
      pca = Scholar.Decomposition.PCA.fit(x, num_components: 2)

      assert length(values(EvalViz.loadings(pca))) == 4
    end

    test "rejects loadings that are not a matrix" do
      assert_raise ArgumentError, ~r/rank-2 tensor/, fn ->
        EvalViz.loadings(Nx.tensor([0.1, 0.2]))
      end
    end

    test "rejects the wrong number of component names" do
      assert_raise ArgumentError, ~r/one entry per component/, fn ->
        EvalViz.loadings(components(), component_names: ["only one"])
      end
    end
  end
end
