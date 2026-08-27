defmodule EvalViz.BiplotTest do
  use ExUnit.Case, async: true

  defmodule NoTransform do
    defstruct [:components]
  end

  # Features 0 and 2 load on the first component, feature 1 on the second. Every
  # loading vector is unit length, and the widest point sits at radius 2, so the
  # default 0.9 share puts every arrow tip at 1.8.
  defp loadings, do: Nx.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
  defp projection, do: Nx.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp layer(plot, index), do: spec(plot)["layer"] |> Enum.at(index)
  defp points(plot), do: layer(plot, 0)["data"]["values"]
  defp arrows(plot), do: layer(plot, 1)["data"]["values"]

  defp plot(opts \\ []), do: EvalViz.biplot({projection(), loadings()}, opts)

  describe "biplot/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = plot()
    end

    test "draws points, arrows and arrow labels" do
      assert length(spec(plot())["layer"]) == 3
    end

    test "plots one point per row of the projection" do
      assert points(plot()) |> Enum.map(& &1["x"]) == [2.0, -2.0, 0.0, 0.0]
    end

    # A loading belongs to a component, which is a row. Reading columns instead
    # would give one arrow per component rather than per feature.
    test "draws one arrow per feature, taken from the component rows" do
      arrows = arrows(plot())

      assert length(arrows) == 3
      assert Enum.map(arrows, & &1["feature"]) == ["0", "1", "2"]
    end

    test "points each arrow the way its feature loads" do
      [a, b, c] = arrows(plot())

      assert_in_delta a["x"], 1.8, 1.0e-9
      assert_in_delta a["y"], 0.0, 1.0e-9

      assert_in_delta b["x"], 0.0, 1.0e-9
      assert_in_delta b["y"], 1.8, 1.0e-9

      # feature 2 loads exactly like feature 0, so the arrows coincide
      assert a["x"] == c["x"] and a["y"] == c["y"]
    end

    test "starts every arrow at the origin" do
      assert arrows(plot()) |> Enum.map(& &1["zero"]) |> Enum.uniq() == [0]
    end

    test "scales the arrows against the point cloud" do
      far = Nx.tensor([[20.0, 0.0], [-20.0, 0.0], [0.0, 20.0], [0.0, -20.0]])
      [a | _] = EvalViz.biplot({far, loadings()}) |> arrows()

      assert_in_delta a["x"], 18.0, 1.0e-9
    end

    test "takes a different share when asked" do
      [a | _] = plot(arrow_scale: 0.5) |> arrows()
      assert_in_delta a["x"], 1.0, 1.0e-9
    end

    test "names the arrows" do
      arrows = plot(feature_names: ["height", "weight", "height_again"]) |> arrows()
      assert Enum.map(arrows, & &1["feature"]) == ["height", "weight", "height_again"]
    end

    # Loadings run either side of zero, so an off-centre origin would misplace
    # every arrow relative to the cloud.
    test "centres both axes on the origin" do
      [min, max] = layer(plot(), 0)["encoding"]["x"]["scale"]["domain"]

      assert_in_delta min, -max, 1.0e-9
      assert layer(plot(), 0)["encoding"]["y"]["scale"]["domain"] == [min, max]
    end

    test "colours the points when given labels" do
      plot = plot(labels: Nx.tensor([0, 0, 1, 1]), label_names: ["a", "b"])

      assert points(plot) |> Enum.map(& &1["label"]) == ["a", "a", "b", "b"]
      assert layer(plot, 0)["encoding"]["color"]["field"] == "label"
    end

    test "leaves colour alone without labels" do
      refute layer(plot(), 0)["encoding"]["color"]
    end

    test "derives the points from a model that can transform" do
      x = Nx.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]])
      pca = Scholar.Decomposition.PCA.fit(x, num_components: 2)

      assert length(EvalViz.biplot(pca, x) |> points()) == 4
    end

    test "says so when the model cannot transform" do
      assert_raise ArgumentError, ~r/has no transform\/2/, fn ->
        EvalViz.biplot(%NoTransform{components: loadings()}, projection())
      end
    end

    test "rejects loadings that are not a matrix" do
      assert_raise ArgumentError, ~r/rank-2 tensor/, fn ->
        EvalViz.biplot({projection(), Nx.tensor([1.0, 2.0])})
      end
    end

    test "rejects a component the projection does not have" do
      assert_raise ArgumentError, ~r/index the projection's 2 columns/, fn ->
        plot(components: [0, 5])
      end
    end

    test "rejects a component the loadings do not have" do
      wide = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])

      assert_raise ArgumentError, ~r/index the loadings' 2 rows/, fn ->
        EvalViz.biplot({wide, loadings()}, components: [0, 3])
      end
    end

    test "rejects labels that do not line up with the projection" do
      assert_raise ArgumentError, ~r/one entry per row/, fn ->
        plot(labels: Nx.tensor([0, 1]))
      end
    end
  end
end
