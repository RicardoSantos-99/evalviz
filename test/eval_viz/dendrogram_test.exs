defmodule EvalViz.DendrogramTest do
  use ExUnit.Case, async: true

  # 1-D points in three well separated groups. scipy.cluster.hierarchy.linkage
  # on this data with single linkage produces
  #
  #   [[4, 5, 0.2, 2], [2, 3, 0.4, 2], [0, 1, 0.5, 2],
  #    [6, 9, 0.5, 3], [8, 10, 3.0, 5], [7, 11, 3.6, 7]]
  #
  # and its dendrogram puts the leaves in the order 4 5 2 3 6 0 1.
  defp data, do: Nx.tensor([[1.0], [1.5], [5.0], [5.4], [9.0], [9.2], [2.0]])

  defp model, do: Scholar.Cluster.Hierarchical.fit(data(), linkage: :single)

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]

  defp brackets(plot) do
    plot
    |> values()
    |> Enum.group_by(& &1["link"])
    |> Enum.sort_by(fn {link, _} -> link end)
    |> Enum.map(fn {_link, points} ->
      points = Enum.sort_by(points, & &1["order"])
      {Enum.map(points, & &1["x"]), Enum.map(points, &Float.round(&1["y"], 4))}
    end)
  end

  describe "dendrogram/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.dendrogram(model())
    end

    test "accepts a {clades, heights} pair as well as a model" do
      m = model()
      from_model = values(EvalViz.dendrogram(m))
      from_pair = values(EvalViz.dendrogram({m.clades, m.dissimilarities}))

      assert from_model == from_pair
    end

    test "orders leaves the way scipy does" do
      label_expr = spec(EvalViz.dendrogram(model()))["encoding"]["x"]["axis"]["labelExpr"]
      assert label_expr == "['4','5','2','3','6','0','1'][datum.value]"
    end

    test "draws the same brackets scipy does" do
      # scipy reports these as icoord/dcoord on a 5, 15, 25... leaf scale;
      # dividing by 10 and shifting by 0.5 gives the positions below.
      assert brackets(EvalViz.dendrogram(model())) == [
               {[0.0, 0.0, 1.0, 1.0], [0.0, 0.2, 0.2, 0.0]},
               {[2.0, 2.0, 3.0, 3.0], [0.0, 0.4, 0.4, 0.0]},
               {[5.0, 5.0, 6.0, 6.0], [0.0, 0.5, 0.5, 0.0]},
               {[4.0, 4.0, 5.5, 5.5], [0.0, 0.5, 0.5, 0.5]},
               {[2.5, 2.5, 4.75, 4.75], [0.4, 3.0, 3.0, 0.5]},
               {[0.5, 0.5, 3.625, 3.625], [0.2, 3.6, 3.6, 3.0]}
             ]
    end

    test "gives every merge its own line so the brackets stay separate" do
      plot = EvalViz.dendrogram(model())

      assert spec(plot)["encoding"]["detail"]["field"] == "link"
      assert plot |> values() |> Enum.map(& &1["link"]) |> Enum.uniq() |> length() == 6
      # four corners per bracket
      assert length(values(plot)) == 6 * 4
    end

    test "uses supplied labels, in leaf order rather than data order" do
      plot =
        EvalViz.dendrogram(model(),
          labels: ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
        )

      assert spec(plot)["encoding"]["x"]["axis"]["labelExpr"] ==
               "['echo','foxtrot','charlie','delta','golf','alpha','bravo'][datum.value]"
    end

    test "is a single colour when no threshold is given" do
      plot = EvalViz.dendrogram(model())
      assert spec(plot)["encoding"]["color"]["value"] == "#4c78a8"
    end

    test "color_threshold splits the tree into the clusters below the cut" do
      plot = EvalViz.dendrogram(model(), color_threshold: 1.0)

      clusters = plot |> values() |> Enum.map(& &1["cluster"]) |> Enum.uniq() |> Enum.sort()
      assert clusters == ["Above cut", "Cluster 1", "Cluster 2", "Cluster 3"]
    end

    test "paints the links above the cut neutral, not as another cluster" do
      plot = EvalViz.dendrogram(model(), color_threshold: 1.0)
      scale = spec(plot)["encoding"]["color"]["scale"]

      assert List.last(scale["domain"]) == "Above cut"
      assert List.last(scale["range"]) == "#b0b0b0"
      # the cut is at 1.0, so only the merges at 3.0 and 3.6 sit above it
      above = plot |> values() |> Enum.filter(&(&1["cluster"] == "Above cut"))
      assert above |> Enum.map(& &1["link"]) |> Enum.uniq() |> Enum.sort() == [4, 5]
    end

    test "a threshold above every merge leaves one cluster and nothing above it" do
      plot = EvalViz.dendrogram(model(), color_threshold: 10.0)
      clusters = plot |> values() |> Enum.map(& &1["cluster"]) |> Enum.uniq()

      assert clusters == ["Cluster 1"]
    end

    test "starts the height axis at zero so bar lengths stay comparable" do
      assert spec(EvalViz.dendrogram(model()))["encoding"]["y"]["scale"]["domainMin"] == 0
    end

    test "rejects labels that do not cover every point" do
      assert_raise ArgumentError, ~r/one entry per point/, fn ->
        EvalViz.dendrogram(model(), labels: ["a", "b"])
      end
    end

    test "rejects a mismatched clades and heights pair" do
      assert_raise ArgumentError, ~r/one height per merge/, fn ->
        EvalViz.dendrogram({Nx.tensor([[0, 1], [2, 3]]), Nx.tensor([0.5])})
      end
    end

    test "rejects a tree too small to draw" do
      assert_raise ArgumentError, ~r/at least 3 points/, fn ->
        EvalViz.dendrogram({Nx.tensor([[0, 1]]), Nx.tensor([0.5])})
      end
    end
  end
end
