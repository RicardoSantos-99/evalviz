defmodule EvalViz.SilhouetteTest do
  use ExUnit.Case, async: true

  # Same fixture as Scholar.Metrics.Clustering.silhouette_samples/3's doctest,
  # which returns [0.76477534, 0.7781199, 0.67543036, 0.4934419, 0.66279924].
  defp x, do: Nx.tensor([[0, 0], [1, 0], [1, 1], [3, 3], [4, 4.5]])
  defp labels, do: Nx.tensor([0, 0, 0, 1, 1])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp bars_layer(plot), do: spec(plot)["layer"] |> hd()

  describe "silhouette/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.silhouette(x(), labels(), num_clusters: 2)
    end

    test "draws one bar per point, carrying the score Scholar computes" do
      plot = EvalViz.silhouette(x(), labels(), num_clusters: 2)
      scores = plot |> values() |> Enum.map(& &1["score"]) |> Enum.sort()

      assert length(values(plot)) == 5

      Enum.zip(scores, [0.4934419, 0.66279924, 0.67543036, 0.76477534, 0.7781199])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "sorts descending within each cluster" do
      plot = EvalViz.silhouette(x(), labels(), num_clusters: 2)

      by_cluster =
        plot
        |> values()
        |> Enum.group_by(& &1["cluster"])
        |> Map.new(fn {cluster, rows} ->
          {cluster, rows |> Enum.sort_by(& &1["position"]) |> Enum.map(& &1["score"])}
        end)

      for {_cluster, scores} <- by_cluster do
        assert scores == Enum.sort(scores, :desc)
      end
    end

    test "leaves a gap between clusters so the blocks stay separate" do
      plot = EvalViz.silhouette(x(), labels(), num_clusters: 2)
      positions = plot |> values() |> Enum.map(& &1["position"]) |> Enum.sort()

      # cluster 0 takes rows 0..2, then a gap, then cluster 1
      assert positions == [0, 1, 2, 5, 6]
    end

    test "gives each bar an explicit extent on both axes" do
      # with two quantitative axes the bar has no implicit baseline, so x2/y2
      # are what make it a horizontal bar rather than a point
      layer = bars_layer(EvalViz.silhouette(x(), labels(), num_clusters: 2))

      assert layer["encoding"]["x"]["field"] == "zero"
      assert layer["encoding"]["x2"]["field"] == "score"
      assert layer["encoding"]["y"]["field"] == "position"
      assert layer["encoding"]["y2"]["field"] == "position_end"
    end

    test "draws the mean score as a reference" do
      plot = EvalViz.silhouette(x(), labels(), num_clusters: 2)
      [_bars, mean_layer] = spec(plot)["layer"]

      assert [%{"mean" => mean}] = mean_layer["data"]["values"]
      expected = (0.76477534 + 0.7781199 + 0.67543036 + 0.4934419 + 0.66279924) / 5
      assert_in_delta mean, expected, 1.0e-6
    end

    test "average_line: false drops the reference" do
      plot = EvalViz.silhouette(x(), labels(), num_clusters: 2, average_line: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "keeps room for negative scores instead of clipping at zero" do
      # a point put in the wrong cluster scores below zero
      bad_labels = Nx.tensor([0, 1, 0, 1, 0])
      layer = bars_layer(EvalViz.silhouette(x(), bad_labels, num_clusters: 2))

      [low, high] = layer["encoding"]["x"]["scale"]["domain"]
      assert low < 0
      assert high == 1.0
    end

    test "names clusters when asked" do
      plot =
        EvalViz.silhouette(x(), labels(), num_clusters: 2, cluster_names: ["near", "far"])

      clusters = plot |> values() |> Enum.map(& &1["cluster"]) |> Enum.uniq() |> Enum.sort()
      assert clusters == ["far", "near"]
    end

    test "rejects cluster_names that do not cover every cluster" do
      assert_raise ArgumentError, ~r/one entry per cluster/, fn ->
        EvalViz.silhouette(x(), labels(), num_clusters: 2, cluster_names: ["only one"])
      end
    end

    test "requires num_clusters" do
      assert_raise NimbleOptions.ValidationError, fn ->
        EvalViz.silhouette(x(), labels())
      end
    end
  end
end
