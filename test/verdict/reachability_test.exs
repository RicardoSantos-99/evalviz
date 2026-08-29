defmodule Verdict.ReachabilityTest do
  use ExUnit.Case, async: true

  # Expected values from scikit-learn 1.6.1 on this data.
  defp x, do: Nx.tensor([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], type: :f64)

  defp model(opts \\ [eps: 4.5, min_samples: 2]), do: Scholar.Cluster.OPTICS.fit(x(), opts)

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp layers(plot), do: spec(plot)["layer"]
  defp bars(plot), do: layers(plot) |> hd() |> get_in(["data", "values"])

  defp marks(plot) do
    layers(plot) |> Enum.map(&{&1["mark"]["type"], length(&1["data"]["values"])})
  end

  describe "reachability/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.reachability(model())
    end

    # Both arrays are stored in the order the points were given, and the plot
    # only means anything in the order OPTICS reached them.
    test "puts the points in the order OPTICS visited them" do
      assert bars(Verdict.reachability(model())) |> Enum.map(& &1["point"]) == [1, 2, 5, 3, 4]
    end

    test "reads each point's distance through that same ordering" do
      distances = bars(Verdict.reachability(model())) |> Enum.map(& &1["reachability"])

      Enum.zip(distances, [3.16227766, 1.41421356, 5.0, 4.12310563, 1.0])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "lays the bars out edge to edge along the ordering" do
      rows = bars(Verdict.reachability(model()))

      assert Enum.map(rows, & &1["position"]) == [1, 2, 3, 4, 5]
      assert Enum.map(rows, & &1["position_end"]) == [2, 3, 4, 5, 6]
    end

    # An unreachable point has no place on the axis, so it is drawn as a rule
    # rather than given a number it does not have.
    test "draws a point it could not reach as a rule instead of a bar" do
      assert marks(Verdict.reachability(model())) == [{"bar", 5}, {"rule", 1}, {"rule", 1}]

      [wall] = layers(Verdict.reachability(model())) |> Enum.at(1) |> get_in(["data", "values"])
      assert wall["point"] == 0
      assert wall["reachability"] == nil
    end

    test "draws every wall when a max_eps leaves several points unreached" do
      assert marks(Verdict.reachability(model(max_eps: 2, min_samples: 2))) ==
               [{"bar", 2}, {"rule", 4}, {"rule", 1}]
    end

    test "leaves room above the tallest bar it can draw" do
      [_, top] =
        layers(Verdict.reachability(model()))
        |> hd()
        |> get_in(["encoding", "y", "scale", "domain"])

      assert_in_delta top, 5.0 * 1.05, 1.0e-9
    end

    test "spans the whole ordering on the x axis, walls included" do
      domain =
        layers(Verdict.reachability(model()))
        |> hd()
        |> get_in(["encoding", "x", "scale", "domain"])

      assert domain == [0, 6]
    end

    test "names the clusters when asked, and never renames noise" do
      plot =
        Verdict.reachability(model(max_eps: 2, min_samples: 2), label_names: ["left", "right"])

      clusters = bars(plot) |> Enum.map(& &1["cluster"])

      assert clusters == ["left", "right"]

      walls = layers(plot) |> Enum.at(1) |> get_in(["data", "values"])
      assert walls |> Enum.map(& &1["cluster"]) |> Enum.uniq() == ["noise", "left", "right"]
    end

    # Ordered by cluster rather than by where each first turns up, and noise
    # last, since it is what everything else was picked out of.
    test "greys noise out rather than spending a cluster colour on it" do
      scale =
        layers(Verdict.reachability(model(max_eps: 2, min_samples: 2)))
        |> hd()
        |> get_in(["encoding", "color", "scale"])

      assert scale["domain"] == ["0", "1", "noise"]
      assert scale["range"] |> List.last() == Verdict.Theme.muted()
      refute Verdict.Theme.muted() in Enum.drop(scale["range"], -1)
    end

    test "draws the distance the model extracted its clusters at" do
      [rule] = layers(Verdict.reachability(model())) |> List.last() |> get_in(["data", "values"])
      assert rule["cut_off"] == 4.5
    end

    test "takes a cut-off of its own, which is how a different one is chosen" do
      [rule] =
        layers(Verdict.reachability(model(), cut_off: 2.0))
        |> List.last()
        |> get_in(["data", "values"])

      assert rule["cut_off"] == 2.0
    end

    test "leaves the cut-off out when told to" do
      assert marks(Verdict.reachability(model(), cut_off: false)) == [{"bar", 5}, {"rule", 1}]
    end

    # eps defaults to max_eps, which defaults to infinity, and an infinite
    # cut-off has no position on the axis either.
    test "leaves an infinite cut-off out on its own" do
      assert marks(Verdict.reachability(model(min_samples: 2))) == [{"bar", 5}, {"rule", 1}]
    end

    test "still draws a model that reached nothing" do
      plot = Verdict.reachability(model(min_samples: 2))
      scale = layers(plot) |> hd() |> get_in(["encoding", "color", "scale"])

      assert scale["domain"] == ["noise"]
      assert scale["range"] == [Verdict.Theme.muted()]
    end

    test "rejects anything that is not a fitted OPTICS model" do
      assert_raise ArgumentError, ~r/Scholar.Cluster.OPTICS/, fn ->
        Verdict.reachability(%{labels: Nx.tensor([0, 1])})
      end
    end
  end
end
