defmodule Verdict.TargetingTest do
  use ExUnit.Case, async: true

  defp y_true, do: Nx.tensor([0, 0, 1, 1, 0, 1])
  defp scores, do: Nx.tensor([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp field(plot, name), do: values(plot) |> Enum.map(& &1[name])

  defp close(got, want) do
    assert length(got) == length(want)
    Enum.zip(got, want) |> Enum.each(fn {g, w} -> assert_in_delta g, w, 1.0e-6 end)
  end

  # Reference values computed in numpy from the definition: sort by score
  # descending, group ties, accumulate weight contacted and positives captured.
  describe "cumulative_gain/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.cumulative_gain(y_true(), scores())
    end

    test "draws the share of samples contacted against positives captured" do
      plot = Verdict.cumulative_gain(y_true(), scores())

      close(field(plot, "contacted"), [0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0])
      close(field(plot, "captured"), [0.0, 1 / 3, 2 / 3, 2 / 3, 1.0, 1.0, 1.0])
    end

    test "starts at nobody contacted and ends at everybody captured" do
      plot = Verdict.cumulative_gain(y_true(), scores())

      assert field(plot, "contacted") |> List.first() == 0.0
      assert field(plot, "captured") |> List.first() == 0.0
      assert field(plot, "contacted") |> List.last() == 1.0
      assert field(plot, "captured") |> List.last() == 1.0
    end

    test "draws the diagonal a random ordering would trace" do
      [reference | _] = spec(Verdict.cumulative_gain(y_true(), scores()))["layer"]

      assert reference["data"]["values"] == [%{"x" => 0, "y" => 0}, %{"x" => 1, "y" => 1}]
      assert reference["mark"]["strokeDash"] == [4, 4]
    end

    test "leaves the diagonal out when told to" do
      plot = Verdict.cumulative_gain(y_true(), scores(), chance_line: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "holds both axes to a share" do
      [_, curve] = spec(Verdict.cumulative_gain(y_true(), scores()))["layer"]

      assert curve["encoding"]["x"]["scale"]["domain"] == [0, 1]
      assert curve["encoding"]["y"]["scale"]["domain"] == [0, 1]
    end
  end

  describe "lift_curve/3" do
    test "draws the gain divided by the diagonal it is read against" do
      plot = Verdict.lift_curve(y_true(), scores())

      close(field(plot, "contacted"), [1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0])
      close(field(plot, "lift"), [2.0, 2.0, 4 / 3, 1.5, 1.2, 1.0])
    end

    # Contacting nobody captures a share of nothing, so the curve has to start
    # at the first contact rather than at the origin the gain curve has.
    test "leaves out the point where nobody has been contacted" do
      refute 0.0 in field(Verdict.lift_curve(y_true(), scores()), "contacted")
    end

    test "ends at no skill, since contacting everybody captures everybody" do
      assert field(Verdict.lift_curve(y_true(), scores()), "lift") |> List.last() == 1.0
    end

    test "draws no skill as a flat one rather than as a diagonal" do
      [reference | _] = spec(Verdict.lift_curve(y_true(), scores()))["layer"]
      assert reference["data"]["values"] == [%{"x" => 0, "y" => 1}, %{"x" => 1, "y" => 1}]
    end

    # Lift starts well above one and has no ceiling, so pinning it to a share
    # would cut the part of the curve worth reading.
    test "holds the share contacted to a unit range and lets lift run free" do
      [_, curve] = spec(Verdict.lift_curve(y_true(), scores()))["layer"]

      assert curve["encoding"]["x"]["scale"]["domain"] == [0, 1]
      refute curve["encoding"]["y"]["scale"]
    end
  end

  describe "shared with the other curves" do
    test "compares models on one pair of axes" do
      other = Nx.tensor([0.9, 0.7, 0.2, 0.4, 0.6, 0.1])

      plot =
        Verdict.cumulative_gain([
          {"Good", y_true(), scores()},
          {"Bad", y_true(), other}
        ])

      assert field(plot, "series") |> Enum.uniq() == ["Good", "Bad"]
    end

    test "draws one one-vs-rest curve per class" do
      y = Nx.tensor([0, 1, 2, 0, 1, 2])

      score =
        Nx.tensor([
          [0.7, 0.2, 0.1],
          [0.2, 0.6, 0.2],
          [0.1, 0.3, 0.6],
          [0.6, 0.3, 0.1],
          [0.3, 0.5, 0.2],
          [0.2, 0.2, 0.6]
        ])

      plot = Verdict.lift_curve(y, score, class_names: ["cat", "dog", "bird"])
      assert field(plot, "series") |> Enum.uniq() == ["cat", "dog", "bird"]
    end

    # A weight of two has to say the same thing as the same row written twice.
    test "counts a weighted sample the way it counts a repeated one" do
      weighted =
        Verdict.cumulative_gain(y_true(), scores(), sample_weights: [1, 1, 2, 1, 1, 1])

      repeated =
        Verdict.cumulative_gain(
          Nx.tensor([0, 0, 1, 1, 1, 0, 1]),
          Nx.tensor([0.1, 0.4, 0.35, 0.35, 0.8, 0.2, 0.9])
        )

      close(field(weighted, "contacted"), field(repeated, "contacted"))
      close(field(weighted, "captured"), field(repeated, "captured"))
    end

    test "puts each class in its own panel when asked" do
      y = Nx.tensor([0, 1, 2, 0, 1, 2])

      score =
        Nx.tensor([
          [0.7, 0.2, 0.1],
          [0.2, 0.6, 0.2],
          [0.1, 0.3, 0.6],
          [0.6, 0.3, 0.1],
          [0.3, 0.5, 0.2],
          [0.2, 0.2, 0.6]
        ])

      plot = Verdict.cumulative_gain(y, score, facet: true)
      assert spec(plot)["facet"]["field"] == "series"
    end

    test "rejects a y_true that is not binary" do
      assert_raise ArgumentError, ~r/only 0 and 1/, fn ->
        Verdict.lift_curve(Nx.tensor([0, 1, 2, 1]), Nx.tensor([0.1, 0.2, 0.3, 0.4]))
      end
    end
  end
end
