defmodule Verdict.ElbowTest do
  use ExUnit.Case, async: true

  defp ks, do: [2, 3, 4, 5, 6]
  defp inertias, do: [500.0, 120.0, 100.0, 90.0, 85.0]

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp rule(plot), do: spec(plot)["layer"] |> hd()

  describe "elbow/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.elbow(ks(), inertias())
    end

    test "plots one point per k" do
      assert values(Verdict.elbow(ks(), inertias())) |> Enum.map(& &1["k"]) == ks()
    end

    test "finds the corner" do
      assert rule(Verdict.elbow(ks(), inertias()))["data"]["values"] == [%{"k" => 3}]
      assert spec(Verdict.elbow(ks(), inertias()))["title"]["subtitle"] == "elbow at k = 3"
    end

    # k spans single digits and inertia spans hundreds, so distance measured on
    # the raw axes would be decided entirely by the inertia.
    test "finds the same corner when the score is rescaled" do
      scaled = Enum.map(inertias(), &(&1 * 1000))
      assert rule(Verdict.elbow(ks(), scaled))["data"]["values"] == [%{"k" => 3}]
    end

    test "mark_elbow: false leaves the rule and subtitle out" do
      plot = Verdict.elbow(ks(), inertias(), mark_elbow: false)

      assert length(spec(plot)["layer"]) == 1
      refute spec(plot)["title"]
    end

    test "marks nothing when there are too few points to have a corner" do
      plot = Verdict.elbow([2, 3], [500.0, 120.0])

      assert length(spec(plot)["layer"]) == 1
      refute spec(plot)["title"]
    end

    test "marks nothing when every score is the same" do
      plot = Verdict.elbow(ks(), [10.0, 10.0, 10.0, 10.0, 10.0])
      assert length(spec(plot)["layer"]) == 1
    end

    # A straight run has no corner, so naming one would invent a result.
    test "marks nothing when the scores fall in a straight line" do
      plot = Verdict.elbow(ks(), [500.0, 400.0, 300.0, 200.0, 100.0])

      assert length(spec(plot)["layer"]) == 1
      refute spec(plot)["title"]
    end

    test "accepts tensors" do
      from_tensors = values(Verdict.elbow(Nx.tensor(ks()), Nx.tensor(inertias(), type: :f64)))
      assert Enum.map(from_tensors, & &1["k"]) == ks()
    end

    test "reads k and the score off fitted models" do
      key = Nx.Random.key(42)
      {x, _key} = Nx.Random.normal(key, 0.0, 1.0, shape: {30, 2})

      models = Enum.map(2..4, &Scholar.Cluster.KMeans.fit(x, num_clusters: &1))
      plot = Verdict.elbow(models)

      assert values(plot) |> Enum.map(& &1["k"]) == [2, 3, 4]
      assert values(plot) |> Enum.map(& &1["value"]) |> Enum.all?(&is_float/1)
    end

    test "lets the score axis start where the data does" do
      layer = spec(Verdict.elbow(ks(), inertias()))["layer"] |> List.last()
      assert layer["encoding"]["y"]["scale"]["zero"] == false
    end

    test "renames the axes" do
      plot = Verdict.elbow(ks(), inertias(), metric_title: "Distortion", k_title: "Groups")
      layer = spec(plot)["layer"] |> List.last()

      assert layer["encoding"]["y"]["title"] == "Distortion"
      assert layer["encoding"]["x"]["title"] == "Groups"
    end

    test "rejects scores that do not line up with k" do
      assert_raise ArgumentError, ~r/one score per k/, fn ->
        Verdict.elbow(ks(), [1.0, 2.0])
      end
    end

    test "rejects an empty plot" do
      assert_raise ArgumentError, ~r/at least one k/, fn ->
        Verdict.elbow([], [])
      end
    end
  end
end
