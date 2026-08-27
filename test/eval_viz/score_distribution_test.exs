defmodule EvalViz.ScoreDistributionTest do
  use ExUnit.Case, async: true

  defp y_true, do: Nx.tensor([0, 0, 0, 0, 1, 1])
  defp scores, do: Nx.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], type: :f64)

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp bars(plot), do: spec(plot)["layer"] |> hd()

  defp of_class(plot, name) do
    plot |> values() |> Enum.filter(&(&1["class"] == name)) |> Enum.map(& &1["value"])
  end

  describe "score_distribution/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.score_distribution(y_true(), scores())
    end

    test "draws a series per true class" do
      assert EvalViz.score_distribution(y_true(), scores(), bins: 2)
             |> values()
             |> Enum.map(& &1["class"])
             |> Enum.uniq() == ["0", "1"]
    end

    # Four negatives and two positives: raw counts would make the positive hump
    # look half the size whatever its shape.
    test "makes each class sum to one by default" do
      plot = EvalViz.score_distribution(y_true(), scores(), bins: 2)

      assert_in_delta plot |> of_class("0") |> Enum.sum(), 1.0, 1.0e-9
      assert_in_delta plot |> of_class("1") |> Enum.sum(), 1.0, 1.0e-9
    end

    test "normalize: :none shows the raw counts" do
      plot = EvalViz.score_distribution(y_true(), scores(), bins: 2, normalize: :none)

      assert plot |> of_class("0") |> Enum.sum() == 4
      assert plot |> of_class("1") |> Enum.sum() == 2
    end

    test "labels the y axis for the normalization in force" do
      shared = EvalViz.score_distribution(y_true(), scores(), bins: 2)
      raw = EvalViz.score_distribution(y_true(), scores(), bins: 2, normalize: :none)

      assert bars(shared)["encoding"]["y"]["title"] == "Share of class"
      assert bars(raw)["encoding"]["y"]["title"] == "Count"
    end

    # Given only x and x2, a bar has no vertical extent and renders as a dash
    # floating at the count. Both axes have to be told where the bar runs.
    test "gives the bars an extent on both axes" do
      encoding = bars(EvalViz.score_distribution(y_true(), scores()))["encoding"]

      assert encoding["x"]["field"] == "lower"
      assert encoding["x2"]["field"] == "upper"
      assert encoding["y"]["field"] == "zero"
      assert encoding["y2"]["field"] == "value"
    end

    # Comparing the shapes only works if the bars line up.
    test "shares one set of bin edges across the classes" do
      plot = EvalViz.score_distribution(y_true(), scores(), bins: 4)

      edges = plot |> values() |> Enum.map(& &1["lower"]) |> Enum.uniq() |> Enum.sort()
      assert length(edges) <= 4

      Enum.zip(edges, [0.0, 0.25, 0.5, 0.75])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-9 end)
    end

    # Stacked bars would hide exactly the overlap the plot is read for, so
    # every bar starts at zero rather than on top of the class before it.
    test "overlays the classes instead of stacking them" do
      rows = values(EvalViz.score_distribution(y_true(), scores()))
      assert rows |> Enum.map(& &1["zero"]) |> Enum.uniq() == [0]
    end

    test "names the classes when asked" do
      plot = EvalViz.score_distribution(y_true(), scores(), class_names: ["no", "yes"])
      assert plot |> values() |> Enum.map(& &1["class"]) |> Enum.uniq() == ["no", "yes"]
    end

    test "takes more than two classes" do
      plot =
        EvalViz.score_distribution(
          Nx.tensor([0, 1, 2, 0, 1, 2]),
          Nx.tensor([0.1, 0.5, 0.9, 0.2, 0.6, 0.8], type: :f64)
        )

      assert plot |> values() |> Enum.map(& &1["class"]) |> Enum.uniq() == ["0", "1", "2"]
    end

    test "marks a threshold when given one" do
      plot = EvalViz.score_distribution(y_true(), scores(), threshold: 0.55)
      [_bars, rule] = spec(plot)["layer"]

      assert rule["mark"]["type"] == "rule"
      assert rule["data"]["values"] == [%{"threshold" => 0.55}]
    end

    test "draws no rule without a threshold" do
      assert length(spec(EvalViz.score_distribution(y_true(), scores()))["layer"]) == 1
    end

    test "rejects inputs of different lengths" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        EvalViz.score_distribution(y_true(), Nx.tensor([0.1, 0.2]))
      end
    end
  end
end
