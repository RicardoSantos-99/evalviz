defmodule Verdict.FoldScoresTest do
  use ExUnit.Case, async: true

  defp scores, do: Nx.tensor([[0.80, 0.86, 0.84], [1.20, 1.10, 1.15]], type: :f64)

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp panels(plot), do: spec(plot)["vconcat"]

  describe "fold_scores/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.fold_scores(scores())
    end

    test "draws a panel per metric" do
      assert length(panels(Verdict.fold_scores(scores()))) == 2
    end

    test "plots one point per fold" do
      [first, _second] = panels(Verdict.fold_scores(scores()))

      assert first["data"]["values"] |> Enum.map(& &1["fold"]) == [0, 1, 2]
      assert first["data"]["values"] |> Enum.map(& &1["score"]) == [0.80, 0.86, 0.84]
    end

    # The spread is the whole reason to look past the mean, so it is written
    # down as well as drawn.
    test "states the mean and deviation in each panel" do
      [first, second] = panels(Verdict.fold_scores(scores()))

      assert first["title"]["subtitle"] == "mean 0.8333 ± 0.0249"
      assert second["title"]["subtitle"] == "mean 1.15 ± 0.0408"
    end

    test "names the metrics" do
      plot = Verdict.fold_scores(scores(), metric_names: ["Accuracy", "Loss"])
      assert panels(plot) |> Enum.map(& &1["title"]["text"]) == ["Accuracy", "Loss"]
    end

    test "falls back to numbering the metrics" do
      assert panels(Verdict.fold_scores(scores())) |> Enum.map(& &1["title"]["text"]) ==
               ["Metric 0", "Metric 1"]
    end

    test "draws one chart for a single metric" do
      plot = Verdict.fold_scores(Nx.tensor([0.80, 0.86, 0.84], type: :f64))

      refute spec(plot)["vconcat"]
      assert spec(plot)["title"]["subtitle"] == "mean 0.8333 ± 0.0249"
    end

    test "marks the mean the folds scatter around" do
      [first, _] = panels(Verdict.fold_scores(scores()))
      [rule, points] = first["layer"]

      assert rule["mark"]["type"] == "rule"
      assert points["mark"]["type"] == "point"
      assert_in_delta hd(rule["data"]["values"])["mean"], 0.8333333333, 1.0e-9
    end

    test "mean_line: false leaves the mean out" do
      [first, _] = panels(Verdict.fold_scores(scores(), mean_line: false))
      assert length(first["layer"]) == 1
    end

    # Folds are interchangeable, so a line between them would show a trend
    # across an order that means nothing.
    test "does not join the folds" do
      [first, _] = panels(Verdict.fold_scores(scores()))
      assert first["layer"] |> Enum.map(& &1["mark"]["type"]) |> Enum.member?("line") == false
    end

    test "keeps the fold axis on whole numbers" do
      [first, _] = panels(Verdict.fold_scores(scores()))
      points = List.last(first["layer"])

      assert points["encoding"]["x"]["axis"]["tickMinStep"] == 1
    end

    test "lets the score axis start where the data does" do
      [first, _] = panels(Verdict.fold_scores(scores()))
      points = List.last(first["layer"])

      assert points["encoding"]["y"]["scale"]["zero"] == false
    end

    test "accepts lists as well as tensors" do
      from_lists = Verdict.fold_scores([[0.80, 0.86, 0.84], [1.20, 1.10, 1.15]])
      assert length(panels(from_lists)) == 2

      single = Verdict.fold_scores([0.80, 0.86, 0.84])
      refute spec(single)["vconcat"]
    end

    test "rejects scores of the wrong rank" do
      assert_raise ArgumentError, ~r/rank 1 or 2/, fn ->
        Verdict.fold_scores(Nx.broadcast(0.5, {2, 2, 2}))
      end
    end

    test "rejects the wrong number of metric names" do
      assert_raise ArgumentError, ~r/one entry per metric/, fn ->
        Verdict.fold_scores(scores(), metric_names: ["only one"])
      end
    end
  end
end
