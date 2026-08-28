defmodule Verdict.ThresholdTest do
  use ExUnit.Case, async: true

  # scikit-learn's precision_recall_curve on this data returns
  #   precision [0.5, 0.667, 0.5, 1.0, 1.0]
  #   recall    [1.0, 1.0,   0.5, 0.5, 0.0]
  #   thresholds[0.1, 0.35,  0.4, 0.8]
  # The last precision/recall pair has no threshold and is dropped here.
  defp y_true, do: Nx.tensor([0, 0, 1, 1])
  defp scores, do: Nx.tensor([0.1, 0.4, 0.35, 0.8])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp lines(plot), do: spec(plot)["layer"] |> List.last()

  defp series(plot, metric) do
    plot
    |> values()
    |> Enum.filter(&(&1["metric"] == metric))
    |> Enum.sort_by(& &1["threshold"])
  end

  describe "threshold_curve/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.threshold_curve(y_true(), scores())
    end

    test "plots one point per threshold, per metric" do
      plot = Verdict.threshold_curve(y_true(), scores())

      assert length(values(plot)) == 4 * 3

      for metric <- ["Precision", "Recall", "F1"] do
        assert plot |> series(metric) |> Enum.map(& &1["threshold"]) ==
                 [0.10000000149011612, 0.3499999940395355, 0.4000000059604645, 0.800000011920929]
      end
    end

    test "carries the precision and recall scikit-learn computes" do
      plot = Verdict.threshold_curve(y_true(), scores())

      plot
      |> series("Precision")
      |> Enum.map(& &1["value"])
      |> Enum.zip([0.5, 0.6666667, 0.5, 1.0])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)

      plot
      |> series("Recall")
      |> Enum.map(& &1["value"])
      |> Enum.zip([1.0, 1.0, 0.5, 0.5])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "derives F1 from the precision and recall at each threshold" do
      plot = Verdict.threshold_curve(y_true(), scores())

      expected =
        Enum.zip([0.5, 0.6666667, 0.5, 1.0], [1.0, 1.0, 0.5, 0.5])
        |> Enum.map(fn {p, r} -> 2 * p * r / (p + r) end)

      plot
      |> series("F1")
      |> Enum.map(& &1["value"])
      |> Enum.zip(expected)
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "drops the trailing point that has no threshold" do
      # precision_recall_curve returns 5 pairs but only 4 thresholds; plotting
      # the extra pair would put a point at a threshold that does not exist
      plot = Verdict.threshold_curve(y_true(), scores(), metrics: [:precision])
      assert length(values(plot)) == 4
    end

    test "marks the best F1 and reports it in the subtitle" do
      plot = Verdict.threshold_curve(y_true(), scores())
      [rule, _lines] = spec(plot)["layer"]

      best = plot |> series("F1") |> Enum.max_by(& &1["value"])

      assert rule["mark"]["type"] == "rule"
      assert [%{"threshold" => marked}] = rule["data"]["values"]
      assert marked == best["threshold"]

      assert spec(plot)["title"]["subtitle"] =~ "best F1"
    end

    test "mark_best_f1: false drops the rule" do
      plot = Verdict.threshold_curve(y_true(), scores(), mark_best_f1: false)
      assert length(spec(plot)["layer"]) == 1
      refute spec(plot)["title"]
    end

    test "there is nothing to mark when F1 is not plotted" do
      plot = Verdict.threshold_curve(y_true(), scores(), metrics: [:precision, :recall])

      assert length(spec(plot)["layer"]) == 1

      assert plot |> values() |> Enum.map(& &1["metric"]) |> Enum.uniq() |> Enum.sort() ==
               ["Precision", "Recall"]
    end

    test "keeps the requested metrics in the given order" do
      plot = Verdict.threshold_curve(y_true(), scores(), metrics: [:recall, :precision])
      assert lines(plot)["encoding"]["color"]["scale"]["domain"] == ["Recall", "Precision"]
    end

    test "pins the value axis to 0..1" do
      layer = lines(Verdict.threshold_curve(y_true(), scores()))
      assert layer["encoding"]["y"]["scale"]["domain"] == [0, 1]
    end

    test "rejects a y_true that is not binary" do
      assert_raise ArgumentError, ~r/only 0 and 1/, fn ->
        Verdict.threshold_curve(Nx.tensor([0, 1, 2, 1]), scores())
      end
    end

    test "rejects inputs of different lengths" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        Verdict.threshold_curve(y_true(), Nx.tensor([0.1, 0.2]))
      end
    end
  end
end
