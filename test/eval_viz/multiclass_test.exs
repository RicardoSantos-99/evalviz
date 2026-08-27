defmodule EvalViz.MulticlassTest do
  use ExUnit.Case, async: true

  # Every expected number below comes from scikit-learn 1.6.1 on this data,
  # using label_binarize for the one-vs-rest split and np.interp to average.
  defp y_true, do: Nx.tensor([0, 1, 2, 2, 0, 1, 2, 0, 1, 1])

  defp scores do
    Nx.tensor(
      [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.3, 0.3, 0.4],
        [0.6, 0.2, 0.2],
        [0.1, 0.5, 0.4],
        [0.2, 0.2, 0.6],
        [0.4, 0.4, 0.2],
        [0.3, 0.6, 0.1],
        [0.2, 0.3, 0.5]
      ],
      type: :f64
    )
  end

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]

  defp series_names(plot), do: plot |> values() |> Enum.map(& &1["series"]) |> Enum.uniq()

  defp series(plot, name, field) do
    plot
    |> values()
    |> Enum.filter(&(&1["series"] == name))
    |> Enum.map(& &1[field])
  end

  defp assert_close(got, want) do
    assert length(got) == length(want)

    Enum.zip(got, want)
    |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-6 end)
  end

  describe "one-vs-rest expansion" do
    test "draws one curve per score column, named for the class" do
      assert EvalViz.roc_curve(y_true(), scores())
             |> series_names()
             |> Enum.map(&(String.split(&1, " (") |> hd())) == ["0", "1", "2"]
    end

    test "carries the per-class AUC scikit-learn computes" do
      # 1.0, 0.9375 and 0.928571 before the legend rounds them
      assert series_names(EvalViz.roc_curve(y_true(), scores())) ==
               ["0 (AUC = 1.0)", "1 (AUC = 0.938)", "2 (AUC = 0.929)"]
    end

    test "names the classes when asked" do
      plot = EvalViz.roc_curve(y_true(), scores(), class_names: ["cat", "dog", "bird"])

      assert series_names(plot) ==
               ["cat (AUC = 1.0)", "dog (AUC = 0.938)", "bird (AUC = 0.929)"]
    end

    test "keeps the model label alongside the class when comparing models" do
      plot = EvalViz.roc_curve([{"A", y_true(), scores()}, {"B", y_true(), scores()}])

      assert plot |> series_names() |> Enum.map(&(String.split(&1, " (") |> hd())) ==
               ["A: 0", "A: 1", "A: 2", "B: 0", "B: 1", "B: 2"]
    end

    test "rejects labels that are not indices into the score columns" do
      assert_raise ArgumentError, ~r/class indices in 0\.\.2/, fn ->
        EvalViz.roc_curve(Nx.tensor([0, 1, 3, 2, 0, 1, 2, 0, 1, 1]), scores())
      end
    end

    test "rejects a y_true that does not line up with the score rows" do
      assert_raise ArgumentError, ~r/one entry per row/, fn ->
        EvalViz.roc_curve(Nx.tensor([0, 1, 2]), scores())
      end
    end

    test "leaves binary input alone" do
      plot = EvalViz.roc_curve(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0.1, 0.4, 0.35, 0.8]))

      assert spec(plot)["title"]["subtitle"] == "AUC = 0.75"
      refute spec(plot)["data"]["values"] |> hd() |> Map.has_key?("class")
    end
  end

  describe "micro average" do
    test "pools every (sample, class) pair into one curve" do
      plot = EvalViz.roc_curve(y_true(), scores(), average: :micro)

      assert_close(
        series(plot, "micro-average (AUC = 0.955)", "fpr"),
        [0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.3, 0.7, 1.0]
      )

      assert_close(
        series(plot, "micro-average (AUC = 0.955)", "tpr"),
        [0.0, 0.1, 0.3, 0.6, 0.7, 0.9, 1.0, 1.0, 1.0]
      )
    end

    test "comes after the per-class curves" do
      assert EvalViz.roc_curve(y_true(), scores(), average: :micro)
             |> series_names()
             |> List.last() ==
               "micro-average (AUC = 0.955)"
    end
  end

  describe "macro average" do
    test "resamples onto the union of the per-class grids" do
      plot = EvalViz.roc_curve(y_true(), scores(), average: :macro)

      assert_close(
        series(plot, "macro-average (AUC = 0.956)", "fpr"),
        [0.0, 0.142857, 0.166667, 0.285714, 0.333333, 0.571429, 0.714286, 0.833333, 1.0]
      )

      assert_close(
        series(plot, "macro-average (AUC = 0.956)", "tpr"),
        [0.805556, 0.805556, 0.824074, 0.97619, 1.0, 1.0, 1.0, 1.0, 1.0]
      )
    end

    # These curves repeat an x, and taking the wrong tied y silently shifts the
    # average. Precision-recall is where the two rules disagree.
    test "interpolates precision-recall the way numpy does" do
      plot = EvalViz.precision_recall_curve(y_true(), scores(), average: :macro)

      assert_close(
        series(plot, "macro-average (AP = 0.928)", "recall"),
        [0.0, 0.25, 0.333333, 0.5, 0.666667, 0.75, 1.0]
      )

      assert_close(
        series(plot, "macro-average (AP = 0.928)", "precision"),
        [1.0, 1.0, 1.0, 0.944444, 0.944444, 0.883333, 0.755556]
      )
    end

    test "reports the mean of the per-class summaries" do
      # (1.0 + 0.9375 + 0.928571) / 3
      assert EvalViz.roc_curve(y_true(), scores(), average: :macro)
             |> series_names()
             |> List.last() == "macro-average (AUC = 0.956)"
    end

    test "has no summary to report when the curve has none" do
      assert EvalViz.det_curve(y_true(), scores(), average: :macro)
             |> series_names()
             |> List.last() == "macro-average"
    end

    test "draws both averages when both are asked for" do
      names = EvalViz.roc_curve(y_true(), scores(), average: [:micro, :macro]) |> series_names()

      assert length(names) == 5
      assert Enum.at(names, 3) =~ "micro-average"
      assert Enum.at(names, 4) =~ "macro-average"
    end
  end

  describe "precision-recall baseline" do
    test "is left out when the classes have different positive rates" do
      plot = EvalViz.precision_recall_curve(y_true(), scores())
      assert length(spec(plot)["layer"]) == 1
    end

    test "is drawn when every curve shares it" do
      balanced = Nx.tensor([0, 1, 2, 0, 1, 2])

      probabilities =
        Nx.tensor(
          [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.2, 0.2, 0.6]
          ],
          type: :f64
        )

      [baseline, _curves] = spec(EvalViz.precision_recall_curve(balanced, probabilities))["layer"]

      assert_close(Enum.map(baseline["data"]["values"], & &1["y"]), [1 / 3, 1 / 3])
    end
  end

  describe "calibration" do
    test "draws a curve per class and pools them for the micro average" do
      plot =
        EvalViz.calibration_curve(y_true(), scores(),
          bins: 2,
          class_names: [:a, :b, :c],
          average: :micro
        )

      assert plot |> series_names() |> Enum.map(&(String.split(&1, " (") |> hd())) ==
               ["a", "b", "c", "micro-average"]
    end

    test "reports each class's Brier score" do
      # class a one-vs-rest: mean((p - y)^2) = 0.88 / 10
      plot = EvalViz.calibration_curve(y_true(), scores(), bins: 2, class_names: [:a, :b, :c])
      assert plot |> series_names() |> hd() == "a (Brier = 0.088)"
    end

    test "has no macro option, since the per-class bins do not line up" do
      assert_raise NimbleOptions.ValidationError, ~r/:average/, fn ->
        EvalViz.calibration_curve(y_true(), scores(), average: :macro)
      end
    end
  end

  describe "threshold curve" do
    test "puts the class on colour and the metric on the dash pattern" do
      encoding = EvalViz.threshold_curve(y_true(), scores()) |> spec() |> get_in(["layer"])

      assert [%{"encoding" => encoding}] = encoding
      assert encoding["color"]["field"] == "class"
      assert encoding["strokeDash"]["field"] == "metric"
      assert encoding["color"]["scale"]["domain"] == ["0", "1", "2"]
    end

    test "gives the dash legend a swatch that shows the pattern" do
      legend =
        EvalViz.threshold_curve(y_true(), scores())
        |> spec()
        |> get_in(["layer"])
        |> List.last()
        |> get_in(["encoding", "strokeDash", "legend"])

      assert legend["symbolFillColor"] == "transparent"
      assert legend["symbolStrokeColor"]
    end

    test "leaves the best-F1 rule out, since each class peaks somewhere else" do
      plot = EvalViz.threshold_curve(y_true(), scores())

      assert length(spec(plot)["layer"]) == 1
      refute spec(plot)["title"]
    end

    test "still marks the best F1 on binary input" do
      plot = EvalViz.threshold_curve(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0.1, 0.4, 0.35, 0.8]))

      assert length(spec(plot)["layer"]) == 2
      assert spec(plot)["title"]["subtitle"] =~ "best F1"
    end

    test "keeps colour on the metric for binary input" do
      encoding =
        EvalViz.threshold_curve(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0.1, 0.4, 0.35, 0.8]))
        |> spec()
        |> get_in(["layer"])
        |> List.last()
        |> Map.fetch!("encoding")

      assert encoding["color"]["field"] == "metric"
      refute encoding["strokeDash"]
    end

    test "adds a pooled set of curves for the micro average" do
      plot = EvalViz.threshold_curve(y_true(), scores(), average: :micro)

      assert plot |> values() |> Enum.map(& &1["class"]) |> Enum.uniq() ==
               ["0", "1", "2", "micro-average"]
    end
  end
end
