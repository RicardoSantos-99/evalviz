defmodule EvalViz.FacetTest do
  use ExUnit.Case, async: true

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
  defp child(plot), do: spec(plot)["spec"]
  defp values(plot), do: spec(plot)["data"]["values"]

  describe "faceting the one-vs-rest curves" do
    test "splits into a facet spec rather than one set of axes" do
      plot = EvalViz.roc_curve(y_true(), scores(), facet: true)

      assert spec(plot)["facet"]["field"] == "series"
      assert spec(plot)["spec"]["layer"]
      refute spec(plot)["layer"]
    end

    # The panel already names its curve, AUC included, so a legend repeating it
    # would just take up room.
    test "drops the colour legend, which the panel title replaces" do
      plot = EvalViz.roc_curve(y_true(), scores(), facet: true)
      curve = child(plot)["layer"] |> List.last()

      refute curve["encoding"]["color"]
    end

    # `columns` is a property of the outer spec; inside the facet definition it
    # is ignored and every panel lands on one row.
    test "wraps the panels at the column count" do
      assert spec(EvalViz.roc_curve(y_true(), scores(), facet: true))["columns"] == 3
      assert spec(EvalViz.roc_curve(y_true(), scores(), facet: true, columns: 2))["columns"] == 2
    end

    # Otherwise the header prints the field's own name above the panels.
    test "leaves the facet header unlabelled" do
      assert Map.fetch!(
               spec(EvalViz.roc_curve(y_true(), scores(), facet: true))["facet"],
               "title"
             ) ==
               nil
    end

    test "carries the title over the whole thing" do
      plot = EvalViz.roc_curve(y_true(), scores(), facet: true, title: "Six ways")
      assert spec(plot)["title"] == "Six ways"
    end

    test "has nothing to split with a single curve" do
      plot =
        EvalViz.roc_curve(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0.1, 0.4, 0.35, 0.8]), facet: true)

      refute spec(plot)["facet"]
      assert spec(plot)["layer"]
    end

    test "still keeps the diagonal in every panel" do
      plot = EvalViz.roc_curve(y_true(), scores(), facet: true)
      [reference, _curve] = child(plot)["layer"]

      assert reference["data"]["values"] == [%{"x" => 0, "y" => 0}, %{"x" => 1, "y" => 1}]
    end
  end

  describe "the precision-recall baseline under faceting" do
    # Overlaid, the classes disagree on their share of positives and no single
    # baseline is honest. A panel holds one class, so it can draw its own.
    test "is dropped when overlaid but drawn per panel when faceted" do
      overlaid = EvalViz.precision_recall_curve(y_true(), scores())
      faceted = EvalViz.precision_recall_curve(y_true(), scores(), facet: true)

      assert length(spec(overlaid)["layer"]) == 1

      [baseline, _curve] = child(faceted)["layer"]
      assert baseline["mark"]["type"] == "rule"
      assert baseline["encoding"]["y"]["field"] == "baseline"
    end

    test "gives each class its own share of positives" do
      # y_true holds class 0 three times, class 1 four, class 2 three, of ten
      baselines =
        EvalViz.precision_recall_curve(y_true(), scores(), facet: true)
        |> values()
        |> Enum.group_by(&String.first(&1["series"]), & &1["baseline"])
        |> Enum.map(fn {class, rates} -> {class, rates |> Enum.uniq() |> hd()} end)
        |> Enum.sort()

      assert [{"0", zero}, {"1", one}, {"2", two}] = baselines

      assert_in_delta zero, 0.3, 1.0e-6
      assert_in_delta one, 0.4, 1.0e-6
      assert_in_delta two, 0.3, 1.0e-6
    end

    # data_from_values rejects rows that disagree on their columns, so the
    # macro curve needs a baseline too, and averaging the per-class ones is
    # what its own curve does.
    test "gives the macro curve the average of the per-class ones" do
      rows = values(EvalViz.precision_recall_curve(y_true(), scores(), average: :macro))

      macro =
        rows
        |> Enum.filter(&(&1["series"] =~ "macro"))
        |> Enum.map(& &1["baseline"])
        |> Enum.uniq()

      # (0.3 + 0.4 + 0.3) / 3
      assert [rate] = macro
      assert_in_delta rate, 1 / 3, 1.0e-6
    end

    test "keeps every row agreeing on its columns" do
      rows = values(EvalViz.precision_recall_curve(y_true(), scores(), average: [:micro, :macro]))
      assert rows |> Enum.map(&(Map.keys(&1) |> Enum.sort())) |> Enum.uniq() |> length() == 1
    end
  end

  describe "faceting calibration" do
    test "splits into a panel per class" do
      plot = EvalViz.calibration_curve(y_true(), scores(), bins: 2, facet: true)

      assert spec(plot)["facet"]["field"] == "series"
      refute child(plot)["layer"] |> List.last() |> get_in(["encoding", "color"])
    end
  end

  describe "faceting the threshold curve" do
    test "splits on the class rather than the curve" do
      plot = EvalViz.threshold_curve(y_true(), scores(), facet: true)
      assert spec(plot)["facet"]["field"] == "class"
    end

    # A panel holds one class, so colour is free again and the dash pattern it
    # was pushed onto is no longer needed.
    test "gives colour back to the metric and drops the dash" do
      plot = EvalViz.threshold_curve(y_true(), scores(), facet: true)
      encoding = child(plot)["layer"] |> List.last() |> Map.fetch!("encoding")

      assert encoding["color"]["field"] == "metric"
      assert encoding["color"]["scale"]["domain"] == ["Precision", "Recall", "F1"]
      refute encoding["strokeDash"]
    end

    test "leaves binary input alone" do
      plot =
        EvalViz.threshold_curve(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0.1, 0.4, 0.35, 0.8]),
          facet: true
        )

      refute spec(plot)["facet"]
      assert spec(plot)["layer"]
    end
  end
end
