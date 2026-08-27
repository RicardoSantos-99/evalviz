defmodule EvalViz.ReportTest do
  use ExUnit.Case, async: true

  defp y_true, do: Nx.tensor([0, 0, 1, 1, 0, 1, 1, 0])
  defp scores, do: Nx.tensor([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.3], type: :f64)

  defp target, do: Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  defp predicted, do: Nx.tensor([1.2, 1.9, 3.3, 3.8, 5.4, 5.7])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp panels(plot), do: spec(plot)["concat"]
  # Vega-Lite takes a title as a plain string, or as a map once there is a
  # subtitle to go with it, and the panels here come in both shapes.
  defp titles(plot), do: panels(plot) |> Enum.map(&title_of(&1["title"]))

  defp title_of(%{"text" => text}), do: text
  defp title_of(text) when is_binary(text), do: text

  describe "report/3 for a classifier" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.report(y_true(), scores())
    end

    test "draws the four plots that go together" do
      assert titles(EvalViz.report(y_true(), scores())) == [
               "Confusion matrix at 0.5",
               "ROC",
               "Precision-recall",
               "Scores by true class"
             ]
    end

    test "takes the plots it is asked for, in order" do
      plot = EvalViz.report(y_true(), scores(), plots: [:calibration, :det])
      assert titles(plot) == ["Calibration", "DET"]
    end

    # The report is worth more than the plots separately because one cut-off
    # runs through it: the matrix counts what the rule on the histogram splits.
    test "reads one threshold in the matrix and the score distribution" do
      [matrix, _roc, _pr, distribution] =
        panels(EvalViz.report(y_true(), scores(), threshold: 0.7))

      assert title_of(matrix["title"]) == "Confusion matrix at 0.7"

      rule = distribution["layer"] |> List.last()
      assert rule["data"]["values"] == [%{"threshold" => 0.7}]
    end

    test "counts the matrix at that threshold, not at a fixed one" do
      [matrix, _, _, _] = panels(EvalViz.report(y_true(), scores(), threshold: 0.85))

      # only the 0.9 score clears 0.85, and it is a true positive
      predicted_positive =
        matrix["data"]["values"]
        |> Enum.filter(&(&1["predicted"] == "1"))
        |> Enum.map(& &1["value"])
        |> Enum.sum()

      assert predicted_positive == 1
    end

    test "names the classes throughout" do
      plot = EvalViz.report(y_true(), scores(), class_names: ["no", "yes"])
      [matrix, _roc, _pr, distribution] = panels(plot)

      assert matrix["data"]["values"] |> Enum.map(& &1["actual"]) |> Enum.uniq() == ["no", "yes"]

      assert distribution["data"]["values"] |> Enum.map(& &1["class"]) |> Enum.uniq() ==
               ["no", "yes"]
    end

    test "titles and lays out the whole thing" do
      plot = EvalViz.report(y_true(), scores(), title: "Held out", columns: 4)

      assert spec(plot)["title"] == "Held out"
      assert spec(plot)["columns"] == 4
    end

    # Every panel reads one column of scores, and the one-vs-rest plots take a
    # matrix on their own, so the error points at them rather than guessing.
    test "sends a multiclass model to the plots that handle it" do
      assert_raise ArgumentError, ~r/call roc_curve\/3 and the others directly/, fn ->
        EvalViz.report(
          Nx.tensor([0, 1, 2]),
          Nx.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        )
      end
    end

    test "rejects a plot that is not in the set" do
      assert_raise ArgumentError, ~r/:dendrogram is not one of the classification plots/, fn ->
        EvalViz.report(y_true(), scores(), plots: [:dendrogram])
      end
    end
  end

  describe "report/3 for a regression" do
    test "draws the four regression plots" do
      plot = EvalViz.report(target(), predicted(), kind: :regression)

      assert titles(plot) == [
               "Predicted vs actual",
               "Residuals",
               "Residual distribution",
               "Normal Q-Q"
             ]
    end

    test "reads the second argument as predictions" do
      [scatter | _] = panels(EvalViz.report(target(), predicted(), kind: :regression))

      assert scatter["data"]["values"] |> Enum.map(& &1["actual"]) ==
               Nx.to_flat_list(target())
    end

    test "rejects a classification plot in a regression report" do
      assert_raise ArgumentError, ~r/:roc is not one of the regression plots/, fn ->
        EvalViz.report(target(), predicted(), kind: :regression, plots: [:roc])
      end
    end
  end
end
