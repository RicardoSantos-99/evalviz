defmodule EvalViz.ValidationCurveTest do
  use ExUnit.Case, async: true

  defp alphas, do: [0.01, 0.1, 1.0, 10.0]
  defp train, do: [[0.99, 0.98], [0.97, 0.96], [0.90, 0.88], [0.72, 0.70]]
  defp validation, do: [[0.80, 0.78], [0.88, 0.86], [0.86, 0.84], [0.70, 0.68]]

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp line(plot), do: spec(plot)["layer"] |> List.last()

  defp series(plot, name) do
    plot |> values() |> Enum.filter(&(&1["series"] == name)) |> Enum.map(& &1["score"])
  end

  describe "validation_curve/4" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.validation_curve(alphas(), train(), validation())
    end

    test "averages the folds at each parameter value" do
      plot = EvalViz.validation_curve(alphas(), train(), validation())

      plot
      |> series("Validation")
      |> Enum.zip([0.79, 0.87, 0.85, 0.69])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-9 end)
    end

    test "spreads the band one standard deviation either side" do
      [first | _] = EvalViz.validation_curve(alphas(), train(), validation()) |> values()

      assert_in_delta first["lower"], 0.98, 1.0e-9
      assert_in_delta first["upper"], 0.99, 1.0e-9
    end

    # The band and the line share the colour field, so hiding the legend on the
    # band hides the merged one and leaves two curves with nothing naming them.
    test "keeps the legend the band could suppress" do
      [band | _] = spec(EvalViz.validation_curve(alphas(), train(), validation()))["layer"]

      assert band["mark"]["type"] == "area"
      refute Map.has_key?(band["encoding"]["color"], "legend")
    end

    test "accepts one score per value, and then has no band" do
      plot = EvalViz.validation_curve(alphas(), [0.9, 0.9, 0.9, 0.9], [0.8, 0.85, 0.8, 0.7])
      assert length(spec(plot)["layer"]) == 2
    end

    test "marks the best validation score" do
      plot = EvalViz.validation_curve(alphas(), train(), validation(), param_name: "alpha")

      assert spec(plot)["title"]["subtitle"] == "best 0.87 at alpha = 0.1"
      assert spec(plot)["layer"] |> Enum.any?(&(&1["mark"]["type"] == "rule"))
    end

    # An error is best at its smallest, and taking the maximum would point at
    # the worst setting on the plot.
    test "best: :min marks the smallest score instead" do
      errors = [[0.5, 0.5], [0.2, 0.2], [0.3, 0.3], [0.9, 0.9]]
      plot = EvalViz.validation_curve(alphas(), train(), errors, best: :min)

      assert spec(plot)["title"]["subtitle"] =~ "best 0.2 at Parameter value = 0.1"
    end

    test "best: :none marks nothing" do
      plot = EvalViz.validation_curve(alphas(), train(), validation(), best: :none)

      refute spec(plot)["title"]
      refute spec(plot)["layer"] |> Enum.any?(&(&1["mark"]["type"] == "rule"))
    end

    test "puts the parameter on a quantitative axis when it is numeric" do
      assert line(EvalViz.validation_curve(alphas(), train(), validation()))["encoding"]["x"][
               "type"
             ] == "quantitative"
    end

    # Regularisation strengths are swept over orders of magnitude, and a linear
    # axis would crush every small value against the origin.
    test "scale: :log puts the axis on a log scale" do
      plot = EvalViz.validation_curve(alphas(), train(), validation(), scale: :log)
      assert line(plot)["encoding"]["x"]["scale"]["type"] == "log"
    end

    test "refuses a log scale that would take the log of zero" do
      assert_raise ArgumentError, ~r/positive for a log scale/, fn ->
        EvalViz.validation_curve([0.0, 0.1, 1.0, 10.0], train(), validation(), scale: :log)
      end
    end

    test "refuses a log scale on values that are not numbers" do
      assert_raise ArgumentError, ~r/numeric parameter values/, fn ->
        EvalViz.validation_curve([:a, :b, :c, :d], train(), validation(), scale: :log)
      end
    end

    # A parameter can be a boolean or an atom, and those have no position on a
    # number line.
    test "puts a non-numeric parameter on a nominal axis, in the order given" do
      plot = EvalViz.validation_curve([true, false], [[0.9], [0.8]], [[0.7], [0.75]])

      assert line(plot)["encoding"]["x"]["type"] == "nominal"
      assert line(plot)["encoding"]["x"]["sort"] == ["true", "false"]
      assert plot |> values() |> Enum.map(& &1["param"]) |> Enum.uniq() == ["true", "false"]
    end

    test "renames the curves and the axes" do
      plot =
        EvalViz.validation_curve(alphas(), train(), validation(),
          labels: ["Fit", "Held out"],
          score_title: "R²"
        )

      assert plot |> values() |> Enum.map(& &1["series"]) |> Enum.uniq() == ["Fit", "Held out"]
      assert line(plot)["encoding"]["y"]["title"] == "R²"
    end

    test "lets the score axis start where the data does" do
      assert line(EvalViz.validation_curve(alphas(), train(), validation()))["encoding"]["y"][
               "scale"
             ]["zero"] == false
    end

    test "rejects scores that do not line up with the parameter values" do
      assert_raise ArgumentError, ~r/one entry per parameter value/, fn ->
        EvalViz.validation_curve(alphas(), [[0.9]], validation())
      end
    end

    test "rejects a scores tensor of the wrong rank" do
      assert_raise ArgumentError, ~r/rank 1 or 2/, fn ->
        EvalViz.validation_curve(alphas(), Nx.broadcast(0.5, {4, 2, 2}), validation())
      end
    end
  end
end
