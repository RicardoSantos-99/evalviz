defmodule EvalViz.LearningCurveTest do
  use ExUnit.Case, async: true

  defp sizes, do: [10, 20, 40]
  defp train, do: [[1.0, 0.9], [0.9, 0.8], [0.8, 0.7]]
  defp validation, do: [[0.5, 0.3], [0.7, 0.5], [0.8, 0.6]]

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp line(plot), do: spec(plot)["layer"] |> List.last()

  defp series(plot, name) do
    plot |> values() |> Enum.filter(&(&1["series"] == name)) |> Enum.sort_by(& &1["size"])
  end

  describe "learning_curve/4" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.learning_curve(sizes(), train(), validation())
    end

    test "averages the folds at each training size" do
      plot = EvalViz.learning_curve(sizes(), train(), validation())

      plot
      |> series("Training")
      |> Enum.map(& &1["score"])
      |> Enum.zip([0.95, 0.85, 0.75])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-9 end)

      plot
      |> series("Validation")
      |> Enum.map(& &1["score"])
      |> Enum.zip([0.4, 0.6, 0.7])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-9 end)
    end

    test "spreads the band one standard deviation either side of the mean" do
      plot = EvalViz.learning_curve(sizes(), train(), validation())
      [first | _] = series(plot, "Validation")

      # scores 0.5 and 0.3: mean 0.4, population deviation 0.1
      assert_in_delta first["lower"], 0.3, 1.0e-9
      assert_in_delta first["upper"], 0.5, 1.0e-9
    end

    test "draws the band under the lines, so the means stay legible" do
      [band, line] = spec(EvalViz.learning_curve(sizes(), train(), validation()))["layer"]

      assert band["mark"]["type"] == "area"
      assert line["mark"]["type"] == "line"
    end

    # The band and the line share the colour field, so hiding the legend on the
    # band hides the merged one and leaves two curves with nothing naming them.
    test "keeps the legend the band could suppress" do
      [band, _line] = spec(EvalViz.learning_curve(sizes(), train(), validation()))["layer"]

      refute Map.has_key?(band["encoding"]["color"], "legend")
    end

    test "accepts one score per size, and then has no band to draw" do
      plot = EvalViz.learning_curve(sizes(), [1.0, 0.9, 0.8], [0.5, 0.7, 0.8])

      assert length(spec(plot)["layer"]) == 1

      plot
      |> series("Training")
      |> Enum.map(& &1["score"])
      |> Enum.zip([1.0, 0.9, 0.8])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-9 end)
    end

    test "spread: false drops the band even with folds" do
      plot = EvalViz.learning_curve(sizes(), train(), validation(), spread: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "accepts tensors as well as lists" do
      from_lists = values(EvalViz.learning_curve(sizes(), train(), validation()))

      from_tensors =
        values(
          EvalViz.learning_curve(
            Nx.tensor(sizes()),
            Nx.tensor(train(), type: :f64),
            Nx.tensor(validation(), type: :f64)
          )
        )

      assert Enum.map(from_lists, & &1["series"]) == Enum.map(from_tensors, & &1["series"])

      Enum.zip(from_lists, from_tensors)
      |> Enum.each(fn {a, b} -> assert_in_delta a["score"], b["score"], 1.0e-9 end)
    end

    test "renames the curves when asked" do
      plot =
        EvalViz.learning_curve(sizes(), train(), validation(), labels: ["Fit", "Held out"])

      assert plot |> values() |> Enum.map(& &1["series"]) |> Enum.uniq() == ["Fit", "Held out"]
      assert line(plot)["encoding"]["color"]["scale"]["domain"] == ["Fit", "Held out"]
    end

    test "lets the score axis start where the data does" do
      # scores usually live in a narrow band near the top, so forcing zero
      # would flatten the very gap the plot exists to show
      layer = line(EvalViz.learning_curve(sizes(), train(), validation()))
      assert layer["encoding"]["y"]["scale"]["zero"] == false
    end

    test "rejects scores that do not line up with the sizes" do
      assert_raise ArgumentError, ~r/one entry per training size/, fn ->
        EvalViz.learning_curve(sizes(), [[1.0]], validation())
      end
    end

    test "rejects a scores tensor of the wrong rank" do
      assert_raise ArgumentError, ~r/rank 1 or 2/, fn ->
        EvalViz.learning_curve(sizes(), Nx.broadcast(0.5, {3, 2, 2}), validation())
      end
    end
  end
end
