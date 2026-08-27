defmodule EvalViz.GridSearchTest do
  use ExUnit.Case, async: true

  # The shape Scholar.ModelSelection.grid_search/5 returns.
  defp results do
    [
      %{hyperparameters: [alpha: 0.0, iterations: 10], score: Nx.tensor([0.7, 2.0])},
      %{hyperparameters: [alpha: 0.0, iterations: 50], score: Nx.tensor([0.8, 1.0])},
      %{hyperparameters: [alpha: 1.0, iterations: 10], score: Nx.tensor([0.6, 3.0])},
      %{hyperparameters: [alpha: 1.0, iterations: 50], score: Nx.tensor([0.9, 0.5])}
    ]
  end

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp heatmap(plot), do: spec(plot)["layer"] |> hd()

  defp cell(plot, x, y) do
    plot |> values() |> Enum.find(&(&1["x"] == x and &1["y"] == y))
  end

  describe "grid_search/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.grid_search(results())
    end

    test "draws a cell per combination" do
      assert length(values(EvalViz.grid_search(results()))) == 4
    end

    test "infers the axes from the hyperparameters that vary" do
      plot = EvalViz.grid_search(results())

      assert heatmap(plot)["encoding"]["x"]["title"] == "alpha"
      assert heatmap(plot)["encoding"]["y"]["title"] == "iterations"
    end

    test "takes the axes when given them" do
      plot = EvalViz.grid_search(results(), x: :iterations, y: :alpha)

      assert heatmap(plot)["encoding"]["x"]["title"] == "iterations"
      assert_in_delta cell(plot, "50", "1.0")["score"], 0.9, 1.0e-6
    end

    # Sorting the labels instead of the values would put 20.0 before 5.0.
    test "orders the axes by value, not by label" do
      wide =
        for alpha <- [0.0, 5.0, 20.0], iterations <- [10, 100] do
          %{hyperparameters: [alpha: alpha, iterations: iterations], score: Nx.tensor([alpha])}
        end

      plot = EvalViz.grid_search(wide)

      assert heatmap(plot)["encoding"]["x"]["sort"] == ["0.0", "5.0", "20.0"]
      assert heatmap(plot)["encoding"]["y"]["sort"] == ["10", "100"]
    end

    test "ignores a hyperparameter that never changes" do
      fixed = Enum.map(results(), &%{&1 | hyperparameters: [{:classes, 3} | &1.hyperparameters]})

      assert %VegaLite{} = EvalViz.grid_search(fixed)
    end

    test "picks the metric out of each score" do
      plot = EvalViz.grid_search(results(), metric: 1, best: :min)
      assert cell(plot, "1.0", "50")["score"] == 0.5
    end

    test "rejects a metric the scores do not have" do
      assert_raise ArgumentError, ~r/:metric to index the 2 scores/, fn ->
        EvalViz.grid_search(results(), metric: 5)
      end
    end

    test "marks the best cell and names it in the subtitle" do
      plot = EvalViz.grid_search(results())

      assert spec(plot)["title"]["subtitle"] == "best 0.9 at alpha 1.0, iterations 50"

      assert spec(plot)["layer"] |> List.last() |> get_in(["data", "values"]) == [
               %{"x" => "1.0", "y" => "50"}
             ]
    end

    test "best: :min marks the smallest instead" do
      plot = EvalViz.grid_search(results(), metric: 1, best: :min)
      assert spec(plot)["title"]["subtitle"] == "best 0.5 at alpha 1.0, iterations 50"
    end

    # Darker should always read as better, whichever direction the score runs.
    test "reverses the ramp when smaller is better" do
      refute heatmap(EvalViz.grid_search(results()))["encoding"]["color"]["scale"]["reverse"]

      assert heatmap(EvalViz.grid_search(results(), best: :min))["encoding"]["color"]["scale"][
               "reverse"
             ]
    end

    # The text has to flip on the dark end, and which end that is depends on
    # the ramp, so the row carries the shade rather than the raw score.
    test "carries a shade that runs from worst to best either way" do
      up = EvalViz.grid_search(results())
      down = EvalViz.grid_search(results(), best: :min)

      assert cell(up, "1.0", "50")["shade"] == 1.0
      assert cell(down, "1.0", "50")["shade"] == 0.0
    end

    test "best: :none marks nothing" do
      plot = EvalViz.grid_search(results(), best: :none)

      refute spec(plot)["title"]
      assert length(spec(plot)["layer"]) == 2
    end

    test "values: false leaves the numbers out" do
      assert length(spec(EvalViz.grid_search(results(), values: false))["layer"]) == 2
    end

    # A real search varies more than two parameters, and a heatmap has room for
    # two. Dropping the rest silently would read as full coverage.
    test "collapses extra parameters and says so" do
      wide =
        for alpha <- [0.0, 1.0], iterations <- [10, 50], depth <- [1, 2] do
          %{
            hyperparameters: [alpha: alpha, iterations: iterations, depth: depth],
            score: Nx.tensor([alpha + iterations / 100 + depth / 10])
          }
        end

      plot = EvalViz.grid_search(wide, x: :alpha, y: :iterations)

      assert length(values(plot)) == 4
      assert spec(plot)["title"]["subtitle"] =~ "highest over depth"
      # depth 2 wins every cell, so alpha 1.0 with 50 iterations tops out
      assert_in_delta cell(plot, "1.0", "50")["score"], 1.7, 1.0e-6
    end

    test "asks which two to plot when more than two vary" do
      wide = [
        %{hyperparameters: [a: 1, b: 1, c: 1], score: Nx.tensor([0.1])},
        %{hyperparameters: [a: 2, b: 2, c: 2], score: Nx.tensor([0.2])}
      ]

      assert_raise ArgumentError, ~r/exactly two hyperparameters to vary/, fn ->
        EvalViz.grid_search(wide)
      end
    end

    test "rejects an empty search" do
      assert_raise ArgumentError, ~r/at least one grid search result/, fn ->
        EvalViz.grid_search([])
      end
    end
  end
end
