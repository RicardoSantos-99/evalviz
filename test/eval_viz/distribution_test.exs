defmodule EvalViz.DistributionTest do
  use ExUnit.Case, async: true

  alias EvalViz.Internal

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]

  describe "normal_quantile/1" do
    # scipy.stats.norm.ppf on the same probabilities. Acklam's approximation
    # claims a relative error under 1.15e-9 and lands right at it.
    test "matches scipy" do
      [
        {0.001, -3.0902323062},
        {0.01, -2.3263478740},
        {0.025, -1.9599639845},
        {0.25, -0.6744897502},
        {0.5, 0.0},
        {0.75, 0.6744897502},
        {0.975, 1.9599639845},
        {0.99, 2.3263478740},
        {0.999, 3.0902323062}
      ]
      |> Enum.each(fn {p, want} ->
        assert_in_delta Internal.normal_quantile(p), want, 5.0e-9
      end)
    end

    test "is symmetric about the median" do
      Enum.each([0.01, 0.1, 0.3, 0.45], fn p ->
        assert_in_delta Internal.normal_quantile(p), -Internal.normal_quantile(1 - p), 1.0e-12
      end)
    end

    test "rejects anything that is not a probability" do
      for p <- [0.0, 1.0, -0.1, 1.5] do
        assert_raise ArgumentError, ~r/strictly between 0 and 1/, fn ->
          Internal.normal_quantile(p)
        end
      end
    end
  end

  describe "residual_distribution/3" do
    defp y_true, do: Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defp y_pred, do: Nx.tensor([1.5, 2.0, 2.0, 4.5, 5.5, 6.0])

    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.residual_distribution(y_true(), y_pred())
    end

    test "counts every residual exactly once" do
      counts = EvalViz.residual_distribution(y_true(), y_pred(), bins: 4) |> values()
      assert counts |> Enum.map(& &1["count"]) |> Enum.sum() == 6
    end

    test "takes the residual as prediction minus truth, like residuals/3" do
      # y_pred - y_true = [0.5, 0.0, -1.0, 0.5, 0.5, 0.0], so the span is -1..0.5
      rows = EvalViz.residual_distribution(y_true(), y_pred(), bins: 3) |> values()

      assert_in_delta rows |> Enum.map(& &1["lower"]) |> Enum.min(), -1.0, 1.0e-6
      assert_in_delta rows |> Enum.map(& &1["upper"]) |> Enum.max(), 0.5, 1.0e-6
    end

    test "puts the largest value in the last bin rather than past it" do
      values = Nx.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
      counts = Internal.bin_counts([0.0, 1.0, 2.0, 3.0, 4.0], Internal.bin_span([0.0, 4.0], 4), 4)

      assert length(counts) == 4
      assert counts |> Enum.map(&elem(&1, 2)) |> Enum.sum() == 5
      assert %VegaLite{} = EvalViz.qq_plot(values)
    end

    test "draws the zero a residual with no bias would centre on" do
      assert length(spec(EvalViz.residual_distribution(y_true(), y_pred()))["layer"]) == 2
    end

    test "zero_line: false leaves it out" do
      plot = EvalViz.residual_distribution(y_true(), y_pred(), zero_line: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "rejects inputs of different lengths" do
      assert_raise ArgumentError, ~r/same length/, fn ->
        EvalViz.residual_distribution(y_true(), Nx.tensor([1.0, 2.0]))
      end
    end
  end

  describe "qq_plot/2" do
    # scipy.stats.probplot on this sample gives these order statistic medians.
    defp sample do
      Nx.tensor(
        [2.31, -0.4, 1.05, -1.87, 0.22, 0.91, -0.63, 1.44, -2.10, 0.05, 0.77, -1.21],
        type: :f64
      )
    end

    test "returns a VegaLite spec" do
      assert %VegaLite{} = EvalViz.qq_plot(sample())
    end

    test "sorts the sample up the y axis" do
      assert EvalViz.qq_plot(sample()) |> values() |> Enum.map(& &1["sample"]) ==
               Enum.sort(Nx.to_flat_list(sample()))
    end

    test "places the points where scipy's probplot does" do
      want = [
        -1.5881546430,
        -1.0981497547,
        -0.7825592681,
        -0.5306911286,
        -0.3089235255,
        -0.1015340025,
        0.1015340025,
        0.3089235255,
        0.5306911286,
        0.7825592681,
        1.0981497547,
        1.5881546430
      ]

      EvalViz.qq_plot(sample())
      |> values()
      |> Enum.map(& &1["theoretical"])
      |> Enum.zip(want)
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 5.0e-9 end)
    end

    test "fits the reference line the way scipy does" do
      # scipy.stats.probplot returns slope 1.4385946653, intercept 0.045
      [p1, p2] = spec(EvalViz.qq_plot(sample()))["layer"] |> hd() |> get_in(["data", "values"])

      slope = (p2["y"] - p1["y"]) / (p2["x"] - p1["x"])
      assert_in_delta slope, 1.4385946653, 1.0e-8
      assert_in_delta p1["y"] - slope * p1["x"], 0.045, 1.0e-8
    end

    test "reference_line: false leaves the fit out" do
      assert length(spec(EvalViz.qq_plot(sample(), reference_line: false))["layer"]) == 1
    end

    test "needs at least two values" do
      assert_raise ArgumentError, ~r/at least two values/, fn ->
        EvalViz.qq_plot(Nx.tensor([1.0]))
      end
    end
  end
end
