defmodule Verdict.CalibrationTest do
  use ExUnit.Case, async: true

  defp y_true,
    do: Nx.tensor([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1])

  # f64 so the bin edges are hit exactly; see the precision test below
  defp y_prob do
    Nx.tensor(
      [
        0.1,
        0.2,
        0.8,
        0.9,
        0.15,
        0.7,
        0.85,
        0.6,
        0.3,
        0.05,
        0.95,
        0.25,
        0.55,
        0.75,
        0.4,
        0.1,
        0.65,
        0.35,
        0.9,
        0.7
      ],
      type: :f64
    )
  end

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp curve_layer(plot), do: spec(plot)["layer"] |> List.last()

  defp rounded(plot, field) do
    plot |> values() |> Enum.map(&Float.round(&1[field], 6))
  end

  describe "calibration_curve/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.calibration_curve(y_true(), y_prob())
    end

    test "matches scikit-learn's uniform binning" do
      # sklearn.calibration.calibration_curve(y, p, n_bins=5, strategy="uniform")
      plot = Verdict.calibration_curve(y_true(), y_prob(), bins: 5)

      assert rounded(plot, "predicted") == [0.12, 0.325, 0.575, 0.72, 0.9]
      assert rounded(plot, "observed") == [0.0, 0.0, 1.0, 1.0, 1.0]
    end

    test "matches scikit-learn's quantile binning" do
      # sklearn.calibration.calibration_curve(y, p, n_bins=5, strategy="quantile")
      plot = Verdict.calibration_curve(y_true(), y_prob(), bins: 5, strategy: :quantile)

      assert rounded(plot, "predicted") == [0.1, 0.275, 0.55, 0.7375, 0.9]
      assert rounded(plot, "observed") == [0.0, 0.0, 0.75, 1.0, 1.0]
    end

    test "a probability sitting on a bin edge follows the input's precision" do
      # 0.2 as f32 is 0.20000000298, just above the 0.2 edge, so it moves up a
      # bin. scikit-learn does the same when handed f32-rounded input.
      f32 = Nx.as_type(y_prob(), :f32)
      plot = Verdict.calibration_curve(y_true(), f32, bins: 5)

      assert rounded(plot, "predicted") == [0.1, 0.275, 0.475, 0.68, 0.88]
    end

    test "counts how many points landed in each bin" do
      plot = Verdict.calibration_curve(y_true(), y_prob(), bins: 5)

      assert plot |> values() |> Enum.map(& &1["count"]) |> Enum.sum() == 20
    end

    test "drops empty bins rather than plotting them at zero" do
      # everything sits in the bottom bin, so only one point should come back
      probs = Nx.tensor([0.01, 0.02, 0.03, 0.04], type: :f64)
      plot = Verdict.calibration_curve(Nx.tensor([0, 0, 1, 0]), probs, bins: 10)

      assert length(values(plot)) == 1
    end

    test "puts the Brier score in the series label" do
      plot = Verdict.calibration_curve([{"Model", y_true(), y_prob()}], bins: 5)
      assert [%{"series" => label} | _] = values(plot)
      assert label =~ "Brier = "
    end

    test "draws the perfect-calibration diagonal by default" do
      [reference, _curve] = spec(Verdict.calibration_curve(y_true(), y_prob()))["layer"]

      assert reference["data"]["values"] == [
               %{"x" => 0, "y" => 0},
               %{"x" => 1, "y" => 1}
             ]
    end

    test "perfect_line: false drops the diagonal" do
      plot = Verdict.calibration_curve(y_true(), y_prob(), perfect_line: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "pins both axes to 0..1, since both hold probabilities" do
      layer = curve_layer(Verdict.calibration_curve(y_true(), y_prob()))

      assert layer["encoding"]["x"]["scale"]["domain"] == [0, 1]
      assert layer["encoding"]["y"]["scale"]["domain"] == [0, 1]
    end

    test "compares models on the same axes" do
      other = Nx.multiply(y_prob(), 0.5)

      plot =
        Verdict.calibration_curve([
          {"A", y_true(), y_prob()},
          {"B", y_true(), other}
        ])

      assert curve_layer(plot)["encoding"]["color"]["field"] == "series"
    end

    test "rejects scores that are not probabilities" do
      assert_raise ArgumentError, ~r/between 0 and 1/, fn ->
        Verdict.calibration_curve(Nx.tensor([0, 1]), Nx.tensor([-3.0, 7.2]))
      end
    end

    test "rejects a y_true that is not binary" do
      assert_raise ArgumentError, ~r/only 0 and 1/, fn ->
        Verdict.calibration_curve(Nx.tensor([0, 1, 2]), Nx.tensor([0.1, 0.5, 0.9]))
      end
    end
  end
end
