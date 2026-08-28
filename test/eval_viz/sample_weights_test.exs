defmodule EvalViz.SampleWeightsTest do
  use ExUnit.Case, async: true

  # A weight of two means "count this sample twice", so every plot that takes
  # weights has to agree with the same data holding that row twice. That is the
  # definition, and it pins the behaviour harder than any reference value would.
  defp y_true, do: Nx.tensor([0, 0, 1, 1, 1])
  defp y_pred, do: Nx.tensor([0, 1, 1, 1, 0])
  defp scores, do: Nx.tensor([0.1, 0.4, 0.35, 0.8, 0.6], type: :f64)
  defp weights, do: [2, 1, 1, 1, 1]

  defp doubled_true, do: Nx.tensor([0, 0, 0, 1, 1, 1])
  defp doubled_pred, do: Nx.tensor([0, 0, 1, 1, 1, 0])
  defp doubled_scores, do: Nx.tensor([0.1, 0.1, 0.4, 0.35, 0.8, 0.6], type: :f64)

  defp values(plot), do: VegaLite.to_spec(plot)["data"]["values"]

  defp assert_same(weighted, duplicated, field) do
    got = weighted |> values() |> Enum.map(& &1[field])
    want = duplicated |> values() |> Enum.map(& &1[field])

    assert length(got) == length(want)

    Enum.zip(got, want)
    |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-9 end)
  end

  describe "weighting equals duplicating" do
    test "confusion matrix" do
      assert_same(
        EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 2, sample_weights: weights()),
        EvalViz.confusion_matrix(doubled_true(), doubled_pred(), num_classes: 2),
        "value"
      )
    end

    test "roc curve" do
      assert_same(
        EvalViz.roc_curve(y_true(), scores(), sample_weights: weights()),
        EvalViz.roc_curve(doubled_true(), doubled_scores()),
        "tpr"
      )
    end

    test "precision-recall curve" do
      assert_same(
        EvalViz.precision_recall_curve(y_true(), scores(), sample_weights: weights()),
        EvalViz.precision_recall_curve(doubled_true(), doubled_scores()),
        "precision"
      )
    end

    test "threshold curve" do
      assert_same(
        EvalViz.threshold_curve(y_true(), scores(), sample_weights: weights()),
        EvalViz.threshold_curve(doubled_true(), doubled_scores()),
        "value"
      )
    end

    test "calibration curve" do
      weighted = EvalViz.calibration_curve(y_true(), scores(), bins: 2, sample_weights: weights())
      duplicated = EvalViz.calibration_curve(doubled_true(), doubled_scores(), bins: 2)

      assert_same(weighted, duplicated, "observed")
      assert_same(weighted, duplicated, "predicted")
      assert_same(weighted, duplicated, "count")
    end

    test "score distribution" do
      assert_same(
        EvalViz.score_distribution(y_true(), scores(), bins: 3, sample_weights: weights()),
        EvalViz.score_distribution(doubled_true(), doubled_scores(), bins: 3),
        "value"
      )
    end
  end

  describe "what the weights change" do
    test "the confusion matrix counts the weight, not the rows" do
      total =
        EvalViz.confusion_matrix(y_true(), y_pred(), num_classes: 2, sample_weights: weights())
        |> values()
        |> Enum.map(& &1["value"])
        |> Enum.sum()

      assert total == 6
    end

    # A class still sums to one, but on the weight it carries rather than on how
    # many rows it has.
    test "the score distribution normalizes on the weight a class carries" do
      shares =
        EvalViz.score_distribution(y_true(), scores(), bins: 3, sample_weights: weights())
        |> values()
        |> Enum.group_by(& &1["class"], & &1["value"])
        |> Enum.map(fn {class, values} -> {class, Enum.sum(values)} end)

      Enum.each(shares, fn {_class, sum} -> assert_in_delta sum, 1.0, 1.0e-9 end)
    end

    test "the Brier score is weighted too" do
      weighted =
        EvalViz.calibration_curve([{"m", y_true(), scores()}], bins: 2, sample_weights: weights())

      duplicated = EvalViz.calibration_curve([{"m", doubled_true(), doubled_scores()}], bins: 2)

      assert values(weighted) |> Enum.map(& &1["series"]) ==
               values(duplicated) |> Enum.map(& &1["series"])
    end
  end

  describe "rejections" do
    test "weights that do not line up with the samples" do
      assert_raise ArgumentError, ~r/one sample weight per sample/, fn ->
        EvalViz.roc_curve(y_true(), scores(), sample_weights: [1, 2])
      end
    end

    test "weights that are not a flat list" do
      assert_raise ArgumentError, ~r/:sample_weights to be rank 1/, fn ->
        EvalViz.roc_curve(y_true(), scores(), sample_weights: Nx.broadcast(1.0, {5, 2}))
      end
    end

    # The micro average holds one row per (sample, class) pair rather than per
    # sample, so the caller's weights no longer describe its rows.
    test "weights alongside the micro average" do
      multi = Nx.tensor([0, 1, 2])
      matrix = Nx.tensor([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], type: :f64)

      assert_raise ArgumentError, ~r/cannot be combined with the micro average/, fn ->
        EvalViz.roc_curve(multi, matrix, average: :micro, sample_weights: [1, 2, 1])
      end
    end

    test "but per-class curves take weights fine" do
      multi = Nx.tensor([0, 1, 2])
      matrix = Nx.tensor([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], type: :f64)

      assert %VegaLite{} = EvalViz.roc_curve(multi, matrix, sample_weights: [1, 2, 1])
    end
  end
end
