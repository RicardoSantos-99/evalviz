defmodule Verdict.GridTest do
  use ExUnit.Case, async: true

  defp y_true, do: Nx.tensor([0, 0, 1, 1])
  defp scores, do: Nx.tensor([0.1, 0.4, 0.35, 0.8])

  defp roc, do: Verdict.roc_curve(y_true(), scores())
  defp pr, do: Verdict.precision_recall_curve(y_true(), scores())

  defp spec(plot), do: VegaLite.to_spec(plot)

  describe "grid/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.grid([roc(), pr()])
    end

    test "concatenates the plots it is given" do
      assert length(spec(Verdict.grid([roc(), pr()]))["concat"]) == 2
    end

    test "wraps at two columns by default" do
      assert spec(Verdict.grid([roc(), pr()]))["columns"] == 2
    end

    test "takes a column count" do
      assert spec(Verdict.grid([roc(), pr()], columns: 1))["columns"] == 1
    end

    test "titles the whole grid" do
      assert spec(Verdict.grid([roc()], title: "Held-out set"))["title"] == "Held-out set"
    end

    test "keeps each plot's own title" do
      titled = Verdict.roc_curve(y_true(), scores(), title: "ROC")
      [first] = spec(Verdict.grid([titled]))["concat"]

      assert first["title"]["text"] == "ROC"
    end

    # fold_scores is itself a concat, and Vega-Lite nests those happily.
    test "takes a plot that is already composed" do
      inner = Verdict.fold_scores(Nx.tensor([[0.8, 0.9], [1.0, 1.1]], type: :f64))

      assert length(spec(Verdict.grid([roc(), inner]))["concat"]) == 2
    end

    # VegaLite.config/2 sets a key Vega-Lite only allows on the outermost spec,
    # and the README tells people to pipe into it, so the error has to say
    # where it belongs rather than repeating Vega-Lite's wording.
    test "says where a top-level key belongs" do
      configured = VegaLite.config(roc(), axis: [grid: false])

      assert_raise ArgumentError, ~r/Apply it to the grid instead/, fn ->
        Verdict.grid([configured])
      end
    end

    test "rejects anything that is not a plot" do
      assert_raise ArgumentError, ~r/expected a VegaLite specification/, fn ->
        Verdict.grid([roc(), %{not: "a plot"}])
      end
    end

    test "rejects an empty grid" do
      assert_raise ArgumentError, ~r/at least one plot/, fn ->
        Verdict.grid([])
      end
    end
  end
end
