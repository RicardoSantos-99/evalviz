defmodule Verdict.ZoomTest do
  use ExUnit.Case, async: true

  defp y_true, do: Nx.tensor([0, 0, 1, 1, 0, 1, 1, 0])
  defp scores, do: Nx.tensor([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.3])
  defp embedding, do: Nx.tensor([[0.0, 0.0], [1.0, 0.5], [5.0, 5.0], [5.5, 4.8]])

  defp spec(plot), do: VegaLite.to_spec(plot)

  defp views(%{} = spec) do
    children =
      Enum.flat_map(~w(layer concat hconcat vconcat spec), fn key ->
        case spec[key] do
          views when is_list(views) -> views
          view when is_map(view) -> [view]
          nil -> []
        end
      end)

    [spec | Enum.flat_map(children, &views/1)]
  end

  defp params(plot) do
    plot
    |> spec()
    |> views()
    |> Enum.flat_map(&(&1["params"] || []))
    |> Enum.filter(&(&1["name"] == "verdict_zoom"))
  end

  describe "zoomable/2" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.zoomable(Verdict.projection(embedding()))
    end

    test "binds a drag to the scales" do
      [param] = params(Verdict.zoomable(Verdict.projection(embedding())))

      assert param["bind"] == "scales"
      assert param["select"] == %{"type" => "interval", "encodings" => ["x", "y"]}
    end

    test "moves both axes by default" do
      [param] = params(Verdict.zoomable(Verdict.roc_curve(y_true(), scores())))
      assert param["select"]["encodings"] == ["x", "y"]
    end

    test "pins the axis it is not given" do
      plot = Verdict.zoomable(Verdict.roc_curve(y_true(), scores()), encodings: [:x])
      [param] = params(plot)

      assert param["select"]["encodings"] == ["x"]
    end

    # Layers share one pair of scales, and a second copy of the parameter
    # collides on the signal the binding generates, which Vega-Lite rejects.
    test "gives a layered plot exactly one parameter" do
      assert length(params(Verdict.zoomable(Verdict.roc_curve(y_true(), scores())))) == 1
      assert length(params(Verdict.zoomable(Verdict.calibration_curve(y_true(), scores())))) == 1
    end

    test "never puts the parameter on the parent of a layer" do
      plot = Verdict.zoomable(Verdict.roc_curve(y_true(), scores()))
      refute spec(plot)["params"]
    end

    # The reference line is drawn first and carries only the axis it spans, so
    # taking the first layer on sight would bind a scale that is not there.
    test "picks the layer that has the axes it was asked for" do
      y_true = Nx.tensor([1.0, 2.0, 3.0])
      y_pred = Nx.tensor([1.1, 1.9, 3.2])

      layers = spec(Verdict.zoomable(Verdict.residuals(y_true, y_pred)))["layer"]
      [{layer, index}] = Enum.with_index(layers) |> Enum.filter(fn {l, _} -> l["params"] end)

      assert index == 1
      assert layer["encoding"]["x"]["type"] == "quantitative"
      assert layer["encoding"]["y"]["type"] == "quantitative"
    end

    test "zooms every panel of a concatenated plot" do
      plot =
        Verdict.projection(embedding(), [
          {"True class", Nx.tensor([0, 0, 1, 1])},
          {"KMeans", Nx.tensor([1, 0, 1, 1])}
        ])

      assert length(params(Verdict.zoomable(plot))) == 2
    end

    test "zooms every plot in a grid" do
      grid =
        Verdict.grid([Verdict.roc_curve(y_true(), scores()), Verdict.projection(embedding())])

      assert length(params(Verdict.zoomable(grid))) == 2
    end

    # A report puts a confusion matrix beside three continuous plots. Vega-Lite
    # binds a drag to a scale only where that scale is continuous.
    test "leaves the categorical views of a report alone" do
      zoomed = Verdict.zoomable(Verdict.report(y_true(), scores()))

      assert length(spec(zoomed)["concat"]) == 4
      assert length(params(zoomed)) == 3
    end

    test "rejects a plot that puts both axes on categories" do
      matrix =
        Verdict.confusion_matrix(y_true(), Nx.tensor([0, 1, 1, 1, 0, 1, 0, 0]), num_classes: 2)

      assert_raise ArgumentError, ~r/no view in this plot can be zoomed on x and y/, fn ->
        Verdict.zoomable(matrix)
      end
    end

    test "rejects being applied twice" do
      assert_raise ArgumentError, ~r/already zoomable/, fn ->
        Verdict.projection(embedding()) |> Verdict.zoomable() |> Verdict.zoomable()
      end
    end

    test "finds an earlier application nested in a concatenation" do
      assert_raise ArgumentError, ~r/already zoomable/, fn ->
        Verdict.report(y_true(), scores()) |> Verdict.zoomable() |> Verdict.zoomable()
      end
    end

    test "rejects being given no axis to move" do
      assert_raise ArgumentError, ~r/at least one axis/, fn ->
        Verdict.zoomable(Verdict.projection(embedding()), encodings: [])
      end
    end

    test "keeps a parameter the caller added" do
      plot =
        Verdict.projection(embedding())
        |> VegaLite.param("mine", value: 3)
        |> Verdict.zoomable()

      assert spec(plot)["params"] |> Enum.map(& &1["name"]) == ["mine", "verdict_zoom"]
    end
  end
end
