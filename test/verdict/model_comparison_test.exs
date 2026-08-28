defmodule Verdict.ModelComparisonTest do
  use ExUnit.Case, async: true

  defp y_true, do: Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f64)
  defp linear, do: Nx.tensor([1.1, 2.1, 2.9, 4.2], type: :f64)
  defp ridge, do: Nx.tensor([1.4, 1.8, 3.4, 3.6], type: :f64)

  defp two_models, do: [{"Linear", y_true(), linear()}, {"Ridge", y_true(), ridge()}]

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp scatter(plot), do: spec(plot)["layer"] |> List.last()
  defp series(plot), do: plot |> values() |> Enum.map(& &1["series"]) |> Enum.uniq()

  describe "predicted_vs_actual/3 comparing models" do
    test "carries every model's points" do
      assert length(values(Verdict.predicted_vs_actual(two_models()))) == 8
      assert series(Verdict.predicted_vs_actual(two_models())) == ["Linear", "Ridge"]
    end

    test "colours by model" do
      encoding = scatter(Verdict.predicted_vs_actual(two_models()))["encoding"]

      assert encoding["color"]["field"] == "series"
      assert encoding["color"]["scale"]["domain"] == ["Linear", "Ridge"]
    end

    # The whole point of overlaying them is reading distance from one diagonal,
    # so a range fitted to only the first model would misreport the second.
    test "fits one range to every model" do
      only_linear = Verdict.predicted_vs_actual(y_true(), linear())
      both = Verdict.predicted_vs_actual(two_models())

      [_, linear_max] = scatter(only_linear)["encoding"]["x"]["scale"]["domain"]
      [both_min, both_max] = scatter(both)["encoding"]["x"]["scale"]["domain"]

      # Ridge reaches 1.4 at the bottom and stops at 3.6, Linear runs 1.1..4.2
      assert both_max >= linear_max
      assert both_min <= 1.1
    end

    test "puts the diagonal across that same range" do
      plot = Verdict.predicted_vs_actual(two_models())
      [diagonal, scatter] = spec(plot)["layer"]

      domain = scatter["encoding"]["x"]["scale"]["domain"]
      assert diagonal["data"]["values"] |> Enum.map(& &1["x"]) == domain
    end

    test "leaves a single model exactly as it was" do
      plot = Verdict.predicted_vs_actual(y_true(), linear())

      refute scatter(plot)["encoding"]["color"]
      refute plot |> values() |> hd() |> Map.has_key?("series")
    end
  end

  describe "residuals/3 comparing models" do
    test "carries every model's residuals" do
      assert length(values(Verdict.residuals(two_models()))) == 8
      assert series(Verdict.residuals(two_models())) == ["Linear", "Ridge"]
    end

    test "still takes the residual as prediction minus truth" do
      linear_residuals =
        Verdict.residuals(two_models())
        |> values()
        |> Enum.filter(&(&1["series"] == "Linear"))
        |> Enum.map(& &1["residual"])

      Enum.zip(linear_residuals, [0.1, 0.1, -0.1, 0.2])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-9 end)
    end

    test "colours by model" do
      assert scatter(Verdict.residuals(two_models()))["encoding"]["color"]["field"] == "series"
    end

    test "keeps one zero line for all of them" do
      [zero, _scatter] = spec(Verdict.residuals(two_models()))["layer"]
      assert zero["data"]["values"] == [%{"zero" => 0}]
    end

    test "leaves a single model exactly as it was" do
      plot = Verdict.residuals(y_true(), linear())

      refute scatter(plot)["encoding"]["color"]
      refute plot |> values() |> hd() |> Map.has_key?("series")
    end
  end

  describe "learning_curve/4 comparing models" do
    defp sizes, do: [10, 20, 40]

    defp curves do
      [
        {"Linear", [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]], [[0.7, 0.7], [0.8, 0.8], [0.85, 0.85]]},
        {"Forest", [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[0.6, 0.6], [0.75, 0.75], [0.88, 0.88]]}
      ]
    end

    defp line(plot), do: spec(plot)["layer"] |> List.last()

    test "tags every row with its model" do
      plot = Verdict.learning_curve(sizes(), curves())

      assert plot |> values() |> Enum.map(& &1["model"]) |> Enum.uniq() == ["Linear", "Forest"]
      assert length(values(plot)) == 2 * 2 * 3
    end

    # Two things have to be told apart at once, so colour takes the model and
    # the dash pattern takes training against validation.
    test "moves colour onto the model and the split onto the dash" do
      encoding = line(Verdict.learning_curve(sizes(), curves()))["encoding"]

      assert encoding["color"]["field"] == "model"
      assert encoding["color"]["scale"]["domain"] == ["Linear", "Forest"]
      assert encoding["strokeDash"]["field"] == "series"
      assert encoding["strokeDash"]["scale"]["domain"] == ["Training", "Validation"]
    end

    # A filled dot shows no pattern, so every dash entry came out an identical
    # blob until the swatch was outlined and left hollow.
    test "gives the dash legend a swatch that shows the pattern" do
      legend = line(Verdict.learning_curve(sizes(), curves()))["encoding"]["strokeDash"]["legend"]

      assert legend["symbolFillColor"] == "transparent"
      assert legend["symbolStrokeColor"]
    end

    # An area takes no dash pattern, so without `detail` the two bands of one
    # model would merge into a single shape spanning both curves.
    test "splits the bands on detail, which an area can carry" do
      [band, _line] = spec(Verdict.learning_curve(sizes(), curves()))["layer"]

      assert band["mark"]["type"] == "area"
      assert band["encoding"]["color"]["field"] == "model"
      assert band["encoding"]["detail"]["field"] == "series"
    end

    test "spread: false drops the bands, which is worth it with several models" do
      plot = Verdict.learning_curve(sizes(), curves(), spread: false)
      assert length(spec(plot)["layer"]) == 1
    end

    test "leaves a single model exactly as it was" do
      plot =
        Verdict.learning_curve(
          sizes(),
          [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]],
          [[0.7, 0.7], [0.8, 0.8], [0.85, 0.85]]
        )

      assert line(plot)["encoding"]["color"]["field"] == "series"
      refute line(plot)["encoding"]["strokeDash"]
      refute plot |> values() |> hd() |> Map.has_key?("model")
    end

    test "takes one score per size per model" do
      plot =
        Verdict.learning_curve(sizes(), [
          {"Linear", [0.9, 0.9, 0.9], [0.7, 0.8, 0.85]}
        ])

      assert length(spec(plot)["layer"]) == 1
    end

    test "rejects scores that do not line up, naming the model's argument" do
      assert_raise ArgumentError, ~r/train_scores to have one entry per training size/, fn ->
        Verdict.learning_curve(sizes(), [{"Linear", [0.9], [0.7, 0.8, 0.85]}])
      end
    end
  end
end
