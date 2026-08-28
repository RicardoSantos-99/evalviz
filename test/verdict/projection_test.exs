defmodule Verdict.ProjectionTest do
  use ExUnit.Case, async: true

  defp embedding, do: Nx.tensor([[0.0, 0.0], [1.0, 0.5], [5.0, 5.0], [5.5, 4.8]])
  defp labels, do: Nx.tensor([0, 0, 1, 1])

  defp spec(plot), do: VegaLite.to_spec(plot)
  defp values(plot), do: spec(plot)["data"]["values"]
  defp encoding(plot), do: spec(plot)["encoding"]

  describe "projection/3" do
    test "returns a VegaLite spec" do
      assert %VegaLite{} = Verdict.projection(embedding())
    end

    test "draws one point per row" do
      assert length(values(Verdict.projection(embedding()))) == 4
    end

    test "takes the coordinates from the chosen columns" do
      rows = values(Verdict.projection(embedding()))

      Enum.zip(Enum.map(rows, & &1["x"]), [0.0, 1.0, 5.0, 5.5])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)

      Enum.zip(Enum.map(rows, & &1["y"]), [0.0, 0.5, 5.0, 4.8])
      |> Enum.each(fn {got, want} -> assert_in_delta got, want, 1.0e-6 end)
    end

    test "reads a struct that carries the embedding, as MDS does" do
      from_struct = values(Verdict.projection(%{embedding: embedding()}))
      assert from_struct == values(Verdict.projection(embedding()))
    end

    test "picks out the components it is given" do
      wide = Nx.tensor([[0.0, 9.0, 1.0], [2.0, 9.0, 3.0]])
      plot = Verdict.projection(wide, components: [0, 2])

      assert values(plot) |> Enum.map(& &1["y"]) == [1.0, 3.0]
      assert encoding(plot)["y"]["title"] == "Component 2"
    end

    test "leaves colour alone when there are no labels" do
      refute encoding(Verdict.projection(embedding()))["color"]
    end

    test "colours by label" do
      plot = Verdict.projection(embedding(), labels())

      assert values(plot) |> Enum.map(& &1["label"]) == ["0", "0", "1", "1"]
      assert encoding(plot)["color"]["field"] == "label"
    end

    test "names the labels when asked" do
      plot = Verdict.projection(embedding(), labels(), label_names: ["left", "right"])
      assert values(plot) |> Enum.map(& &1["label"]) == ["left", "left", "right", "right"]
    end

    # The two directions of an embedding are the same kind of quantity, so
    # letting them scale apart would misreport the shape of the data.
    test "gives both axes one range" do
      plot = Verdict.projection(embedding())

      assert encoding(plot)["x"]["scale"]["domain"] == encoding(plot)["y"]["scale"]["domain"]
      [min, max] = encoding(plot)["x"]["scale"]["domain"]
      assert min < 0.0 and max > 5.5
    end

    test "lets the axes scale apart when told to" do
      plot = Verdict.projection(embedding(), equal_axes: false)

      refute encoding(plot)["x"]["scale"]["domain"]
      assert encoding(plot)["x"]["scale"]["zero"] == false
    end

    test "rejects an embedding that is not a matrix" do
      assert_raise ArgumentError, ~r/rank-2 tensor/, fn ->
        Verdict.projection(Nx.tensor([1.0, 2.0, 3.0]))
      end
    end

    test "rejects components the embedding does not have" do
      assert_raise ArgumentError, ~r/index the embedding's 2 columns/, fn ->
        Verdict.projection(embedding(), components: [0, 4])
      end
    end

    test "rejects labels that do not line up with the rows" do
      assert_raise ArgumentError, ~r/one entry per row/, fn ->
        Verdict.projection(embedding(), Nx.tensor([0, 1]))
      end
    end
  end

  describe "side by side" do
    defp panels do
      Verdict.projection(embedding(), [
        {"True class", labels()},
        {"KMeans", Nx.tensor([1, 0, 1, 1])}
      ])
    end

    test "concatenates one panel per labelling" do
      assert length(spec(panels())["hconcat"]) == 2
    end

    test "titles each panel" do
      assert spec(panels())["hconcat"] |> Enum.map(& &1["title"]) == ["True class", "KMeans"]
    end

    test "shares the axis range across panels, so the clouds line up" do
      [left, right] = spec(panels())["hconcat"]

      assert left["encoding"]["x"]["scale"]["domain"] ==
               right["encoding"]["x"]["scale"]["domain"]
    end

    test "gives each panel its own labels" do
      [left, right] = spec(panels())["hconcat"]

      assert left["data"]["values"] |> Enum.map(& &1["label"]) == ["0", "0", "1", "1"]
      assert right["data"]["values"] |> Enum.map(& &1["label"]) == ["1", "0", "1", "1"]
    end

    test "builds each panel's colour scale from its own labels" do
      plot =
        Verdict.projection(embedding(), [
          {"Two groups", labels()},
          {"Four groups", Nx.tensor([0, 1, 2, 3])}
        ])

      [left, right] = spec(plot)["hconcat"]

      assert left["encoding"]["color"]["scale"]["domain"] == ["0", "1"]
      assert right["encoding"]["color"]["scale"]["domain"] == ["0", "1", "2", "3"]
    end

    test "keeps its own title above the panels" do
      plot = Verdict.projection(embedding(), [{"A", labels()}, {"B", labels()}], title: "Both")
      assert spec(plot)["title"] == "Both"
    end

    test "uses the panel title when only one is given" do
      plot = Verdict.projection(embedding(), [{"Only", labels()}])

      assert spec(plot)["title"] == "Only"
      refute spec(plot)["hconcat"]
    end

    test "rejects a panel that is not a {title, labels} pair" do
      assert_raise ArgumentError, ~r/expected \{title, labels\}/, fn ->
        Verdict.projection(embedding(), [labels()])
      end
    end
  end
end
