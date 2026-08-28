defmodule Verdict.Theme do
  @moduledoc """
  The colours and marks every plot shares.

  Reference marks are the dashed grey lines that stand for "no skill" or "no
  error": the ROC diagonal, the residual zero line, the mean silhouette. Keeping
  them identical everywhere is what lets them read as background rather than as
  another series.
  """

  @primary "#4c78a8"
  @secondary "#f58518"
  @muted "#b0b0b0"
  @reference "#999999"
  @symbol "#4a4a4a"

  # Vega's category10, ordered so neighbouring clusters stay distinguishable.
  @categorical [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#eeca3b",
    "#bab0ac"
  ]

  @doc "The single colour used when a plot draws one undifferentiated series."
  def primary, do: @primary

  @doc "The second colour, for a plot that overlays exactly two things."
  def secondary, do: @secondary

  @doc "Grey for marks that are present but not the subject, such as a cut-off tree."
  def muted, do: @muted

  @doc "Grey for reference lines."
  def reference, do: @reference

  @doc """
  Distinct colours for `n` series, cycling once it runs out.
  """
  def categorical(n) when n > 0 do
    @categorical |> Stream.cycle() |> Enum.take(n)
  end

  @doc """
  Mark options for a reference line, so every plot draws them the same.
  """
  def reference_mark, do: [stroke_dash: [4, 4], color: @reference, size: 1]

  @doc """
  A `heatmap` colour scheme name.
  """
  def heatmap_scheme, do: "blues"

  @doc """
  Scheme for values that run either side of zero, such as PCA loadings, where a
  sequential ramp would hide the sign.
  """
  def diverging_scheme, do: "blueorange"

  @doc """
  Legend options for a dash-pattern channel.

  The swatch has to show the pattern rather than a colour, and the default
  filled dot shows neither. Outlining it and leaving it hollow lets the dash
  through.
  """
  def dash_legend do
    [symbol_stroke_color: @symbol, symbol_stroke_width: 2, symbol_fill_color: "transparent"]
  end
end
