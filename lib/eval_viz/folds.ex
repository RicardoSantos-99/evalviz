defmodule EvalViz.Folds do
  @moduledoc false

  @doc """
  Scores as one list of fold values per position, whether they arrived as one
  number per position or one per fold.
  """
  def normalize(%Nx.Tensor{} = tensor, expected, name, per) do
    case Nx.rank(tensor) do
      1 -> tensor |> Nx.to_flat_list() |> Enum.map(&[&1])
      2 -> Nx.to_list(tensor)
      rank -> raise ArgumentError, "expected #{name} to be rank 1 or 2, got rank #{rank}"
    end
    |> check_length(expected, name, per)
  end

  def normalize(list, expected, name, per) when is_list(list) do
    list
    |> Enum.map(fn
      scores when is_list(scores) -> scores
      score -> [score]
    end)
    |> check_length(expected, name, per)
  end

  @doc """
  A row per position carrying the mean and a band one standard deviation either
  side of it.
  """
  def rows(positions, folds, label, x_key) do
    Enum.zip_with(positions, folds, fn position, scores ->
      mean = Enum.sum(scores) / length(scores)
      deviation = deviation(scores, mean)

      %{
        x_key => position,
        "score" => mean,
        "lower" => mean - deviation,
        "upper" => mean + deviation,
        "series" => label
      }
    end)
  end

  @doc """
  Whether there is more than one score per position, and so a band to draw.
  """
  def folded?(folds), do: folds |> hd() |> length() > 1

  defp deviation(scores, _mean) when length(scores) < 2, do: 0.0

  defp deviation(scores, mean) do
    variance =
      scores
      |> Enum.map(&((&1 - mean) * (&1 - mean)))
      |> Enum.sum()
      |> Kernel./(length(scores))

    :math.sqrt(variance)
  end

  defp check_length(folds, expected, name, per) do
    if length(folds) != expected do
      raise ArgumentError,
            "expected #{name} to have one entry per #{per}, " <>
              "got #{length(folds)} for #{expected} of them"
    end

    folds
  end
end
