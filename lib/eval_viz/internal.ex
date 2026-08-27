defmodule EvalViz.Internal do
  @moduledoc false

  @doc """
  Turns a rank-1 tensor into a flat list of numbers.
  """
  def to_list(tensor), do: tensor |> Nx.to_flat_list()

  @doc """
  Zips two rank-1 tensors into a list of maps under the given keys.
  """
  def points(x, y, x_key, y_key) do
    Enum.zip_with(to_list(x), to_list(y), &%{x_key => &1, y_key => &2})
  end

  @doc """
  Labels for `n` classes: the caller's names when given, otherwise "0".."n-1".

  Names are stringified so Vega-Lite treats the axis as nominal, which keeps
  the categories in the order we hand them over rather than sorting numerically.
  """
  def class_labels(nil, n), do: Enum.map(0..(n - 1), &Integer.to_string/1)

  def class_labels(names, n) when length(names) == n do
    Enum.map(names, &to_string/1)
  end

  def class_labels(names, n) do
    raise ArgumentError,
          "expected :class_names to have one entry per class, " <>
            "got #{length(names)} names for #{n} classes"
  end

  @doc """
  Rounds for display without pulling in a formatting dependency.
  """
  def round_to(value, digits) when is_float(value), do: Float.round(value, digits)
  def round_to(value, _digits), do: value

  @doc """
  Raises unless both tensors are rank-1 and the same length.
  """
  def assert_paired!(a, b, a_name, b_name) do
    if Nx.rank(a) != 1 do
      raise ArgumentError,
            "expected #{a_name} to be a rank-1 tensor, got rank #{Nx.rank(a)}"
    end

    if Nx.rank(b) != 1 do
      raise ArgumentError,
            "expected #{b_name} to be a rank-1 tensor, got rank #{Nx.rank(b)}"
    end

    if Nx.axis_size(a, 0) != Nx.axis_size(b, 0) do
      raise ArgumentError,
            "expected #{a_name} and #{b_name} to have the same length, " <>
              "got #{Nx.axis_size(a, 0)} and #{Nx.axis_size(b, 0)}"
    end

    :ok
  end
end
