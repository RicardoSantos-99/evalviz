defmodule Verdict.Internal do
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
  The option spec for per-sample weights, shared by every plot that takes them.
  """
  def weights_option do
    [
      type: {:or, [{:list, {:or, [:float, :integer]}}, :any]},
      doc: """
      Per-sample weights, as a list or rank-1 tensor. A weight of two counts a
      sample twice, exactly as duplicating the row would.
      """
    ]
  end

  @doc """
  Weights as a tensor, or `1.0` when there are none, checked against `count`.
  """
  def weights(nil, _count), do: 1.0

  def weights(weights, count) do
    tensor = if is_list(weights), do: Nx.tensor(weights), else: weights

    if Nx.rank(tensor) != 1 do
      raise ArgumentError,
            "expected :sample_weights to be rank 1, got rank #{Nx.rank(tensor)}"
    end

    if Nx.axis_size(tensor, 0) != count do
      raise ArgumentError,
            "expected one sample weight per sample, " <>
              "got #{Nx.axis_size(tensor, 0)} for #{count} samples"
    end

    tensor
  end

  @doc """
  Weights as a plain list of numbers, all ones when there are none.
  """
  def weight_list(nil, count), do: List.duplicate(1.0, count)

  def weight_list(weights, count) do
    weights |> weights(count) |> Nx.to_flat_list()
  end

  @doc """
  The span `count` equal bins need to cover `values`, as `{lower, width}`.

  Callers share one span across several series so their bars line up.
  """
  def bin_span(values, count) do
    {min, max} = Enum.min_max(values)
    width = if max > min, do: (max - min) / count, else: 1.0

    {min, width}
  end

  @doc """
  Counts `values` into `count` bins over `span`, as `[{lower, upper, count}]`.

  With `weights`, a bin holds the sum of the weights that landed in it, so a
  weight of two counts a sample twice.
  """
  def bin_counts(values, span, count), do: bin_counts(values, span, count, nil)

  def bin_counts(values, {lower, width}, count, weights) do
    tally =
      values
      |> Enum.zip(weights || List.duplicate(1, length(values)))
      |> Enum.reduce(%{}, fn {value, weight}, acc ->
        Map.update(acc, bin_index(value, lower, width, count), weight, &(&1 + weight))
      end)

    Enum.map(0..(count - 1), fn index ->
      {lower + index * width, lower + (index + 1) * width, Map.get(tally, index, 0)}
    end)
  end

  # The largest value divides exactly, which would put it one bin past the end.
  defp bin_index(value, lower, width, count) do
    ((value - lower) / width) |> trunc() |> max(0) |> min(count - 1)
  end

  @a [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00
  ]

  @b [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01
  ]

  @c [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00
  ]

  @d [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00
  ]

  @low 0.02425

  @doc """
  The value a standard normal falls below with probability `p`.

  Acklam's rational approximation, whose relative error stays under 1.15e-9.
  Nx has no inverse normal CDF, and a Q-Q plot cannot be drawn without one.
  """
  def normal_quantile(p) when p <= 0 or p >= 1 do
    raise ArgumentError, "expected a probability strictly between 0 and 1, got #{inspect(p)}"
  end

  def normal_quantile(p) when p < @low do
    q = :math.sqrt(-2 * :math.log(p))
    tail(q)
  end

  def normal_quantile(p) when p > 1 - @low do
    q = :math.sqrt(-2 * :math.log(1 - p))
    -tail(q)
  end

  def normal_quantile(p) do
    [a1, a2, a3, a4, a5, a6] = @a
    [b1, b2, b3, b4, b5] = @b

    q = p - 0.5
    r = q * q

    (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
      (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
  end

  defp tail(q) do
    [c1, c2, c3, c4, c5, c6] = @c
    [d1, d2, d3, d4] = @d

    (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
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
