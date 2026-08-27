defmodule EvalViz.Multiclass do
  @moduledoc false

  alias EvalViz.Internal

  def multiclass?(y_score), do: Nx.rank(y_score) == 2

  def averages(opts), do: List.wrap(opts[:average])

  def macro?(y_score, opts), do: multiclass?(y_score) and :macro in averages(opts)

  @doc """
  One binary sub-series per class, so a plot written for binary input works
  unchanged on a `{n, k}` score matrix.
  """
  def per_class({label, y_true, y_score}, opts) do
    if multiclass?(y_score) do
      num_classes = Nx.axis_size(y_score, 1)
      assert_indices!(y_true, y_score, num_classes)

      opts[:class_names]
      |> Internal.class_labels(num_classes)
      |> Enum.with_index(fn name, index ->
        {compose(label, name), Nx.equal(y_true, index), y_score[[.., index]]}
      end)
    else
      [{label, y_true, y_score}]
    end
  end

  @doc """
  The micro-average sub-series: every (sample, class) pair pooled into a single
  binary problem, which is what one-hot encoding and flattening gives.
  """
  def micro({label, y_true, y_score}, opts) do
    if multiclass?(y_score) and :micro in averages(opts) do
      num_classes = Nx.axis_size(y_score, 1)
      one_hot = Nx.equal(Nx.new_axis(y_true, 1), Nx.iota({1, num_classes}))

      [{compose(label, "micro-average"), Nx.flatten(one_hot), Nx.flatten(y_score)}]
    else
      []
    end
  end

  def compose(nil, name), do: name
  def compose(label, name), do: "#{label}: #{name}"

  defp assert_indices!(y_true, y_score, num_classes) do
    if Nx.axis_size(y_true, 0) != Nx.axis_size(y_score, 0) do
      raise ArgumentError,
            "expected y_true to have one entry per row of y_score, " <>
              "got #{Nx.axis_size(y_true, 0)} for #{Nx.axis_size(y_score, 0)} rows"
    end

    distinct = y_true |> Nx.to_flat_list() |> Enum.uniq()

    unless Enum.all?(distinct, &(&1 >= 0 and &1 < num_classes and trunc(&1) == &1)) do
      raise ArgumentError,
            "expected y_true to hold class indices in 0..#{num_classes - 1}, one per " <>
              "column of y_score, got #{inspect(Enum.sort(distinct))}"
    end

    :ok
  end
end
