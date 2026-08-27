defmodule EvalViz.MixProject do
  use Mix.Project

  @version "0.1.0-dev"
  @source_url "https://github.com/RicardoSantos-99/evalviz"

  def project do
    [
      app: :evalviz,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      docs: docs(),
      source_url: @source_url,
      name: "EvalViz"
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.13"},
      {:vega_lite, "~> 0.1.9"},
      {:nimble_options, "~> 1.0"},
      # local checkout while our fixes are merged upstream but not yet released
      {:scholar, path: "../scholar", override: true},
      {:jason, "~> 1.4", only: [:dev, :test]},
      {:vega_lite_convert, "~> 1.0", only: [:dev, :test]},
      {:ex_doc, "~> 0.34", only: :docs, runtime: false}
    ]
  end

  defp description do
    "Model evaluation plots for Nx: confusion matrices, ROC, precision-recall and DET curves."
  end

  defp docs do
    [
      main: "EvalViz",
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end
end
