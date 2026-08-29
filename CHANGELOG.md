# Changelog

## v0.1.0

First release.

  * Classification: confusion matrix, ROC, precision-recall, DET, calibration,
    threshold curve, and score distribution by true class
  * Regression: predicted against actual, residuals, residual distribution,
    and normal Q-Q
  * Clustering and decomposition: dendrogram, silhouette, elbow, PCA scree,
    2-D projection, biplot, and loadings heatmap
  * Model selection: learning curve, validation curve, grid search heatmap,
    and per-fold scores
  * Features: coefficients and correlation matrix
  * `Verdict.report/3` for a whole screen in one call, `Verdict.grid/2` to
    compose any set of plots
  * One-vs-rest multiclass with micro and macro averages, and faceting when
    the overlay gets crowded
  * Model comparison across the curves, calibration, regression plots, and
    learning curves
  * Sample weights across the plots that count
  * `Verdict.zoomable/2` to drag and scroll a plot's axes, applied to every
    view of a screen at once
