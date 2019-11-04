package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const RSS_TOL = 1e-12

// Calculate the value of AIC given a number of selected
// featues (num_feat), number of data points (num_data) and
// the residual sum of squares (rss)
func aic(num_feat int, num_data int, rss float64) float64 {
	return 2.0*float64(num_feat) + float64(num_data)*math.Log(rss)
}

// Calculata a lower and upper bound of all sub-models. The bits up until
// start is common in all models. X is the total design matrix and y is the
// data points. The function return lower_bound, upper_bound
func boundsAIC(model []bool, start int, X *mat.Dense, y []float64) (float64, float64) {
	gcsMod := gcs(model, start)
	lcsMod := lcs(model, start)

	k_gcs := numFeatures(gcsMod)
	k_lcs := numFeatures(lcsMod)

	X_gcs := getDesignMatrix(gcsMod, X)
	X_lcs := getDesignMatrix(lcsMod, X)

	coeff_gcs := fit(X_gcs, y)
	coeff_lcs := fit(X_lcs, y)

	lower := aic(k_lcs, len(y), math.Max(RSS_TOL, rss(X_gcs, coeff_gcs, y)))
	upper := aic(k_gcs, len(y), math.Max(RSS_TOL, rss(X_lcs, coeff_lcs, y)))
	return lower, upper
}
