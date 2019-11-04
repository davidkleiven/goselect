package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const RSS_TOL = 1e-12

type crit func(num_feat int, num_data int, rss float64) float64

// Calculate the value of AIC given a number of selected
// featues (num_feat), number of data points (num_data) and
// the residual sum of squares (rss)
func aic(num_feat int, num_data int, rss float64) float64 {
	return 2.0*float64(num_feat) + float64(num_data)*math.Log(rss)
}

func aicc(num_feat int, num_data int, rss float64) float64 {
	if num_feat >= num_data-1 {
		panic("aicc: Too many features compared to the number of datapoints")
	}

	return aic(num_data, num_data, rss) + float64(2*num_feat*num_feat+2*num_feat)/float64(num_data-num_feat-1)
}

// Calculata a lower and upper bound of all sub-models. The bits up until
// start is common in all models. X is the total design matrix and y is the
// data points. criteria is a function that calculate a cost for instance aic.
// The function return lower_bound, upper_bound
func bounds(model []bool, start int, X *mat.Dense, y []float64, criteria crit) (float64, float64) {
	gcsMod := gcs(model, start)
	lcsMod := lcs(model, start)

	k_gcs := numFeatures(gcsMod)
	k_lcs := numFeatures(lcsMod)

	X_gcs := getDesignMatrix(gcsMod, X)
	X_lcs := getDesignMatrix(lcsMod, X)

	coeff_gcs := fit(X_gcs, y)
	coeff_lcs := fit(X_lcs, y)

	lower := criteria(k_lcs, len(y), math.Max(RSS_TOL, rss(X_gcs, coeff_gcs, y)))
	upper := criteria(k_gcs, len(y), math.Max(RSS_TOL, rss(X_lcs, coeff_lcs, y)))
	return lower, upper
}

func boundsAIC(model []bool, start int, X *mat.Dense, y []float64) (float64, float64) {
	return bounds(model, start, X, y, aic)
}

func boundsAICC(model []bool, start int, X *mat.Dense, y []float64) (float64, float64) {
	return bounds(model, start, X, y, aicc)
}
