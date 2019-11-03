package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Calculate the value of AIC given a number of selected
// featues (num_feat), number of data points (num_data) and
// the residual sum of squares (rss)
func aic(num_feat int, num_data int, rss float64) float64 {
	return 2.0*float64(num_feat) + float64(num_data)*math.Log(rss)
}

func upper_aic(model []bool, X *mat.Dense, y []float64) float64 {
	return 0.0
}
