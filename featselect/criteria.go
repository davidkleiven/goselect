package featselect

import (
	"math"
)

// RssTol is the minimum residual sum of squares allowed. It is introduced to avoid problems inside math.Log
const RssTol = 1e-12

type crit func(num_feat int, num_data int, logL float64) float64

// Aic calculates the value of AIC given a number of selected
// featues (num_feat), number of data points (num_data) and
// the log of the likelihood function
func Aic(numFeat int, numData int, logL float64) float64 {
	return 2.0*float64(numFeat) + float64(numData)*math.Log(logL)
}

// Aicc calculates the corrected Aic which is more accurate
// when the sample size is small
func Aicc(numFeat int, numData int, logL float64) float64 {
	denum := float64(numData - numFeat - 1)
	if numFeat >= numData-1 {
		denum = 1.0
	}

	return Aic(numFeat, numData, logL) + float64(2*numFeat*numFeat+2*numFeat)/denum
}

// Calculata a lower and upper bound of all sub-models. The bits up until
// start is common in all models. X is the total design matrix and y is the
// data points. criteria is a function that calculate a cost for instance aic.
// The function return lower_bound, upper_bound
func bounds(model []bool, start int, X DesignMatrix, y []float64, criteria crit) (float64, float64) {
	gcsMod := Gcs(model, start)
	lcsMod := Lcs(model, start)

	kGcs := NumFeatures(gcsMod)
	kLcs := NumFeatures(lcsMod)

	XGcs := GetDesignMatrix(gcsMod, X)
	XLcs := GetDesignMatrix(lcsMod, X)

	coeffGcs := Fit(XGcs, y)
	coeffLcs := Fit(XLcs, y)

	lower := criteria(kLcs, len(y), Rss(XGcs, coeffGcs, y))
	upper := criteria(kGcs, len(y), Rss(XLcs, coeffLcs, y))
	return lower, upper
}

// BoundsAIC calculates lower and upper bound for AIC for all sub-models of the passed model
func BoundsAIC(model []bool, start int, X DesignMatrix, y []float64) (float64, float64) {
	return bounds(model, start, X, y, Aic)
}

// BoundsAICC calculats lower and upper bound for AICC for all sub-models of the passed model
func BoundsAICC(model []bool, start int, X DesignMatrix, y []float64) (float64, float64) {
	return bounds(model, start, X, y, Aicc)
}
