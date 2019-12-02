package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Fit adapts a linear model to a dataset. X is the design matrix,
// y is the target data.
func Fit(X mat.Matrix, y []float64) []float64 {
	nrows, n := X.Dims()

	if nrows != len(y) {
		panic("Fit: Inconsistent number of rows in design matrix")
	}

	x := make([]float64, n)
	yvec := mat.NewVecDense(len(y), y)

	var svd mat.SVD
	svd.Factorize(X, mat.SVDThin)

	s := svd.Values(nil)
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)

	var uTdoty mat.Dense
	uTdoty.Mul(u.T(), yvec)

	invSigma := 0.0
	for i := 0; i < len(s); i++ {
		if math.Abs(s[i]) > 1e-6 {
			invSigma = 1.0 / s[i]
		} else {
			invSigma = 0.0
		}

		uTdoty.Set(i, 0, uTdoty.At(i, 0)*invSigma)
	}

	var xMat mat.Dense
	xMat.Mul(&v, &uTdoty)
	for i := 0; i < len(x); i++ {
		x[i] = xMat.At(i, 0)
	}
	return x
}

// PredictOne predicts the value given a set of coefficients (coeff)
func PredictOne(x []float64, coeff []float64) float64 {
	res := 0.0

	for i := 0; i < len(x); i++ {
		res += x[i] * coeff[i]
	}
	return res
}

// Predict predicts the outcome of many variables. Each row in the
// matrix X is considered to be one data point
func Predict(X mat.Matrix, coeff []float64) []float64 {
	m, _ := X.Dims()
	res := mat.NewVecDense(m, nil)
	res.MulVec(X, NewSliceVec(coeff))
	return res.RawVector().Data
}

// Rss calculates the residual sum of squares.
// X is the design matrix, coeff is an array with fitted
// coefficients and data is an array with the target data.
// The number of rows in the matrix X has to be the same as
// the length of data array
func Rss(X mat.Matrix, coeff []float64, data []float64) float64 {
	pred := Predict(X, coeff)

	if len(data) != len(pred) {
		panic("rss: Inconsistent number of data points given")
	}

	sumSq := 0.0
	for i := 0; i < len(data); i++ {
		sumSq += math.Pow(pred[i]-data[i], 2)
	}
	return sumSq
}
