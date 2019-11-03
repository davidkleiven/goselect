package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Fits a linear model. X is the design matrix,
// y is the target data.
func fit(X *mat.Dense, y []float64) []float64 {
	_, n := X.Dims()
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

	inv_sigma := 0.0
	for i := 0; i < len(s); i++ {
		if math.Abs(s[i]) > 1e-6 {
			inv_sigma = 1.0 / s[i]
		} else {
			inv_sigma = 0.0
		}

		uTdoty.Set(i, 0, uTdoty.At(i, 0)*inv_sigma)
	}

	var xMat mat.Dense
	xMat.Mul(&v, &uTdoty)
	for i := 0; i < len(x); i++ {
		x[i] = xMat.At(i, 0)
	}
	return x
}

// Predicts the value given a set of coefficients (coeff)
func predictOne(x []float64, coeff []float64) float64 {
	res := 0.0

	for i := 0; i < len(x); i++ {
		res += x[i] * coeff[i]
	}
	return res
}

// Preducts the outcome of many variables. Each row in the
// matrix X is considered to be one data point
func predict(X *mat.Dense, coeff []float64) []float64 {
	m, _ := X.Dims()
	res := make([]float64, m)

	for i := 0; i < m; i++ {
		res[i] = predictOne(X.RawRowView(i), coeff)
	}
	return res
}
