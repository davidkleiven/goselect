package featselect

import "gonum.org/v1/gonum/mat"

//CovMat is a generic interface for returning covariance matrices
type CovMat interface {
	Get(X mat.Matrix) mat.Matrix
}

// Empirical returns the empirical covariance matrix
type Empirical struct{}

// Get returns the empirical covariance matrix
func (e *Empirical) Get(X mat.Matrix) mat.Matrix {
	nr, nc := X.Dims()
	cov := mat.NewDense(nc, nc, nil)
	cov.Product(X.T(), X)
	cov.Scale(1.0/float64(nr), cov)
	return cov
}
