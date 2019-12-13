package featselect

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

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

// Identity use returns the identity matrix as the covariance
type Identity struct{}

// Get returns the identity matrix
func (i *Identity) Get(X mat.Matrix) mat.Matrix {
	_, nc := X.Dims()
	ones := make([]float64, nc)
	for i := range ones {
		ones[i] = 1.0
	}
	res := mat.NewDiagDense(nc, ones)
	return res
}

// SparseThresholded is a type that uses a sparse threshold algorithm to
// make a sparse approximation of the covariance matrix
type SparseThresholded struct {
	X  mat.Matrix
	op *ThresholdOperator
}

// NewSparseThreshold constructs a new sinstance of SparseThresholded
// covariance matrix
func NewSparseThreshold(X mat.Matrix) *SparseThresholded {
	var sp SparseThresholded
	sp.X = X
	fmt.Printf("Searching for a L2 consistent sparse approx...\n")
	sp.op = L2ConsistentCovTO(X, 1000, 1.0, 1.0)
	fmt.Printf("Optimal threshold %f\n", sp.op.threshold)
	return &sp
}

// Get returns the covariance matrix
func (s *SparseThresholded) Get(X mat.Matrix) mat.Matrix {
	if !mat.EqualApprox(X, s.X, 1e-10) {
		panic("Cov matrix of a matrix that is different from the one used during initialisation was passed!")
	}
	nr, nc := X.Dims()
	cov := mat.NewDense(nc, nc, nil)
	cov.Product(X.T(), X)
	cov.Scale(1.0/float64(nr), cov)
	s.op.Apply(cov)
	return cov
}
