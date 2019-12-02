package featselect

import (
	"gonum.org/v1/gonum/mat"
)

// SliceVec is a wrapper around a slice which satisfies the mat.Vector interface
// It does not allocate a new slice for the data (which mat.Vector does)
type SliceVec struct {
	data []float64
}

// AtVec returns the element at position i
func (s *SliceVec) AtVec(i int) float64 {
	return s.data[i]
}

// Len returns the length of the vector
func (s *SliceVec) Len() int {
	return len(s.data)
}

// At returns the i-th element of the vector
func (s *SliceVec) At(i, j int) float64 {
	return s.data[i]
}

// Dims returns the dimension of the a column vector
func (s *SliceVec) Dims() (int, int) {
	return len(s.data), 1
}

// T returns the transpose of the vector (i.e. a column vector)
func (s *SliceVec) T() mat.Matrix {
	return mat.Transpose{s}
}

// NewSliceVec returns a new slice vector
func NewSliceVec(d []float64) *SliceVec {
	var s SliceVec
	s.data = d
	return &s
}
