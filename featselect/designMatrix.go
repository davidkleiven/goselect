package featselect

import "gonum.org/v1/gonum/mat"

// DesignMatrix provides an interface for types that can be used as design
// matrices
type DesignMatrix interface {
	// Dims return the number of rows and number of columns
	Dims() (int, int)

	// Returns a column view of the matrix
	ColView(col int) mat.Vector

	// Returns a row view of the matrix
	At(i int, j int) float64
}
