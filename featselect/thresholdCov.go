package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// ThresholdOperator is a type the is used to threshold matrices
type ThresholdOperator struct {
	threshold float64
}

// Apply sets all elements in X that is smaller than the
// threshold to 0
func (t *ThresholdOperator) Apply(X mat.Mutable) {
	nr, nc := X.Dims()

	for i := 0; i < nr; i++ {
		for j := 0; j < nc; j++ {
			if math.Abs(X.At(i, j)) < t.threshold {
				X.Set(i, j, 0.0)
			}
		}
	}
}
