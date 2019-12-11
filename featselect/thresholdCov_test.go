package featselect

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMatrixThreshold(t *testing.T) {
	for i, test := range []struct {
		X         *mat.Dense
		expect    *mat.Dense
		threshold float64
	}{
		{
			X:         mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
			expect:    mat.NewDense(2, 2, []float64{0.0, 0.0, 3.0, 4.0}),
			threshold: 2.5,
		},
	} {
		var op ThresholdOperator
		op.threshold = test.threshold
		op.Apply(test.X)

		if !mat.EqualApprox(test.X, test.expect, 1e-10) {
			t.Errorf("Test #%d: Expect\n%v\nGot\n%v\n", i, mat.Formatted(test.X), mat.Formatted(test.expect))
		}
	}
}
