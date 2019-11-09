package featselect

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLazyPowerMat(t *testing.T) {
	for i, test := range []struct {
		X        *mat.Dense
		maxPower int
		Y        *mat.Dense
	}{
		{
			X:        mat.NewDense(2, 2, []float64{2.0, 2.0, 2.0, 2.0}),
			maxPower: 1,
			Y:        mat.NewDense(2, 2, []float64{2.0, 2.0, 2.0, 2.0}),
		},
		{
			X:        mat.NewDense(2, 2, []float64{1.0, 1.0, 2.0, 3.0}),
			maxPower: 2,
			Y: mat.NewDense(2, 5, []float64{1.0, 1.0, 1.0, 1.0, 1.0,
				2.0, 3.0, 9.0, 6.0, 4.0}),
		},
	} {
		lazy := NewLazyMatrix(test.X, test.maxPower).FullMatrix()
		if !mat.EqualApprox(lazy, test.Y, 1e-10) {
			t.Errorf("Test #%v failed. Expected:\n%v\nGot:\n%v\n", i, mat.Formatted(test.Y), mat.Formatted(lazy))
		}
	}
}
