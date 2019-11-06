package featselect

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestExtractDesignMatirx(t *testing.T) {
	for _, test := range []struct {
		X      *mat.Dense
		model  []bool
		expect *mat.Dense
	}{
		{
			X:      mat.NewDense(3, 2, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
			model:  []bool{false, true},
			expect: mat.NewDense(3, 1, []float64{2.0, 4.0, 6.0}),
		},
		{
			X:      mat.NewDense(3, 4, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}),
			model:  []bool{false, true, false, true},
			expect: mat.NewDense(3, 2, []float64{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}),
		},
		{
			X:      mat.NewDense(1, 1, []float64{1.0}),
			model:  []bool{true},
			expect: mat.NewDense(1, 1, []float64{1.0}),
		},
	} {
		design := GetDesignMatrix(test.model, test.X)

		if !mat.EqualApprox(test.expect, design, 1e-12) {
			t.Errorf("DesignMatrix: Expect:\n %v \nGot:\n %v\n", mat.Formatted(test.X), mat.Formatted(design))
		}
	}
}
