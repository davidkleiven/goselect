package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNewIndexedView(t *testing.T) {
	for testnum, test := range []struct {
		a             mat.Matrix
		rows          []int
		cols          []int
		accessRows    []int
		accessCols    []int
		expectValues  []float64
		expectNumRows int
		expectNumCols int
	}{
		{
			a:             mat.NewDense(3, 3, []float64{1., 2., 3., 4., 5., 6., 7., 8., 9.}),
			rows:          []int{1, 2},
			cols:          []int{1},
			accessRows:    []int{0, 1},
			accessCols:    []int{0, 0},
			expectValues:  []float64{5., 8.},
			expectNumRows: 2,
			expectNumCols: 1,
		},
		{
			a:             mat.NewDense(3, 3, []float64{1., 2., 3., 4., 5., 6., 7., 8., 9.}),
			rows:          nil,
			cols:          []int{1},
			accessRows:    []int{0, 1, 2},
			accessCols:    []int{0, 0, 0},
			expectValues:  []float64{2., 5., 8.},
			expectNumRows: 3,
			expectNumCols: 1,
		},
		{
			a:             mat.NewDense(3, 3, []float64{1., 2., 3., 4., 5., 6., 7., 8., 9.}),
			rows:          []int{1},
			cols:          nil,
			accessRows:    []int{0, 0, 0},
			accessCols:    []int{0, 1, 2},
			expectValues:  []float64{4., 5., 6.},
			expectNumRows: 1,
			expectNumCols: 3,
		},
		{
			a:             mat.NewDense(3, 3, []float64{1., 2., 3., 4., 5., 6., 7., 8., 9.}),
			rows:          nil,
			cols:          nil,
			accessRows:    []int{0, 0, 0, 1, 1, 1, 2, 2, 2},
			accessCols:    []int{0, 1, 2, 0, 1, 2, 0, 1, 2},
			expectValues:  []float64{1., 2., 3., 4., 5., 6., 7., 8., 9.},
			expectNumRows: 3,
			expectNumCols: 3,
		},
	} {
		v := NewIndexedView(test.a, test.rows, test.cols)

		nr, nc := v.Dims()

		if nr != test.expectNumRows || nc != test.expectNumCols {
			t.Errorf("Test #%v failed. Expected dims: (%v, %v). Got dims: (%v, %v)", testnum, test.expectNumRows, test.expectNumCols, nr, nc)
		}

		for i := 0; i < len(test.accessRows); i++ {
			res := v.At(test.accessRows[i], test.accessCols[i])

			if math.Abs(res-test.expectValues[i]) > 1e-10 {
				t.Errorf("Test #%v failed. Expected: %v. Got: %v", testnum, test.expectValues[i], res)
			}
		}
	}
}
