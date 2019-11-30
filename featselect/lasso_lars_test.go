package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLassoLars(t *testing.T) {
	x := make([]float64, 100)
	for i := 0; i < len(x); i++ {
		x[i] = 0.1 * float64(i)
	}

	X := mat.NewDense(len(x), 5, nil)
	y := make([]float64, len(x))

	for i := 0; i < len(x); i++ {
		for j := 0; j < 5; j++ {
			X.Set(i, j, math.Pow(x[i], float64(j)))
		}
		y[i] = -3.0 + 0.2*x[i]*x[i]
	}

	res := LassoLars(X, y, 1e-10)
	last := res[len(res)-1]

	for i := range last.Coeff {
		if last.Selection[i] == 0 {
			if math.Abs(last.Coeff[i]+3.0) > 1e-10 {
				t.Errorf("bias term should be -3.0, got %f", last.Coeff[i])
			}
		} else if last.Selection[i] == 2 {
			if math.Abs(last.Coeff[i]-0.2) > 1e-10 {
				t.Errorf("quad term should be 0.2, got %f", last.Coeff[i])
			}
		} else {
			if math.Abs(last.Coeff[i]) > 1e-10 {
				t.Errorf("expected term to be zero. got %f", last.Coeff[i])
			}
		}
	}
}
