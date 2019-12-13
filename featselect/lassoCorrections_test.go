package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestCLasso(t *testing.T) {
	classo := NewCLasso(5, 1.0)
	beta := []float64{1., 2., 3., 4., 5.}
	classo.Update(beta)

	if !floats.EqualApprox(beta, classo.beta, 1e-10) {
		t.Errorf("Coefficient vector not update correctly")
	}

	for i := 0; i < 5; i++ {
		deriv := classo.Deriv(beta, i)
		if math.Abs(beta[i]-deriv) > 1e-10 {
			t.Errorf("unexpected derivative. Expected %f got %f\n", beta[i], deriv)
		}
	}
}
