package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAIC(t *testing.T) {
	N := 10
	r := 1e-4
	numFeat := 5
	expect := 10.0 + 10.0*math.Log(r)
	got := aic(numFeat, N, r)

	if math.Abs(got-expect) > 1e-12 {
		t.Errorf("AIC: Expected: %v, Got: %v", expect, got)
	}
}

func TestBoundsAIC(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1.0, 0.0, 1.0, 1.0})
	y := []float64{2.0, 4.0}
	model := []bool{true, false}
	lower, upper := boundsAIC(model, 1, X, y)

	tol := 1e-12
	expectLower := aic(1, 2, tol)
	expectUpper := aic(2, 2, 2.0)

	if math.Abs(lower-expectLower) > tol {
		t.Errorf("AIC bouds: Lower: Expected %v got: %v", expectLower, lower)
	}

	if math.Abs(upper-expectUpper) > tol {
		t.Errorf("AIC: bounds: Upper: Expect: %v Got %v", expectUpper, upper)
	}
}
