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
	got := Aic(numFeat, N, r)

	if math.Abs(got-expect) > 1e-12 {
		t.Errorf("AIC: Expected: %v, Got: %v", expect, got)
	}
}

func TestBoundsAIC(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1.0, 0.0, 1.0, 1.0})
	y := []float64{2.0, 4.0}
	model := []bool{true, false}
	_, upper := BoundsAIC(model, 1, X, y)

	expectUpper := Aic(2, 2, 2.0)

	if math.Abs(upper-expectUpper) > 1e-12 {
		t.Errorf("AIC: bounds: Upper: Expect: %v Got %v", expectUpper, upper)
	}
}

func TestBoundsAICC(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0})
	y := []float64{2.0, 4.0, 6.0, 8.0}
	model := []bool{true, false}
	_, upper := BoundsAICC(model, 1, X, y)
	expectUpper := Aicc(2, 4, 9.0+1.0+1.0+9.0)

	if math.Abs(upper-expectUpper) > 1e-12 {
		t.Errorf("AICC: bounds: Upper: Expect: %v Got %v", expectUpper, upper)
	}
}
