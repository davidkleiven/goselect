package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestSA(t *testing.T) {
	x := floats.Span(make([]float64, 30), 0.0, 1.0)
	y := make([]float64, len(x))
	X := mat.NewDense(len(x), 20, nil)
	for col := 0; col < 20; col++ {
		for row := 0; row < len(x); row++ {
			X.Set(row, col, math.Pow(x[row], float64(col)))
		}
	}

	for i := 0; i < len(x); i++ {
		y[i] = 1.0 + 5.0*x[i]*x[i]
	}

	res := SelectModelSA(X, y, 2, Aicc)
	expectSelected := []int{0, 2}
	expectCoeff := []float64{1.0, 5.0}

	if !sliceEqual(expectSelected, res.Selected) {
		t.Errorf("SA: Expected %v Got %v", expectSelected, res.Selected)
	}

	if !floats.EqualApprox(res.Coeff, expectCoeff, 1e-10) {
		t.Errorf("SA: Expected %v Got %v", expectCoeff, res.Coeff)
	}
}

func sliceEqual(s1 []int, s2 []int) bool {
	if len(s1) != len(s2) {
		return false
	}

	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}
