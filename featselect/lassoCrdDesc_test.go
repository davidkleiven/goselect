package featselect

import (
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
	"gonum.org/v1/gonum/floats"
)

func TestLassoCrdDesc(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()
	data := NewNormalizedData(X, y)
	var cov Empirical
	res := LassoCrdDesc(data, 0.00001, &cov, nil, 100000)
	res[0] = data.LinearTransformationBias([]int{1, 2, 3, 4}, res[1:])
	for i := 1; i < 5; i++ {
		res[i] = data.LinearNormalizationTransformation(i, res[i])
	}

	expect := []float64{-3., 3.0, 0.2, 0.0, 0.0}

	if !floats.EqualApprox(res, expect, 1e-4) {
		t.Errorf("LassoCrdDesc: expected\n%v\nGot\n%v\n", expect, res)
	}
}

func TestLassoCrdDescPath(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()
	data := NewNormalizedData(X, y)

	lambs := Logspace(1e-8, 1.0, 20)
	var cov Empirical
	res := LassoCrdDescPath(data, &cov, lambs, 100000)
	Path2Unnormalized(data, res)

	last := res[len(res)-1]
	expectSel := []int{0, 1, 2}
	for i := range expectSel {
		if expectSel[i] != last.Selection[i] {
			t.Errorf("unexpected selection. Expected\n%v\nGot\n%v\n", expectSel, last.Selection)
			break
		}
	}

	expectCoeff := []float64{-3., 3., 0.2}
	if !floats.EqualApprox(expectCoeff, last.Coeff, 1e-4) {
		t.Errorf("unexpected coefficients. Expected\n%v\nGot\n%v\n", expectCoeff, last.Coeff)
	}
}
