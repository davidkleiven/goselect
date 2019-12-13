package featselect

import (
	"math"
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// FixedLassoCoeff can be used as a CorrectableLasso function. It always returns the same coefficients
func FixedLassoCoeff(dset *NormalizedData, lamb float64, cov CovMat, x0 []float64, maxIter int, tol float64, corr LassoCorrection) []float64 {
	return []float64{1.0, 2.0, 3.0, 4.0, 0, 0, 1.0, 0}
}

// FixedLassoCoeffSelection return the selection corresponding to the coefficients in
// FixedLassoCoeff
func FixedLassoCoeffSelection() []int {
	return []int{0, 1, 2, 3, 6}
}

func TestLassoCrdDesc(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()
	data := NewNormalizedData(X, y)
	var cov Empirical
	var correction PureLasso
	res := LassoCrdDesc(data, 0.00001, &cov, nil, 100000, 1e-10, &correction)
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
	var correction PureLasso
	res := LassoCrdDescPath(data, &cov, lambs, 100000, 1e-10, &correction)
	Path2Unnormalized(data, res)

	last := res[len(res)-1]
	expectSel := []int{0, 1, 2}

	if len(expectSel) != len(last.Selection) {
		t.Errorf("unexpected selection. Expected\n%v\nGot\n%v\n", expectSel, last.Selection)
	} else {
		for i := range expectSel {
			if expectSel[i] != last.Selection[i] {
				t.Errorf("unexpected selection. Expected\n%v\nGot\n%v\n", expectSel, last.Selection)
				break
			}
		}
	}

	expectCoeff := []float64{-3., 3., 0.2}
	if !floats.EqualApprox(expectCoeff, last.Coeff, 1e-4) {
		t.Errorf("unexpected coefficients. Expected\n%v\nGot\n%v\n", expectCoeff, last.Coeff)
	}
}

func TestLassoInterfaces(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()
	ndata := NewNormalizedData(X, y)

	type initfunc = func() CohensKappaTarget
	type hypercheck = func(hyper map[string]float64) bool

	for i, test := range []struct {
		fn     initfunc
		rep    string
		hcheck hypercheck
	}{
		{
			fn: func() CohensKappaTarget {
				lasso := NewPureLassoCohen()
				lasso.lasso = FixedLassoCoeff
				lasso.Dset = ndata
				lasso.Lamb = 1e-5
				return lasso
			},
			rep: "Lambda: 1.000e-05",
			hcheck: func(h map[string]float64) bool {
				return math.Abs(h["lamb"]-1e-5) < 1e-10
			},
		},
		{
			fn: func() CohensKappaTarget {
				cohen := NewCLassoCohen()
				cohen.Eta = 0.5
				cohen.PLasso.lasso = FixedLassoCoeff
				cohen.PLasso.Dset = ndata
				cohen.PLasso.Lamb = 1e-5
				return cohen
			},
			rep: "Lambda: 1.000e-05 Eta: 5.000e-01",
			hcheck: func(h map[string]float64) bool {
				return math.Abs(h["lamb"]-1e-5) < 1e-10 && math.Abs(h["eta"]-0.5) < 1e-10
			},
		},
	} {
		lassocohen := test.fn()

		X2 := lassocohen.GetX()

		if !mat.EqualApprox(X2, ndata.X, 1e-10) {
			t.Errorf("Test #%d: Wrong X matrix returned by GetX", i)
		}

		indices := []int{1, 2}
		selection := lassocohen.GetSelection(indices)

		if !EqualInt(selection, FixedLassoCoeffSelection()) {
			t.Errorf("Test #%d: Wrong selection", i)
		}

		hyper := lassocohen.HyperParameters()

		if !test.hcheck(hyper) {
			t.Errorf("Test #%d: Wrong hyper parameters in got %v", i, hyper)
		}

		strrep := lassocohen.StringRep()

		if strrep != test.rep {
			t.Errorf("Test #%d: Wrong string rep. Expected %s got %s", i, test.rep, strrep)
		}
	}
}
