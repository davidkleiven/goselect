package featselect

import (
	"math"
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestFit(t *testing.T) {
	for _, test := range []struct {
		X      *mat.Dense
		y      []float64
		expect []float64
	}{
		{
			X:      mat.NewDense(2, 2, []float64{1.0, -2.0, 1.0, 2.0}),
			y:      []float64{1.0, -1.0},
			expect: []float64{0.0, -0.5},
		},
		{
			X:      mat.NewDense(5, 2, []float64{1, 0, 1, 1, 1, 2, 1, 3, 1, 4}),
			y:      []float64{2, 4, 6, 8, 10},
			expect: []float64{2, 2},
		},
		{
			X:      mat.NewDense(4, 3, []float64{1, 0, 0, 1, 1, 1, 1, 2, 4, 1, 3, 9}),
			y:      []float64{-1, -2, -9, -22},
			expect: []float64{-1, 2, -3},
		},
		{
			X:      mat.NewDense(2, 3, []float64{1, 0, 0, 1, 1, 1}),
			y:      []float64{0, 6},
			expect: []float64{0, 3, 3},
		},
	} {
		result := Fit(test.X, test.y)

		if !floats.EqualApprox(result, test.expect, 1E-10) {
			t.Errorf("Fit failed. Got:\n%v\nWant:\n%v\n", result, test.expect)
		}
	}
}

func TestPredictOne(t *testing.T) {
	x := []float64{0.0, 2.0, 4.0}
	coeff := []float64{2.0, -1.0, 3.0}
	expect := 10.0
	pred := PredictOne(x, coeff)

	if math.Abs(pred-expect) > 1e-10 {
		t.Errorf("PredictOne: Expected %v. Got %v", expect, pred)
	}
}

func TestPredict(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1.0, -1.0, 2.0, 2.0})
	coeff := []float64{2.0, 2.0}
	expect := []float64{0.0, 8.0}
	pred := Predict(X, coeff)
	if !floats.EqualApprox(pred, expect, 1e-10) {
		t.Errorf("Predict: Expected: %v, Got: %v", expect, pred)
	}
}

func TestRSS(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1.0, -1.0, 2.0, 2.0})
	coeff := []float64{2.0, 2.0}
	data := []float64{1.0, 6.0}
	expect := 5.0
	dev := Rss(X, coeff, data)
	if math.Abs(expect-dev) > 1e-10 {
		t.Errorf("RSS: Expected: %v, Got %v", expect, dev)
	}
}

func TestGCV(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()

	coeff := Fit(X, y)
	pred := Predict(X, coeff)
	denum := DenumMatrixGcv(X)
	gcv := Gcv(denum, y, pred)

	nr, nc := X.Dims()
	model := make([]bool, nc)

	for i := range model {
		model[i] = true
	}

	cvBrute := 0.0
	for i := 0; i < nr; i++ {
		design := mat.NewDense(nr-1, nc, nil)
		yloo := make([]float64, nr-1)
		dr := 0
		for r := 0; r < nr; r++ {
			if r == i {
				continue
			}
			for c := 0; c < nc; c++ {
				design.Set(dr, c, X.At(r, c))
			}
			yloo[dr] = y[i]
			dr++
		}
		coeffCV := Fit(design, yloo)
		ypred := Predict(X, coeffCV)
		cvBrute += math.Pow(ypred[i]-y[i], 2)
	}
	cvBrute = math.Sqrt(cvBrute / float64(nc))

	if math.Abs(cvBrute-gcv) > 1e-10 {
		t.Errorf("GCV differ. Brute force %f, expected %f", cvBrute, gcv)
	}
}
