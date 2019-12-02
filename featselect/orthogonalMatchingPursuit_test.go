package featselect

import (
	"math"
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
	"gonum.org/v1/gonum/floats"
)

func TestArgmax(t *testing.T) {
	for i, test := range []struct {
		array  []float64
		expect int
	}{
		{
			array:  []float64{-1.0, 2.0, 1.9},
			expect: 1,
		},
		{
			array:  []float64{0.0},
			expect: 0,
		},
		{
			array:  []float64{1.0, 5.0, 7.0},
			expect: 2,
		},
	} {
		imax := Argmax(test.array)
		if imax != test.expect {
			t.Errorf("Test #%v: Expected %v, Got %v", i, test.expect, imax)
		}
	}
}

func TestAbs(t *testing.T) {
	for i, test := range []struct {
		array  []float64
		expect []float64
	}{
		{
			array:  []float64{-1.0, 2.0, 3.0},
			expect: []float64{1.0, 2.0, 3.0},
		},
		{
			array:  []float64{2.0, 5.0, 10.0},
			expect: []float64{2.0, 5.0, 10.0},
		},
	} {
		Abs(test.array)
		if !floats.EqualApprox(test.array, test.expect, 1e-10) {
			t.Errorf("Test #%v. Expected: %v Got %v", i, test.expect, test.array)
		}
	}
}

func TestSelectedFeatures(t *testing.T) {
	for i, test := range []struct {
		model  []bool
		expect []int
	}{
		{
			model:  []bool{false, true, true},
			expect: []int{1, 2},
		},
		{
			model:  []bool{false, false, false, false},
			expect: []int{},
		},
		{
			model:  []bool{true, true, true},
			expect: []int{0, 1, 2},
		},
	} {
		feat := SelectedFeatures(test.model)
		for j := 0; j < len(test.expect); j++ {
			if test.expect[j] != feat[j] {
				t.Errorf("Test #%v: Expected %v Got %v", i, test.expect, feat)
				break
			}
		}
	}
}

func TestOmp(t *testing.T) {
	for _, test := range []struct {
		tcase *testfeatselect.OmpTest
		tol   float64
	}{
		{
			tcase: testfeatselect.LinearCubic(floats.Span(make([]float64, 20), 0.0, 3.0)),
			tol:   1e-6,
		},
		{
			tcase: testfeatselect.Exponentials(floats.Span(make([]float64, 20), 0.0, 2.0)),
			tol:   1e-6,
		},
	} {
		res := Omp(test.tcase.X, test.tcase.Target, test.tol)
		for i := 0; i < len(test.tcase.ExpectOrder); i++ {
			if test.tcase.ExpectOrder[i] != res.Order[i] {
				t.Errorf("\nTest %v\nExpected\n%v\nGot\n%v\n", test.tcase.Name, test.tcase.ExpectOrder, res.Order)
				return
			}
		}

		for i := 0; i < len(test.tcase.ExpectCoeff); i++ {
			if math.Abs(test.tcase.ExpectCoeff[i]-res.Coeff[i]) > 1e-12 {
				t.Errorf("\nTest %v\nExpected\n%v\nGot\n%v\n", test.tcase.Name, test.tcase.ExpectCoeff, res.Coeff)
				return
			}
		}

	}
}
