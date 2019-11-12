package testfeatselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// OmpTest is a structure for holding nessecary data for testing
// orthogonal matchin pursuit
type OmpTest struct {
	Name        string
	Target      []float64
	X           *mat.Dense
	ExpectOrder []int
	ExpectCoeff []float64
}

// LinearCubic creates a test case where the target function is a straight line
// and the desigin matrix contains features up to cubic order
func LinearCubic(x []float64) *OmpTest {
	var test OmpTest
	test.Name = "LinearCubic"
	test.Target = make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		test.Target[i] = 1.0 + 10.0*x[i]
	}

	test.X = mat.NewDense(len(x), 4, nil)
	for col := 0; col < 4; col++ {
		for i := 0; i < len(x); i++ {
			test.X.Set(i, col, math.Pow(x[i], float64(col)))
		}
	}

	test.ExpectOrder = []int{1, 0}
	test.ExpectCoeff = []float64{1.0, 10.0}
	return &test
}

// Exponentials return a test case with a sum of exponential functions with different decay rates
func Exponentials(x []float64) *OmpTest {
	var test OmpTest
	test.Name = "Exponentials"
	test.Target = make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		test.Target[i] = -3.0*math.Exp(-x[i]) + 1.0*math.Exp(-5.0*x[i])
	}

	test.X = mat.NewDense(len(x), 6, nil)
	for col := 0; col < 6; col++ {
		for i := 0; i < len(x); i++ {
			test.X.Set(i, col, math.Exp(-float64(col)*x[i]))
		}
	}

	test.ExpectOrder = []int{1, 5}
	test.ExpectCoeff = []float64{0.0, -3.0, 0.0, 0.0, 0.0, 1.0}
	return &test
}
