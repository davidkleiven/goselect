package testfeatselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// DemoSet is a function type for functions that returns a design matrix and a slice
// with target values
type DemoSet func() (*mat.Dense, []float64)

// GetExampleXY returns an example set used for testing
func GetExampleXY() (*mat.Dense, []float64) {
	X := mat.NewDense(100, 5, nil)
	y := make([]float64, 100)

	for i := 0; i < 100; i++ {
		x := 0.1 * float64(i)
		for j := 0; j < 5; j++ {
			X.Set(i, j, math.Pow(x, float64(j)))
		}
		y[i] = -3.0 + 0.2*x*x + 3.0*x
	}
	return X, y
}

// GetExampleAllModelsWrong returns a dataset where none of the
// possible models fits the data
func GetExampleAllModelsWrong() (*mat.Dense, []float64) {
	N := 30
	X := mat.NewDense(N, 7, nil)
	y := make([]float64, N)

	for i := 0; i < N; i++ {
		x := 0.1 * float64(i)
		for j := 0; j < 7; j++ {
			X.Set(i, j, math.Pow(x, float64(j)))
		}
		y[i] = -3.0 + 0.2*math.Sin(x)
	}
	return X, y
}
