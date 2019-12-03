package featselect

import (
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestNormalizedDataset(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{-4.0, -2.0, 0.0, 2.0, 4.0, 0.0})
	y := []float64{-1.0, 0.0, 1.0}

	data := NewNormalizedData(X, y)

	normX := mat.NewDense(3, 2, []float64{-1.0, -1.0, 0.0, 1.0, 1.0, 0.0})
	normY := []float64{-1.0, 0.0, 1.0}

	if !mat.EqualApprox(data.X, normX, 1e-10) {
		t.Errorf("unexpected matrix expected \n%v\ngot\n%v\n", mat.Formatted(data.X), mat.Formatted(normX))
	}

	if !floats.EqualApprox(data.y, y, 1e-10) {
		t.Errorf("unexpected values expected\n%v\ngot\n%v\n", data.y, normY)
	}
}
