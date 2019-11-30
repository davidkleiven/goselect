package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestIndexedColView(t *testing.T) {
	a := mat.NewDense(3, 3, []float64{1., 2., 3., 4., 5., 6., 7., 8., 9.})
	v := NewIndexedColView(a, []int{1, 2})

	r, c := v.Dims()
	if r != 3 || c != 2 {
		t.Errorf("unexpected dims. Expected (3, 2) got (%d, %d)", r, c)
	}

	if math.Abs(v.At(0, 1)-3.0) > 1e-10 {
		t.Errorf("unexpected value. Expected 3.0 got %f", v.At(0, 1))
	}

	vT := v.T()
	if math.Abs(vT.At(0, 1)-5.0) > 1e-10 {
		t.Errorf("unexpected value. Expected 5.0 got %f", vT.At(0, 1))
	}
}

func TestIndexedColVecView(t *testing.T) {
	a := mat.NewVecDense(4, []float64{1., 2., 3., 4.})
	v := NewIndexedColVecView(a, []int{1, 2})

	nr, nc := v.Dims()

	if nr != 2 || nc != 1 {
		t.Errorf("unexpected dims. Expected (2, 1). Got (%d, %d)", nr, nc)
	}

	length := v.Len()
	if length != nr {
		t.Errorf("unexpected length. Expected %d got %d", nr, length)
	}

	val := v.At(0, 0)
	if math.Abs(val-2.) > 1e-10 {
		t.Errorf("unexpected value. Expected 2.0 got %f", val)
	}

	val = v.AtVec(0)
	if math.Abs(val-2.) > 1e-10 {
		t.Errorf("unexpected value. Expected 2.0 got %f", val)
	}

	vT := v.T()
	nr, nc = vT.Dims()
	if nr != 1 || nc != 2 {
		t.Errorf("unexpected dims. Expected (2, 1). Got (%d, %d)", nr, nc)
	}
}
