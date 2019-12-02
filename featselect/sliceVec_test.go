package featselect

import (
	"math"
	"testing"
)

func TestSliceVec(t *testing.T) {
	data := []float64{1., 2., 3., 4., 5.}
	v := NewSliceVec(data)

	if v.Len() != 5 {
		t.Errorf("unexpected length. Want %d Got %d", 5, v.Len())
	}

	val := v.AtVec(3)

	if math.Abs(val-4.0) > 1e-10 {
		t.Errorf("unexpected value. Want 5.0, got %f", val)
	}

	val = v.At(3, 0)

	if math.Abs(val-4.0) > 1e-10 {
		t.Errorf("unexpected value. Want 5.0, got %f", val)
	}

	nr, nc := v.Dims()

	if nr != 5 || nc != 1 {
		t.Errorf("unexpected dimensions. Want (5, 1) got (%d, %d)", nr, nc)
	}

	vT := v.T()

	nr, nc = vT.Dims()

	if nr != 1 || nc != 5 {
		t.Errorf("unexpected dimensions. Want (5, 1) got (%d, %d)", nr, nc)
	}

}
