package featselect

import (
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
	"gonum.org/v1/gonum/mat"
)

func TestIdentity(t *testing.T) {
	id := Identity{}
	X := mat.NewDense(2, 2, nil)
	res := id.Get(X)
	expect := mat.NewDense(2, 2, []float64{1.0, 0.0, 0.0, 1.0})

	if !mat.EqualApprox(expect, res, 1e-10) {
		t.Errorf("Expected\n%v\n matrix got\n%v\n", mat.Formatted(expect), mat.Formatted(res))
	}
}

func TestSparseThreshold(t *testing.T) {
	X, _ := testfeatselect.GetExampleXY()
	sp := NewSparseThreshold(X)

	res := sp.Get(X)
	nr, nc := res.Dims()

	// TODO: Just check that we get a square matrix. Better test?
	if nr != nc {
		t.Errorf("Expected square matrix")
	}
}
