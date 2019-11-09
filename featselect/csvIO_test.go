package featselect

import (
	"strings"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestParseCSV(t *testing.T) {
	reader := strings.NewReader("Feat1,Feat2,Feat3\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
	dset := ParseCSV(reader, 1)
	expectX := mat.NewDense(2, 2, []float64{1.0, 3.0, 4.0, 6.0})
	expectY := []float64{2.0, 5.0}
	expectHeader := []string{"Feat1", "Feat2", "Feat3"}

	if !mat.EqualApprox(dset.X, expectX, 1e-10) {
		t.Errorf("Expected:\n%v\nGot:\n%v\n", mat.Formatted(expectX), mat.Formatted(dset.X))
	}

	if !floats.EqualApprox(expectY, dset.y, 1e-10) {
		t.Errorf("Expected:\n%v\nGot:\n%v\n", expectY, dset.y)
	}

	if !strArrayEqual(expectHeader, dset.names) {
		t.Errorf("Expected:\n%v\nGot:\n%v\n", expectHeader, dset.names)
	}
}

func strArrayEqual(a []string, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
