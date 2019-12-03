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

	if !floats.EqualApprox(expectY, dset.Y, 1e-10) {
		t.Errorf("Expected:\n%v\nGot:\n%v\n", expectY, dset.Y)
	}

	if !strArrayEqual(expectHeader, dset.Names) {
		t.Errorf("Expected:\n%v\nGot:\n%v\n", expectHeader, dset.Names)
	}
}

func TestWriteDataset(t *testing.T) {
	var writer strings.Builder

	var dset Dataset
	dset.X = mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0})
	dset.Y = []float64{-1.0, -3.0}
	dset.TargetCol = 1
	dset.Names = []string{"feat1", "target", "feat2"}

	dset.SaveHandle(&writer)
	str := writer.String()
	expect := "feat1,target,feat2\n1.000000,-1.000000,2.000000\n3.000000,-3.000000,4.000000\n"

	if str != expect {
		t.Errorf("\nExpected\n%v\nGot\n%v\n", expect, str)
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
