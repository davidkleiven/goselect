package featselect

import (
	"strings"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestParseCSV(t *testing.T) {
	strData := "Feat1,Feat2,Feat3\n1.0,2.0,3.0\n4.0,5.0,6.0\n"
	for _, test := range []struct {
		expectX      *mat.Dense
		expectY      []float64
		expectHeader []string
		targCol      int
	}{
		{
			expectX:      mat.NewDense(2, 2, []float64{1.0, 3.0, 4.0, 6.0}),
			expectY:      []float64{2.0, 5.0},
			expectHeader: []string{"Feat1", "Feat2", "Feat3"},
			targCol:      1,
		},
		{
			expectX:      mat.NewDense(2, 2, []float64{1.0, 2.0, 4.0, 5.0}),
			expectY:      []float64{3.0, 6.0},
			expectHeader: []string{"Feat1", "Feat2", "Feat3"},
			targCol:      -1,
		},
	} {
		reader := strings.NewReader(strData)
		dset := ParseCSV(reader, test.targCol)

		if !mat.EqualApprox(dset.X, test.expectX, 1e-10) {
			t.Errorf("Expected:\n%v\nGot:\n%v\n", mat.Formatted(test.expectX), mat.Formatted(dset.X))
		}

		if !floats.EqualApprox(test.expectY, dset.Y, 1e-10) {
			t.Errorf("Expected:\n%v\nGot:\n%v\n", test.expectY, dset.Y)
		}

		if !strArrayEqual(test.expectHeader, dset.Names) {
			t.Errorf("Expected:\n%v\nGot:\n%v\n", test.expectHeader, dset.Names)
		}
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

func TestGetFeatName(t *testing.T) {
	for i, test := range []struct {
		dset   Dataset
		featNo int
		expect string
	}{
		{
			dset:   Dataset{Names: []string{"feat1", "feat2", "feat3"}, TargetCol: 2},
			featNo: 1,
			expect: "feat2",
		},
		{
			dset:   Dataset{Names: []string{"feat1", "feat2", "feat3"}, TargetCol: 0},
			featNo: 1,
			expect: "feat3",
		},
	} {
		name := test.dset.GetFeatName(test.featNo)

		if name != test.expect {
			t.Errorf("Test #%d failed. Expect %s got %s", i, test.expect, name)
		}
	}
}

func TestGetSubset(t *testing.T) {
	for i, test := range []struct {
		dset     Dataset
		features []int
		expect   Dataset
	}{
		{
			dset:     Dataset{Names: []string{"n1", "n2", "n3"}, TargetCol: 0, Y: []float64{1.0, 2.0}, X: mat.NewDense(2, 2, []float64{1., 2., 3., 4.})},
			features: []int{1},
			expect:   Dataset{Names: []string{"n3", "n1"}, TargetCol: 1, Y: []float64{1.0, 2.0}, X: mat.NewDense(2, 1, []float64{2., 4.})},
		},
	} {
		res := test.dset.GetSubset(test.features)

		if !test.expect.IsEqual(res) {
			t.Errorf("Test #%d: Subset does not match expected", i)
		}
	}
}

func TestFeatNoByname(t *testing.T) {
	for i, test := range []struct {
		dset   Dataset
		name   string
		expect int
	}{
		{
			dset:   Dataset{Names: []string{"n1", "n2", "n3", "n4"}, TargetCol: 1},
			name:   "n1",
			expect: 0,
		},
		{
			dset:   Dataset{Names: []string{"n1", "n2", "n3", "n4"}, TargetCol: 1},
			name:   "n3",
			expect: 1,
		},
		{
			dset:   Dataset{Names: []string{"n1", "n2", "n3", "n4"}, TargetCol: 1},
			name:   "n4",
			expect: 2,
		},
	} {
		got := test.dset.FeatNoByName(test.name)

		if got != test.expect {
			t.Errorf("Test #%d: unexpected feat no. Expected %d got %d", i, test.expect, got)
		}
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
