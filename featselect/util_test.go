package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestExtractDesignMatirx(t *testing.T) {
	for _, test := range []struct {
		X      *mat.Dense
		model  []bool
		expect *mat.Dense
	}{
		{
			X:      mat.NewDense(3, 2, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
			model:  []bool{false, true},
			expect: mat.NewDense(3, 1, []float64{2.0, 4.0, 6.0}),
		},
		{
			X:      mat.NewDense(3, 4, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}),
			model:  []bool{false, true, false, true},
			expect: mat.NewDense(3, 2, []float64{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}),
		},
		{
			X:      mat.NewDense(1, 1, []float64{1.0}),
			model:  []bool{true},
			expect: mat.NewDense(1, 1, []float64{1.0}),
		},
	} {
		design := GetDesignMatrix(test.model, test.X)

		if !mat.EqualApprox(test.expect, design, 1e-12) {
			t.Errorf("DesignMatrix: Expect:\n %v \nGot:\n %v\n", mat.Formatted(test.X), mat.Formatted(design))
		}
	}
}

func TestIterProduct(t *testing.T) {
	for i, test := range []struct {
		values []int
		repeat int
		expect [][]int
	}{
		{
			values: []int{0, 1, 2},
			repeat: 1,
			expect: [][]int{{0}, {1}, {2}},
		},
		{
			values: []int{0, 1, 2},
			repeat: 2,
			expect: [][]int{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}},
		},
		{
			values: []int{0, 1},
			repeat: 3,
			expect: [][]int{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}},
		},
	} {
		prod := IterProduct(test.values, test.repeat)
		if !nestedArrayEqual(prod, test.expect) {
			t.Errorf("Test #%v faild. Expected:\n%v\nGot:\n%v\n", i, test.expect, prod)
		}
	}
}

func TestParseCommandLine(t *testing.T) {
	cmd := []string{"--csv=myfile.csv", "--target=10", "--out=outfile.json", "--cutoff=4.0"}
	opt := ParseCommandLineArgs(cmd)

	if opt.Csvfile != "myfile.csv" {
		t.Errorf("Expectd: myfile.csv Got: %v", opt.Csvfile)
	}

	if opt.TargetCol != 10 {
		t.Errorf("Expected: 10 Got: %v", opt.TargetCol)
	}

	if opt.Outfile != "outfile.json" {
		t.Errorf("Expected: outfile.json Got: %v", opt.Outfile)
	}

	if math.Abs(opt.Cutoff-4.0) > 1e-12 {
		t.Errorf("Expected: 4.0. Got: %v", opt.Cutoff)
	}
}

func TestAll(t *testing.T) {
	for i, test := range []struct {
		array  []int
		target int
		expect bool
	}{
		{
			array:  []int{2, 2, 2},
			target: 3,
			expect: false,
		},
		{
			array:  []int{2, 2, 3},
			target: 2,
			expect: false,
		},
		{
			array:  []int{3, 3, 3, 3},
			target: 3,
			expect: true,
		},
	} {
		res := All(test.array, test.target)

		if res != test.expect {
			t.Errorf("Test #%v failed. Expected: %v. Got %v", i, test.expect, res)
		}
	}
}

func TestRearrange(t *testing.T) {
	for i, test := range []struct {
		X      *mat.Dense
		order  []int
		expect *mat.Dense
	}{
		{
			X:      mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
			order:  []int{1, 0},
			expect: mat.NewDense(2, 2, []float64{2.0, 1.0, 4.0, 3.0}),
		},
		{
			X:      mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
			order:  []int{2, 0, 1},
			expect: mat.NewDense(2, 3, []float64{3.0, 1.0, 2.0, 6.0, 4.0, 5.0}),
		},
		{
			X:      mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
			order:  []int{1},
			expect: mat.NewDense(2, 3, []float64{2.0, 1.0, 3.0, 5.0, 4.0, 6.0}),
		},
		{
			X:      mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
			order:  []int{0, 2},
			expect: mat.NewDense(2, 3, []float64{1.0, 3.0, 2.0, 4.0, 6.0, 5.0}),
		},
	} {
		rearranged := RearrangeDense(test.X, test.order)
		if !mat.EqualApprox(rearranged, test.expect, 1e-10) {
			t.Errorf("Test #%v:\nExpected\n%v\nGot\n%v\n", i, mat.Formatted(test.expect), mat.Formatted(rearranged))
		}
	}
}

func TestSelect2Model(t *testing.T) {
	for i, test := range []struct {
		selected []int
		expect   []bool
		ncols    int
	}{
		{
			selected: []int{2, 5, 8},
			ncols:    10,
			expect:   []bool{false, false, true, false, false, true, false, false, true, false},
		},
		{
			selected: []int{1, 0, 5},
			ncols:    6,
			expect:   []bool{true, true, false, false, false, true},
		},
	} {
		model := Selected2Model(test.selected, test.ncols)

		if !boolArrayEqual(model, test.expect) {
			t.Errorf("Test #%v.\nExpected\n%v\nGot\n%v", i, test.expect, model)
		}
	}
}

func nestedArrayEqual(a [][]int, b [][]int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[i]); j++ {
			if len(a[i]) != len(b[i]) {
				return false
			}
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

func boolArrayEqual(a []bool, b []bool) bool {
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
