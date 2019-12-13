package featselect

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"
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

func TestMean(t *testing.T) {
	for i, test := range []struct {
		array  []float64
		expect float64
	}{
		{
			array:  []float64{1.0, 2.0, 3.0},
			expect: 2.0,
		},
		{
			array:  []float64{-2.0, -1.0, 1.0, 2.0},
			expect: 0.0,
		},
	} {
		mu := Mean(test.array)

		if math.Abs(mu-test.expect) > 1e-10 {
			t.Errorf("Test #%v failed. Expected %v, Got %v", i, test.expect, mu)
		}
	}
}

func TestStd(t *testing.T) {
	for i, test := range []struct {
		array  []float64
		expect float64
	}{
		{
			array:  []float64{1.0},
			expect: 0.0,
		},
		{
			array:  []float64{1.0, 3.0},
			expect: math.Sqrt(2.0),
		},
	} {
		std := Std(test.array)

		if math.Abs(std-test.expect) > 1e-10 {
			t.Errorf("Test #%v failed. Expected %v, Got %v", i, test.expect, std)
		}
	}
}

func TestNormalizeArray(t *testing.T) {
	for i, test := range []struct {
		array []float64
	}{
		{
			array: []float64{1.0, 2.0, 3.0, 4.0},
		},
		{
			array: []float64{-2.0, 4.0, -2.0, 1.0, 6.0},
		},
		{
			array: []float64{2.0, 5.0, 6.0, -10.0, 2.0},
		},
	} {
		NormalizeArray(test.array)
		mu := Mean(test.array)
		std := Std(test.array)

		if math.Abs(mu) > 1e-10 {
			t.Errorf("Test #%v Expected 0.0, Got %v", i, mu)
		}

		if math.Abs(std-1.0) > 1e-10 {
			t.Errorf("Test #%v Expected 1.0 Got %v", i, std)
		}
	}
}

func TestNormalizeRows(t *testing.T) {
	for i, test := range []struct {
		array  *mat.Dense
		expect *mat.Dense
	}{
		{
			array:  mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
			expect: mat.NewDense(2, 2, []float64{-1.0 / math.Sqrt(2.0), 1.0 / math.Sqrt(2.0), -1.0 / math.Sqrt(2.0), 1.0 / math.Sqrt(2.0)}),
		},
	} {
		NormalizeRows(test.array)

		if !mat.EqualApprox(test.array, test.expect, 1e-10) {
			t.Errorf("Test #%v failed.\nExpected\n%v\nGot\n%v\n", i, test.expect, test.array)
		}
	}
}

func TestNormalizeCols(t *testing.T) {
	for i, test := range []struct {
		array  *mat.Dense
		expect *mat.Dense
	}{
		{
			array:  mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
			expect: mat.NewDense(2, 2, []float64{-1.0 / math.Sqrt(2.0), -1.0 / math.Sqrt(2.0), 1.0 / math.Sqrt(2.0), 1.0 / math.Sqrt(2.0)}),
		},
	} {
		NormalizeCols(test.array)

		if !mat.EqualApprox(test.array, test.expect, 1e-10) {
			t.Errorf("Test #%v failed.\nExpected\n%v\nGot\n%v\n", i, test.expect, test.array)
		}
	}
}

func TestWeightedAverage(t *testing.T) {
	coeffs := []SparseCoeff{
		{Coeff: []float64{1.0, 2.0, 3.0}, Selection: []int{0, 2, 5}},
		{Coeff: []float64{-1.0, 2.0, 5.0}, Selection: []int{2, 1, 4}},
	}

	weights := []float64{0.5, 1.0}

	avg := WeightedAveragedCoeff(6, weights, coeffs)
	expect := []float64{0.5, 2.0, 0.0, 0.0, 5.0, 1.5}

	if !floats.EqualApprox(avg, expect, 1e-10) {
		t.Errorf("weightedaverage: expected \n%v\ngot\n%v\n", expect, avg)
	}
}

func TestWeightsFromAIC(t *testing.T) {
	aic := []float64{1.0, 2.0, 3.0}
	w := WeightsFromAIC(aic)
	sum := 1.0 + math.Exp(-1.0) + math.Exp(-2.0)
	expect := []float64{1.0 / sum, math.Exp(-1.0) / sum, math.Exp(-2.0) / sum}

	if !floats.EqualApprox(w, expect, 1e-10) {
		t.Errorf("weightAIC: Expect\n%v\ngot\n%v\n", expect, w)
	}
}

func TestArgsort(t *testing.T) {
	for i, test := range []struct {
		a      []float64
		expect []int
	}{
		{
			a:      []float64{-1.0, 1.0, 2.0},
			expect: []int{0, 1, 2},
		},
		{
			a:      []float64{2.0, -1.0, 3.0, -0.5},
			expect: []int{1, 3, 0, 2},
		},
		{
			a:      []float64{1.0},
			expect: []int{0},
		},
	} {
		res := Argsort(test.a)

		for j := range res {
			if res[j] != test.expect[j] {
				t.Errorf("Test #%d: unexpected argsort. Expected %v got %v", i, test.expect, res)
			}
		}
	}
}

func TestPseudoInverse(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1., 2., 3., 4.})
	invX := mat.NewDense(2, 2, nil)
	invX.Inverse(X)

	var svd mat.SVD
	svd.Factorize(X, mat.SVDThin)
	pInv := PseudoInverse(&svd, 1e-6)

	if !mat.EqualApprox(pInv, invX, 1e-6) {
		t.Errorf("Pseudo-inverse error. Expected\n%v\nGot\n%v\n", mat.Formatted(invX), mat.Formatted(pInv))
	}
}

func TestLogspace(t *testing.T) {
	lambMin := 1.0
	lambMax := 10.0
	for i, test := range []struct {
		num    int
		expect []float64
	}{
		{
			num:    0,
			expect: nil,
		},
		{
			num:    1,
			expect: []float64{1.0},
		},
		{
			num:    2,
			expect: []float64{1.0, 10.0},
		},
	} {
		res := Logspace(lambMin, lambMax, test.num)

		if !floats.EqualApprox(res, test.expect, 1e-10) {
			t.Errorf("Test #%d failed. Expected \n%v\nGot\n%v\n", i, test.expect, res)
		}
	}
}

func TestUnionInt(t *testing.T) {
	for i, test := range []struct {
		v1     []int
		v2     []int
		expect []int
	}{
		{
			v1:     []int{0, 1, 2},
			v2:     []int{3, 4},
			expect: []int{0, 1, 2, 3, 4},
		},
		{
			v1:     []int{0, 1, 2},
			v2:     []int{0, 3},
			expect: []int{0, 1, 2, 3},
		},
	} {
		union := UnionInt(test.v1, test.v2)

		for j := range union {
			if union[j] != test.expect[j] {
				t.Errorf("Test #%d. Expected\n%v\nGot\n%v\n", i, test.expect, union)
			}
		}
	}
}

func TestCohenKappa(t *testing.T) {
	for i, test := range []struct {
		s1     []int
		s2     []int
		expect float64
		totNum int
	}{
		{
			s1:     []int{0, 1, 2, 3},
			s2:     []int{0, 1, 2, 3},
			totNum: 4,
			expect: 1.0,
		},
		{
			s1:     []int{0, 1, 2},
			s2:     []int{3, 4, 5},
			totNum: 6,
			expect: -1.0,
		},
		{
			s1:     []int{},
			s2:     []int{},
			totNum: 10,
			expect: 1.0,
		},
	} {
		kappa := CohenKappa(test.s1, test.s2, test.totNum)

		if math.Abs(kappa-test.expect) > 1e-10 {
			t.Errorf("Test #%d: unexpected kappa. Expected %f got %f", i, test.expect, kappa)
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
