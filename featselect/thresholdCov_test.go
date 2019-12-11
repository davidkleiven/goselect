package featselect

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMatrixThreshold(t *testing.T) {
	for i, test := range []struct {
		X         *mat.Dense
		expect    *mat.Dense
		threshold float64
	}{
		{
			X:         mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
			expect:    mat.NewDense(2, 2, []float64{0.0, 0.0, 3.0, 4.0}),
			threshold: 2.5,
		},
	} {
		var op ThresholdOperator
		op.threshold = test.threshold
		op.Apply(test.X)

		if !mat.EqualApprox(test.X, test.expect, 1e-10) {
			t.Errorf("Test #%d: Expect\n%v\nGot\n%v\n", i, mat.Formatted(test.X), mat.Formatted(test.expect))
		}
	}
}

func TestCovMatrix(t *testing.T) {
	X := mat.NewDense(2, 2, []float64{1., 2., 3., 4.})
	expect := mat.NewDense(2, 2, []float64{5., 7., 7., 10.})
	cov := CovarianceMatrix(X)

	if !mat.EqualApprox(cov, expect, 1e-10) {
		t.Errorf("Convariance failed. Expected\n%v\nGot\n%v\n", mat.Formatted(expect), mat.Formatted(cov))
	}
}

func TestRandomSplit(t *testing.T) {
	n := 20
	X := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			X.Set(i, j, float64(i*j))
		}
	}

	numAttemps := 50
	rand.Seed(0)
	for i := 0; i < numAttemps; i++ {
		mat1, mat2 := RandomRowSplit(X, 16)

		r1, c1 := mat1.Dims()
		if r1 != 16 || c1 != n {
			t.Errorf("Inconsistent size of mat1 (%d, %d)", r1, c1)
		}

		r2, c2 := mat2.Dims()
		if r2 != 4 || c2 != n {
			t.Errorf("Inconsistent size for mat2 (%d, %d)", r2, c2)
		}

	outer:
		for i := 0; i < r2; i++ {
			for j := 0; j < r1; j++ {
				if math.Abs(mat2.At(i, 1)-mat1.At(j, 1)) < 1e-10 {
					t.Errorf("The two matrices contains the same rows. Mat1:\n%v\nMat2\n%v\n", mat.Formatted(mat1), mat.Formatted(mat2))
					break outer
				}
			}
		}
	}
}

func TestFNormDiff(t *testing.T) {
	channel := make(chan ValueGridPt)

	mat1 := mat.NewDense(2, 2, []float64{1., 2., 3., 4.})
	mat2 := mat.NewDense(2, 2, []float64{-1., 2., -3., 5.})
	go CalculateFNormDiff(mat1, mat2, 1, 0.0, channel)
	res := <-channel

	if res.gridPt != 1 {
		t.Errorf("Unexpected grid point should be 1. Got %d", res.gridPt)
	}

	expect := math.Sqrt(41.0)
	if math.Abs(res.value-expect) > 1e-10 {
		t.Errorf("Unexpected norm. Expected %f god %f", expect, res.value)
	}
}

func TestL2Consistent(t *testing.T) {
	n := 20
	X := mat.NewDense(n, n, nil)
	fn := func(i, j int, v float64) float64 { return float64(i * j) }
	X.Apply(fn, X)

	// TODO: Now we just check that the code runs. Look for a better test case.
	L2ConsistentCovTO(X, 50, 10)
}
