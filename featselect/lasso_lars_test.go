package featselect

import (
	"math"
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestLassoLars(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()
	data := NewNormalizedData(X, y)
	var estimator MorsePenroseCD
	res := LassoLars(data, 1e-16, &estimator)
	Path2Unnormalized(data, res)
	last := res[len(res)-1]

	for i := range last.Coeff {
		cf := last.Coeff[i]
		if last.Selection[i] == 0 {
			if math.Abs(cf+3.0) > 1e-10 {
				t.Errorf("bias term should be -3.0, got %f", cf)
			}
		} else if last.Selection[i] == 1 {
			if math.Abs(cf-3.0) > 1e-10 {
				t.Errorf("Expecte linear term to be 3.0, got %f", cf)
			}
		} else if last.Selection[i] == 2 {
			if math.Abs(cf-0.2) > 1e-10 {
				t.Errorf("quad term should be 0.2, got %f", cf)
			}
		} else {
			if math.Abs(cf) > 1e-10 {
				t.Errorf("expected term to be zero. got %f", cf)
			}
		}
	}
}

func TestLassoLarsMonotoneCovariance(t *testing.T) {
	for i, test := range []struct {
		f testfeatselect.DemoSet
	}{
		{
			f: testfeatselect.GetExampleXY,
		},
		{
			f: testfeatselect.GetExampleAllModelsWrong,
		},
	} {
		X, y := test.f()
		nr, numFeat := X.Dims()
		data := NewNormalizedData(X, y)
		var estimator MorsePenroseCD
		res := LassoLars(data, 1e-10, &estimator)

		oldCov := make([]float64, numFeat)
		for j := range oldCov {
			oldCov[j] = math.MaxFloat64
		}

		for _, node := range res {
			allCoeff := FullCoeffVector(numFeat, node.Selection, node.Coeff)
			allCoeffVec := mat.NewVecDense(len(allCoeff), allCoeff)
			pred := mat.NewVecDense(nr, nil)
			pred.MulVec(data.X, allCoeffVec)
			devVec := mat.NewVecDense(nr, nil)

			for j := 0; j < nr; j++ {
				devVec.SetVec(j, y[j]-pred.AtVec(j))
			}

			cov := mat.NewVecDense(numFeat, nil)
			cov.MulVec(data.X.T(), devVec)

			for j := 0; j < numFeat; j++ {
				curCov := math.Abs(cov.AtVec(j))
				if curCov > oldCov[j] {
					t.Errorf("Test #%d: Covariances are not decreasing. Prev %e, Current %v", i, oldCov[j], curCov)
					oldCov[j] = curCov
				}
			}
		}
	}
}

func TestSparseCoeffConversion(t *testing.T) {
	n1 := LassoLarsNode{Coeff: []float64{1.0, 2.0}, Selection: []int{0, 1}, Lamb: 2.0}
	n2 := LassoLarsNode{Coeff: []float64{2.0, 3.0}, Selection: []int{2, 3}, Lamb: 1.0}
	lp := []*LassoLarsNode{&n1, &n2}

	sp := LassoNodesSlice2SparsCoeff(lp)

	expectSp := []SparseCoeff{
		{Coeff: []float64{1.0, 2.0}, Selection: []int{0, 1}},
		{Coeff: []float64{2.0, 3.0}, Selection: []int{2, 3}},
	}

	for i := range expectSp {
		if !floats.EqualApprox(sp[i].Coeff, expectSp[i].Coeff, 1e-10) {
			t.Errorf("Sparse conversion of lasso lars slice failed")
		}
	}
}
