package featselect

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// OmpResult is a structure that holds the result of the Orthogonal Matching Pursuit
// Coeff is the fitted coefficients, Order is the order the coefficients where included.
// Order[0] is the index of the coefficient that was first included
type OmpResult struct {
	Coeff []float64
	Order []int
}

// NewOmpResult returns a new instance of OmpResult.
// numFeatures is the  number of features in the dataset
func NewOmpResult(numFeatures int) *OmpResult {
	var res OmpResult
	res.Coeff = make([]float64, numFeatures)
	res.Order = make([]int, numFeatures)

	for i := 0; i < numFeatures; i++ {
		res.Order[i] = -1
	}
	return &res
}

// Omp performs Orthogonal Matching Pursuit
func Omp(X mat.Matrix, y []float64, tol float64) *OmpResult {
	_, ncols := X.Dims()
	res := NewOmpResult(ncols)
	residuals := mat.NewVecDense(len(y), nil)
	for i := 0; i < residuals.Len(); i++ {
		residuals.SetVec(i, y[i])
	}

	current := 0
	model := make([]bool, ncols)
	proj := mat.NewVecDense(ncols, nil)
	innerProds := mat.NewDense(ncols, ncols, nil)
	innerProds.Mul(X.T(), X)
	norms := mat.NewVecDense(ncols, nil)
	for i := 0; i < ncols; i++ {
		norms.SetVec(i, math.Sqrt(innerProds.At(i, i)))
	}

	for current < ncols {
		resNorm := mat.Norm(residuals, 2)
		proj.MulVec(X.T(), residuals)
		for i := 0; i < proj.Len(); i++ {
			proj.SetVec(i, math.Abs(proj.AtVec(i)/(resNorm*norms.AtVec(i))))
		}

		imax := Argmax(proj.RawVector().Data)
		model[imax] = true
		res.Order[current] = imax

		design := GetDesignMatrix(model, X)
		coeff := Fit(design, y)
		selected := SelectedFeatures(model)

		for i, v := range selected {
			res.Coeff[v] = coeff[i]
		}

		ypred := Predict(design, coeff)
		rss := 0.0
		for i := 0; i < len(ypred); i++ {
			residuals.SetVec(i, ypred[i]-y[i])
			rss += math.Pow(y[i]-ypred[i], 2)
		}
		rss = math.Sqrt(rss / float64(len(y)))
		if rss < tol {
			break
		}
		current++
	}
	fmt.Printf("%v", res.Order)

	// Trim the results search for either the first duplicate or first -1
	foundBefore := make([]bool, len(res.Order))
	for i, v := range res.Order {
		if v == -1 || foundBefore[v] {
			res.Order = res.Order[:i]
			break
		} else {
			foundBefore[v] = true
		}
	}
	return res
}

// Abs calculate absolute value of the max element
func Abs(x []float64) []float64 {
	for i := 0; i < len(x); i++ {
		x[i] = math.Abs(x[i])
	}
	return x
}

// Argmax returns the index of the maximum item
func Argmax(v []float64) int {
	imax := 0
	maxVal := v[0]
	for i := 0; i < len(v); i++ {
		if v[i] > maxVal {
			imax = i
			maxVal = v[i]
		}
	}
	return imax
}

// SelectedFeatures return indices of selected features
func SelectedFeatures(model []bool) []int {
	res := make([]int, NumFeatures(model))
	counter := 0
	for i, v := range model {
		if v {
			res[counter] = i
			counter++
		}
	}
	return res
}
