package featselect

import (
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
func Omp(X DesignMatrix, y []float64, tol float64) *OmpResult {
	_, ncols := X.Dims()
	res := NewOmpResult(ncols)
	residuals := mat.NewVecDense(len(y), nil)
	for i := 0; i < residuals.Len(); i++ {
		residuals.SetVec(i, y[i])
	}

	current := 0
	model := make([]bool, ncols)
	proj := make([]float64, ncols)
	for current < ncols {
		for col := 0; col < ncols; col++ {
			proj[col] = math.Abs(mat.Dot(X.ColView(col), residuals)) / (mat.Norm(X.ColView(col), 2) * mat.Norm(residuals, 2))
		}

		imax := Argmax(proj)
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
		for i := 0; i < len(y); i++ {
			residuals.SetVec(i, ypred[i]-y[i])
			rss += residuals.AtVec(i) * residuals.AtVec(i)
		}
		rss = math.Sqrt(rss / float64(len(y)))
		if rss < tol {
			break
		}
		current++
	}
	return res
}

// DotProducts calculates the dot product between a matrix and a vector
func DotProducts(X DesignMatrix, b []float64) []float64 {
	nr, nc := X.Dims()
	res := make([]float64, nr)
	for i := 0; i < nr; i++ {
		res[i] = 0.0
		for j := 0; j < nc; j++ {
			res[i] += X.At(i, j) * b[j]
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
