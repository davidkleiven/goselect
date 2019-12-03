package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// GetDesignMatrix returns the design matrix corresponding to the passed model
func GetDesignMatrix(model []bool, X mat.Matrix) mat.Matrix {
	n, _ := X.Dims()
	numFeat := NumFeatures(model)

	if numFeat == 0 {
		panic("getDesignMatrix: No features in model")
	}
	design := mat.NewDense(n, numFeat, nil)

	col := 0
	for i := 0; i < len(model); i++ {
		if model[i] {
			for j := 0; j < n; j++ {
				design.Set(j, col, X.At(j, i))
			}
			col++
		}
	}
	return design
}

// All checks if all elements in a  is equal to value
func All(a []int, value int) bool {
	for i := 0; i < len(a); i++ {
		if a[i] != value {
			return false
		}
	}
	return true
}

// IterProduct mimics the product function in the itertools module of python
func IterProduct(values []int, repeat int) [][]int {
	res := make([][]int, 1)

	for r := 0; r < repeat; r++ {
		nItems := len(res)
		updatedRes := make([][]int, 0)
		for j := 0; j < nItems; j++ {
			for i := 0; i < len(values); i++ {
				newRow := append(res[j], values[i])
				updatedRes = append(updatedRes, newRow)
			}
		}
		res = updatedRes
	}
	return res
}

// Sum sums all elements in a
func Sum(a []int) int {
	s := 0
	for _, v := range a {
		s += v
	}
	return s
}

// NewLog2Pruned updates the number of pruned solutions.
// current is log2 of the current number of pruned solutions
// numPruned is log2 of the new number of pruned solutions
func NewLog2Pruned(current float64, numPruned int) float64 {
	diff := float64(numPruned) - current
	return current + math.Log2(1+math.Pow(2, diff))
}

// RearrangeDense changes the order of the columns in the matrix X such that they appear
// in the order dictated by colOrder. If colOrder = [2, 0, 4, ...], the first column
// in the new matrix is the third column in the original matrix, the second column in
// the new matrix is the first column in the original etc.
func RearrangeDense(X *mat.Dense, colOrder []int) *mat.Dense {
	nr, nc := X.Dims()

	newMat := mat.NewDense(nr, nc, nil)

	inserted := make([]bool, nc)
	numInserted := 0
	for i := 0; i < len(colOrder); i++ {
		if colOrder[i] != -1 {
			for j := 0; j < nr; j++ {
				newMat.Set(j, i, X.At(j, colOrder[i]))
			}
			inserted[colOrder[i]] = true
			numInserted++
		}
	}

	// Transfer the remaining columns in the original order
	for i := 0; i < len(inserted); i++ {
		if !inserted[i] {
			for j := 0; j < nr; j++ {
				newMat.Set(j, numInserted, X.At(j, i))
			}
			numInserted++
		}
	}
	return newMat
}

// Selected2Model converts a list of selected features into a boolean
// array of true/false indicating whether the feature is selected or not
func Selected2Model(selected []int, numFeatures int) []bool {
	model := make([]bool, numFeatures)

	for _, v := range selected {
		model[v] = true
	}
	return model
}

// Mean calculates the mean of an array
func Mean(v []float64) float64 {
	mu := 0.0
	for i := 0; i < len(v); i++ {
		mu += v[i]
	}
	return mu / float64(len(v))
}

// Std calculates the standard deviation of an array
func Std(v []float64) float64 {
	if len(v) <= 1 {
		return 0.0
	}

	sigmaSq := 0.0
	mu := Mean(v)

	for i := 0; i < len(v); i++ {
		sigmaSq += (v[i] - mu) * (v[i] - mu)
	}
	return math.Sqrt(sigmaSq / float64(len(v)-1))
}

// NormalizeArray normalizes an array to unit variance and zero mean
func NormalizeArray(v []float64) {
	mu := Mean(v)
	std := Std(v)

	for i := 0; i < len(v); i++ {
		v[i] = (v[i] - mu) / std
	}
}

// NormalizeRows normalizes all rows to unit variance and zero mean
func NormalizeRows(X *mat.Dense) {
	nrows, _ := X.Dims()
	for i := 0; i < nrows; i++ {
		NormalizeArray(X.RawRowView(i))
	}
}

// NormalizeCols normalizes all columnss to unit variance and zero mean
func NormalizeCols(X *mat.Dense) {
	nrows, ncols := X.Dims()

	tmp := make([]float64, nrows)
	for col := 0; col < ncols; col++ {
		for row := 0; row < nrows; row++ {
			tmp[row] = X.At(row, col)
		}
		NormalizeArray(tmp)
		for row := 0; row < nrows; row++ {
			X.Set(row, col, tmp[row])
		}
	}
}

// MaxInt returns the maximum value in an array
func MaxInt(a []int) int {
	max := math.MinInt32

	for _, v := range a {
		if v > max {
			max = v
		}
	}
	return max
}

// MinInt returns the minumum value in an int array
func MinInt(a []int) int {
	min := math.MaxInt32

	for _, v := range a {
		if v < min {
			min = v
		}
	}
	return min
}

// FullCoeffVector creates a vector containing the coefficient for all features
// features that are not selected will have a coefficient of zero
func FullCoeffVector(numFeat int, selection []int, coeff []float64) []float64 {
	res := make([]float64, numFeat)
	for i, v := range selection {
		res[v] = coeff[i]
	}
	return res
}
