package featselect

import (
	"gonum.org/v1/gonum/mat"
)

// GetDesignMatrix returns the design matrix corresponding to the passed model
func GetDesignMatrix(model []bool, X *mat.Dense) *mat.Dense {
	n, _ := X.Dims()
	numFeat := NumFeatures(model)

	if numFeat == 0 {
		panic("getDesignMatrix: No features in model")
	}
	design := mat.NewDense(n, numFeat, nil)

	col := 0
	for i := 0; i < len(model); i++ {
		if model[i] {
			colView := X.ColView(i)

			for j := 0; j < n; j++ {
				design.Set(j, col, colView.At(j, 0))
			}
			col++
		}
	}
	return design
}

func All(a []int, value int) bool {
	for i := 0; i < len(a); i++ {
		if a[i] != value {
			return false
		}
	}
	return true
}

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

func Sum(a []int) int {
	s := 0
	for _, v := range a {
		s += v
	}
	return s
}
