package featselect

import "gonum.org/v1/gonum/mat"

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
