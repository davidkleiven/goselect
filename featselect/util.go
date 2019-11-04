package featselect

import "gonum.org/v1/gonum/mat"

func getDesignMatrix(model []bool, X *mat.Dense) *mat.Dense {
	n, _ := X.Dims()
	numFeat := numFeatures(model)

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
			col += 1
		}
	}
	return design
}
