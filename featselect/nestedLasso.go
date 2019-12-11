package featselect

import (
	"fmt"
)

// PureLassoLarsPathWithCrit is a type that holds all the nodes in addition to
// information on the Aicc and Bic values
type PureLassoLarsPathWithCrit struct {
	Nodes []*LassoLarsNode
	Aicc  []float64
	Bic   []float64
}

// NestedLassoLars is a type that holds a lasso path in addition to AICC and BIC
// along the path
type NestedLassoLars struct {
	Dset  *Dataset
	Paths []PureLassoLarsPathWithCrit
}

// NestedLasso performs a sequence of LASSO calculations where the least important
// features are removed on each iteration
func NestedLasso(data *Dataset, lambMin float64, keep float64, estimator CDParam) NestedLassoLars {
	var res NestedLassoLars

	iter := 0
	yCpy := make([]float64, len(data.Y))
	copy(yCpy, data.Y)

	res.Dset = data
	curDset := data.Copy()
	normD := NewNormalizedData(curDset.X, curDset.Y)
	for {
		iter++
		newPath := LassoLars(normD, lambMin, estimator)

		var path LassoLarsPath
		path.LassoLarsNodes = newPath

		path.Dset = curDset
		aicc := path.GetCriteria(Aicc)
		bic := path.GetCriteria(Bic)

		res.Paths = append(res.Paths, PureLassoLarsPathWithCrit{Nodes: newPath, Aicc: aicc, Bic: bic})

		fmt.Printf("Iteration %d:\n", iter)
		PrintHighscore(&path, aicc, bic, 10)

		_, currentNumCoeff := curDset.X.Dims()
		newNum := int(keep * float64(currentNumCoeff))
		survivors := path.PickMostRelevantFeatures(newNum)

		// Map selection
		for _, v := range newPath {
			v.Selection = MapSelectionByName(data, curDset.Names, v.Selection)
		}

		if newNum <= 2 {
			break
		}

		// Update the current dataset and the normalized dataset
		curDset = curDset.GetSubset(survivors)
		normD = NewNormalizedData(curDset.X, curDset.Y)
	}

	// Undo the normalisation of all paths before returning
	normD = NewNormalizedData(data.X, data.Y)
	for _, v := range res.Paths {
		Path2Unnormalized(normD, v.Nodes)
	}
	return res
}

// MapSelectionByName maps the selected features by its name in the corresponding
// dataset
func MapSelectionByName(orig *Dataset, names []string, selection []int) []int {
	for i := range selection {
		selection[i] = orig.FeatNoByName(names[selection[i]])
	}
	return selection
}
