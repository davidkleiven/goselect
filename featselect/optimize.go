package featselect

import (
	"container/list"

	"gonum.org/v1/gonum/mat"
)

// Gcs returns the greatest common model (gcs).
// GCS is equal to the model with the largest
// amount of features, given that all "bits"
// up to start is not altered
func Gcs(model []bool, start int) []bool {
	gcsMod := make([]bool, len(model))
	copy(gcsMod, model[:start])
	for i := start; i < len(model); i++ {
		gcsMod[i] = true
	}
	return gcsMod
}

// Lcs returns the least common model. The lcs is the model with
// as few features as possible given that all "bits"
// up to start are not altered
func Lcs(model []bool, start int) []bool {
	lcsMod := make([]bool, len(model))
	copy(lcsMod, model[:start])

	for i := start; i < len(model); i++ {
		lcsMod[i] = false
	}
	return lcsMod
}

// NumFeatures returns the number of features in a model
func NumFeatures(model []bool) int {
	num := 0
	for _, flag := range model {
		if flag {
			num++
		}
	}
	return num
}

// SelectModel finds the model which minimizes AICC
func SelectModel(X *mat.Dense, y []float64) *Highscore {
	highscore := NewHighscore(100)
	queue := list.New()

	nrows, ncols := X.Dims()

	if ncols < 3 {
		panic("SelectModel: The number of features has to be larger or equal to 3.")
	}

	emptyModel := make([]bool, ncols)
	rootNode := NewNode(0, emptyModel)
	queue.PushBack(rootNode)

	log2Pruned := 0
	numChecked := 0

	for queue.Front() != nil {
		node := queue.Front().Value.(*Node)
		n := NumFeatures(node.Model)

		if n > 0 && isNewNode(node) {
			design := GetDesignMatrix(node.Model, X)
			node.Coeff = Fit(design, y)
			rss := Rss(design, node.Coeff, y)
			node.Score = -Aicc(n, nrows, rss)
			highscore.Insert(node)
			numChecked++
		}
		queue.Remove(queue.Front())

		if node.Level == ncols {
			continue
		}

		// Create the child nodes
		leftChild := node.GetChildNode(false)
		n = NumFeatures(leftChild.Model)

		if n < nrows {
			if n > 0 {
				leftChild.Lower, leftChild.Upper = BoundsAICC(leftChild.Model, leftChild.Level, X, y)
			} else {
				leftChild.Lower = -1e100
				leftChild.Upper = 1e100
			}
			if leftChild.Lower < -highscore.BestScore() || highscore.Len() == 0 {
				queue.PushBack(leftChild)
			} else {
				log2Pruned += ncols - leftChild.Level
			}
		}

		rightChild := node.GetChildNode(true)
		n = NumFeatures(rightChild.Model)
		if n < nrows {
			rightChild.Lower, rightChild.Upper = BoundsAICC(rightChild.Model, rightChild.Level, X, y)
			if rightChild.Lower < -highscore.BestScore() || highscore.Len() == 0 {
				queue.PushBack(rightChild)
			} else {
				log2Pruned += ncols - rightChild.Level
			}
		}
	}
	return highscore
}

// BruteForceSelect runs through all possible models
func BruteForceSelect(X *mat.Dense, y []float64) *Highscore {
	_, ncols := X.Dims()
	model := make([]bool, ncols)
	highscore := NewHighscore(1 << uint(ncols))
	queue := list.New()
	rootNode := NewNode(0, model)
	queue.PushBack(rootNode)

	for queue.Front() != nil {
		currentNode := queue.Front().Value.(*Node)
		if NumFeatures(currentNode.Model) > 0 && isNewNode(currentNode) {
			design := GetDesignMatrix(currentNode.Model, X)
			currentNode.Coeff = Fit(design, y)
			rss := Rss(design, currentNode.Coeff, y)
			currentNode.Score = -Aicc(NumFeatures(currentNode.Model), len(y), rss)
			highscore.Insert(currentNode)
		}
		queue.Remove(queue.Front())

		if currentNode.Level < ncols {
			queue.PushBack(currentNode.GetChildNode(false))
			queue.PushBack(currentNode.GetChildNode(true))
		}
	}
	return highscore
}

func all(model []bool) bool {
	for i := 0; i < len(model); i++ {
		if !model[i] {
			return false
		}
	}
	return true
}

func isNewNode(node *Node) bool {
	if node.Level == 0 {
		return true
	}
	return node.Model[node.Level-1]
}
