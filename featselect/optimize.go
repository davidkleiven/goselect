package featselect

import (
	"container/list"
	"math"

	"gonum.org/v1/gonum/mat"
)

// MaxFeatures limits the maximum number of features allowed in the models considered.
// It is given as a fraction of the number of available features
const MaxFeatures = 0.8

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
	maxFeatures := int(math.Min(MaxFeatures*float64(nrows), float64(nrows-2)))
	log2Pruned := 0
	numChecked := 0.0
	for queue.Front() != nil {
		node := queue.Front().Value.(*Node)
		n := NumFeatures(node.model)

		if n > 0 {
			design := GetDesignMatrix(node.model, X)
			node.coeff = Fit(design, y)
			rss := Rss(design, node.coeff, y)
			aicc := Aicc(n, nrows, rss)
			node.score = -aicc
			highscore.Insert(node)
			numChecked++
		}
		queue.Remove(queue.Front())

		// Create the child nodes
		leftChild := GetChildNode(node, false)
		n = NumFeatures(leftChild.model)

		if n > 0 && n < maxFeatures {
			leftChild.lower, leftChild.upper = BoundsAICC(leftChild.model, leftChild.level, X, y)
			if leftChild.lower < math.Abs(highscore.BestScore()) {
				queue.PushBack(leftChild)
			} else {
				log2Pruned += ncols - leftChild.level
			}
		}

		rightChild := GetChildNode(node, true)
		n = NumFeatures(rightChild.model)
		if n > 0 && n < maxFeatures {
			rightChild.lower, rightChild.upper = BoundsAICC(rightChild.model, rightChild.level, X, y)
			if rightChild.lower < math.Abs(highscore.BestScore()) {
				queue.PushBack(rightChild)
			} else {
				log2Pruned += ncols - rightChild.level
			}
		}
	}
	return highscore
}
