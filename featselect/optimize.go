package featselect

import (
	"container/list"
	"math"

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

// SelectModel finds the model which minimizes AICC. X is the NxM design matrix, y is a vector of length
// N, highscore keeps track of the best models and cutoff is a value that is added to the lower bounds
// when judging if a node shoudl be added. The check for if a node will be added or not is this
//
// lower_bound + cutoff < current_best_score
func SelectModel(X DesignMatrix, y []float64, highscore *Highscore, sp *SearchProgress, cutoff float64, rootModel []bool) {
	queue := list.New()

	_, ncols := X.Dims()

	if ncols < 3 {
		panic("SelectModel: The number of features has to be larger or equal to 3.")
	}

	if rootModel == nil {
		rootModel = make([]bool, ncols)
	} else {
		if len(rootModel) != ncols {
			panic("SelectModel: Inconsistent length of rootModel.")
		}
	}

	rootNode := NewNode(0, rootModel)
	queue.PushBack(rootNode)

	log2Pruned := 0.0
	numChecked := 0
	channels := NewOptimizeChannels()

	numScoreWorkers := 8
	for i := 0; i < numScoreWorkers; i++ {
		go ScoreWorker(channels, X, y)
	}

	numChildWorkers := 4
	for i := 0; i < numChildWorkers; i++ {
		go CreateLeftChild(channels, X, y, cutoff, highscore)
		go CreateRightChild(channels, X, y, cutoff, highscore)
	}

	numInProgress := 1
	channels.node <- rootNode

	for {
		select {
		case ns := <-channels.nodeScore:
			numInProgress--
			sp.Set(highscore.BestScore(), numChecked, log2Pruned)

			if isNewNode(ns) {
				highscore.Insert(ns)
				numChecked++
			}

			if ns.Level < ncols {
				channels.wantLeftNode <- ns
				channels.wantRightNode <- ns
				numInProgress += 2
			}

			if numInProgress <= 0 {
				channels.close()
				return
			}
		case prLevel := <-channels.prunedLevel:
			log2Pruned = NewLog2Pruned(log2Pruned, ncols-prLevel)
			numInProgress--
			if numInProgress <= 0 {
				channels.close()
				return
			}
		}
	}
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
	return node.WasFlipped
}

// OptimizeChannels is s struct that holds the required channels for
// communication in the search
type OptimizeChannels struct {
	node          chan *Node
	nodeScore     chan *Node
	wantLeftNode  chan *Node
	wantRightNode chan *Node
	ks            chan bool
	prunedLevel   chan int
}

// NewOptimizeChannels creates a new instance of the OptimizeChannels struct
func NewOptimizeChannels() *OptimizeChannels {
	var ch OptimizeChannels
	ch.node = make(chan *Node)
	ch.nodeScore = make(chan *Node)
	ch.wantLeftNode = make(chan *Node, 2)
	ch.wantRightNode = make(chan *Node, 2)
	ch.prunedLevel = make(chan int)
	return &ch
}

func (o *OptimizeChannels) close() {
	close(o.node)
	close(o.nodeScore)
	close(o.wantLeftNode)
	close(o.wantRightNode)
	close(o.prunedLevel)
}

// ScoreWorker is a function that calculates the score of a node
func ScoreWorker(channels *OptimizeChannels, X DesignMatrix, y []float64) {
	nrows, _ := X.Dims()
	for n := range channels.node {
		numFeat := NumFeatures(n.Model)

		if numFeat > 0 && isNewNode(n) {
			design := GetDesignMatrix(n.Model, X)
			n.Coeff = Fit(design, y)
			rss := Rss(design, n.Coeff, y)
			n.Score = -Aicc(numFeat, nrows, rss)
		} else {
			n.Score = -math.MaxFloat64
		}
		channels.nodeScore <- n
	}
}

// CreateChild creates a child not of a parent. Returns nil if number of rows is zero or the lower bound
// is lower than the current best score
func CreateChild(node *Node, flip bool, X DesignMatrix, y []float64, cutoff float64, h *Highscore) *Node {
	child := node.GetChildNode(flip)
	n := NumFeatures(child.Model)
	nrows, _ := X.Dims()
	if n < nrows {
		if n > 0 {
			child.Lower, child.Upper = BoundsAICC(child.Model, child.Level, X, y)
		} else {
			child.Lower = -1e100
			child.Upper = 1e100
		}
	} else {
		return nil
	}

	if child.Lower > -h.BestScore() && h.Len() > 0 {
		return nil
	}
	return child
}

// CreateLeftChild creates left child of a parent node
func CreateLeftChild(ch *OptimizeChannels, X DesignMatrix, y []float64, cutoff float64, h *Highscore) {
	for parent := range ch.wantLeftNode {
		n := CreateChild(parent, false, X, y, cutoff, h)
		if n == nil {
			ch.prunedLevel <- parent.Level
		} else {
			ch.node <- n
		}
	}
}

// CreateRightChild creates a right child not of a parent
func CreateRightChild(ch *OptimizeChannels, X DesignMatrix, y []float64, cutoff float64, h *Highscore) {
	for parent := range ch.wantRightNode {
		n := CreateChild(parent, true, X, y, cutoff, h)
		if n == nil {
			ch.prunedLevel <- parent.Level
		} else {
			ch.node <- n
		}
	}
}
