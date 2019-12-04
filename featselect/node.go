package featselect

import (
	"encoding/json"
	"math"

	"gonum.org/v1/gonum/floats"
)

// Node is a type that holds an array of bools (model) indicating which
// features are active. The leven field tells which level in the tree this
// node is on. The lower and upper fields represents bounds of the score
// for all models that are childrens of this node. The score field holsd
// the score of this model.
type Node struct {
	Model      []bool
	Coeff      []float64
	Level      int
	Lower      float64
	Upper      float64
	Score      float64
	WasFlipped bool
}

// GetChildNode creates a child not of node. If flip is true, the "bit" at
// parent.Level is flipped
func (n *Node) GetChildNode(flip bool) *Node {
	var child Node
	child.Model = make([]bool, len(n.Model))
	copy(child.Model, n.Model)
	if flip {
		child.Model[n.Level] = !child.Model[n.Level]
	}
	child.WasFlipped = flip
	child.Level = n.Level + 1
	return &child
}

// NodesEqual compare two nodes
func NodesEqual(node1 *Node, node2 *Node) bool {
	if len(node1.Model) != len(node2.Model) {
		return false
	}

	if node1.Level != node2.Level {
		return false
	}

	for i := 0; i < len(node1.Model); i++ {
		if node1.Model[i] != node2.Model[i] {
			return false
		}
	}

	tol := 1e-10
	if math.Abs(node1.Lower-node2.Lower) > tol || math.Abs(node1.Upper-node2.Upper) > tol || math.Abs(node1.Score-node2.Score) > tol {
		return false
	}

	if !floats.EqualApprox(node1.Coeff, node2.Coeff, tol) {
		return false
	}
	return true
}

// NewNode creates a new node
func NewNode(level int, model []bool) *Node {
	var node Node
	node.Level = level
	node.Model = make([]bool, len(model))
	copy(node.Model, model)
	node.Coeff = make([]float64, NumFeatures(model))
	return &node
}

// Fields that will be written to JSON file when the Node
// struct is jsonified
type nodeJsonified struct {
	Selected       []int     `json:"selected"`
	TotNumFeatures int       `json:"totNumFeatures"`
	Lower          float64   `json:"lower"`
	Upper          float64   `json:"upper"`
	Score          float64   `json:"score"`
	Level          int       `json:"level"`
	Coeff          []float64 `json:"coeff"`
}

// MarshalJSON converts a node to JSON representation
func (n *Node) MarshalJSON() ([]byte, error) {
	selected := make([]int, NumFeatures(n.Model))
	numInserted := 0
	for i, v := range n.Model {
		if v {
			selected[numInserted] = i
			numInserted++
		}
	}
	return json.Marshal(&nodeJsonified{
		Selected:       selected,
		TotNumFeatures: len(n.Model),
		Lower:          n.Lower,
		Upper:          n.Upper,
		Score:          n.Score,
		Level:          n.Level,
		Coeff:          n.Coeff,
	})
}

// UnmarshalJSON decodes a JSON representation of a Node
func (n *Node) UnmarshalJSON(data []byte) error {
	var aux nodeJsonified
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	n.Lower = aux.Lower
	n.Upper = aux.Upper
	n.Score = aux.Score
	n.Level = aux.Level
	n.Coeff = aux.Coeff
	n.Model = make([]bool, aux.TotNumFeatures)

	for _, v := range aux.Selected {
		n.Model[v] = true
	}
	return nil
}

// EstimateMemory estimate the memory consumption of the node in bytes
func (n *Node) EstimateMemory() int {
	estimate := len(n.Model)
	estimate += 8 * len(n.Coeff)

	// Level attribute (int)
	estimate += 4

	// Lower, upper score
	estimate += 3 * 8

	// WasFlipped
	estimate++
	return estimate
}

// ToSparseCoeff converts a node into a SparseCoeff structure
func (n *Node) ToSparseCoeff() SparseCoeff {
	var sp SparseCoeff
	nFeat := NumFeatures(n.Model)
	sp.Coeff = n.Coeff
	sp.Selection = make([]int, nFeat)

	sIdx := 0
	for i, v := range n.Model {
		if v {
			sp.Selection[sIdx] = i
			sIdx++
		}
	}
	return sp
}
