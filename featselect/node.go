package featselect

// Node is a type that holds an array of bools (model) indicating which
// features are active. The leven field tells which level in the tree this
// node is on. The lower and upper fields represents bounds of the score
// for all models that are childrens of this node. The score field holsd
// the score of this model.
type Node struct {
	model []bool
	coeff []float64
	level int
	lower float64
	upper float64
	score float64
}

// GetChildNode creates a child not of node. parent is the parent node the boolean right is true
// if the child node should be in the left in the tree. The left child node is
// characterised with the bit corresponding to the level of the parent is set to
// false, while in the right child node it is set to true
func GetChildNode(parent *Node, right bool) *Node {
	var child Node
	child.model = make([]bool, len(parent.model))
	copy(child.model, parent.model)
	child.model[parent.level] = right
	child.level = parent.level + 1
	return &child
}

// NodesEqual compare two nodes
func NodesEqual(node1 *Node, node2 *Node) bool {
	if len(node1.model) != len(node2.model) {
		return false
	}

	if node1.level != node2.level {
		return false
	}

	for i := 0; i < len(node1.model); i++ {
		if node1.model[i] != node2.model[i] {
			return false
		}
	}
	return true
}

// NewNode creates a new node
func NewNode(level int, model []bool) Node {
	var node Node
	node.level = level
	node.model = make([]bool, len(model))
	copy(node.model, model)
	return node
}
