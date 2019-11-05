package featselect

type Node struct {
	model []bool
	level int
}

// Create a child not of node. parent is the parent node
// the boolean right is true if the child node should be
// in the left in the tree
func GetChildNode(parent *Node, right bool) Node {
	var child Node
	child.model = make([]bool, len(parent.model))
	copy(child.model, parent.model)
	child.model[parent.level] = right
	child.level = parent.level + 1
	return child
}

// Compare two nodes
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

func NewNode(level int, model []bool) Node {
	var node Node
	node.level = level
	node.model = make([]bool, len(model))
	copy(node.model, model)
	return node
}
