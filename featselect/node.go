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
