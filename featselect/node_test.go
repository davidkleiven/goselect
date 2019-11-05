package featselect

import "testing"

func TestGetChildNode(t *testing.T) {
	var parent Node
	parent.model = []bool{false, true, false, false}
	parent.level = 1
	child := GetChildNode(&parent, false)

	if child.level != 2 {
		t.Errorf("ChildNode: Wrong level. Expected 2, Got %v", child.level)
	}

	if child.model[1] {
		t.Errorf("ChildNode: Model is not updated correctly")
	}
}
