package featselect

import "testing"

func TestGetChildNode(t *testing.T) {
	var parent Node
	parent.model = []bool{false, true, false, false}
	parent.level = 1
	child := parent.GetChildNode(false)

	if child.level != 2 {
		t.Errorf("ChildNode: Wrong level. Expected 2, Got %v", child.level)
	}

	if child.model[1] {
		t.Errorf("ChildNode: Model is not updated correctly")
	}
}

func TestNodeEqual(t *testing.T) {
	for i, test := range []struct {
		first  *Node
		second *Node
		expect bool
	}{
		{
			first:  NewNode(0, []bool{false, false}),
			second: NewNode(1, []bool{false, false}),
			expect: false,
		},
		{
			first:  NewNode(0, []bool{false, false, false}),
			second: NewNode(0, []bool{false, false}),
			expect: false,
		},
		{
			first:  NewNode(2, []bool{false, true, false}),
			second: NewNode(2, []bool{false, false, false}),
			expect: false,
		},
		{
			first:  NewNode(2, []bool{false, true, false}),
			second: NewNode(2, []bool{false, true, false}),
			expect: true,
		},
	} {
		if NodesEqual(test.first, test.second) != test.expect {
			t.Errorf("NodeEqual: Test %v failed. Expected %v Got %v", i, test.first, test.second)
		}
	}
}
