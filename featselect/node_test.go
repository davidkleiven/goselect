package featselect

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestGetChildNode(t *testing.T) {
	var parent Node
	parent.Model = []bool{true, false, false, false}
	parent.Level = 1
	child := parent.GetChildNode(false)

	if child.Level != 2 {
		t.Errorf("ChildNode: Wrong level. Expected 2, Got %v", child.Level)
	}

	if child.Model[1] {
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

func TestNodeJSON(t *testing.T) {
	buf := bytes.NewBufferString("")
	origNode := Node{
		Model: []bool{false, true, false},
		Coeff: []float64{5.1},
		Level: 2,
		Lower: -4.1,
		Upper: 4.1,
		Score: 2.2,
	}
	json.NewEncoder(buf).Encode(&origNode)

	res := buf.String()
	expected := "{\"selected\":[1],\"totNumFeatures\":3,\"lower\":-4.1,\"upper\":4.1,\"score\":2.2,\"level\":2,\"coeff\":[5.1]}\n"
	if res != expected {
		t.Errorf("NodeJSON: Expected\n%v\nGot\n%v", expected, res)
	}

	var n2 Node
	json.NewDecoder(buf).Decode(&n2)

	if !NodesEqual(&origNode, &n2) {
		t.Errorf("Expected:\n%+v\nGot\n%+v\n", origNode, n2)
	}
}

func TestNodeMemoryEstimate(t *testing.T) {
	for i, test := range []struct {
		model  []bool
		expect int
	}{
		{
			model:  []bool{true, true, true, true},
			expect: 65,
		},
		{
			model:  []bool{true, true, false, true},
			expect: 57,
		},
	} {
		n := NewNode(0, test.model)
		est := n.EstimateMemory()

		if est != test.expect {
			t.Errorf("Test #%v failed. Expected: %v, Got: %v", i, test.expect, est)
		}
	}
}

func TestNodeToSparseCoeff(t *testing.T) {
	model := make([]bool, 6)
	model[0] = true
	model[4] = true
	node := NewNode(1, model)
	node.Coeff = []float64{2.0, 4.0}
	sp := node.ToSparseCoeff()

	spExp := SparseCoeff{Coeff: []float64{2.0, 4.0}, Selection: []int{0, 4}}

	for i := range sp.Selection {
		if sp.Selection[i] != spExp.Selection[i] {
			t.Errorf("To sparse coeff failed")
		}
	}
}
