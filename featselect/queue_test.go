package featselect

import "testing"

func TestQueue(t *testing.T) {
	queue := NewQueue(20)
	var node Node
	AddToQueue(&queue, &node)

	if queue.next != 1 {
		t.Errorf("Queue: Something went wrong during update")
	}

	first := GetFirst(&queue)

	if !NodesEqual(&node, first) {
		t.Errorf("Queue: First node does not match the only inserted node")
	}
}
