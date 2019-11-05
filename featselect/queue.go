package featselect

type Queue struct {
	nodes []*Node
	next  int
}

// Add a node to the queue
func AddToQueue(queue *Queue, node *Node) {
	queue.nodes[queue.next] = node
	queue.next += 1
	queue.next %= len(queue.nodes)
}

// Get first node in the queue
func GetFirst(queue *Queue) *Node {
	first := queue.next - 1

	if first < 0 {
		first += len(queue.nodes)
	}
	return queue.nodes[first]
}

// Create a new queue with a given capacity
func NewQueue(capacity int) Queue {
	var queue Queue
	queue.next = 0
	queue.nodes = make([]*Node, capacity)
	return queue
}
