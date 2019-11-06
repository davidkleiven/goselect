package featselect

import (
	"container/list"
)

// Highscore is a structure that holds a list of Nodes sorted by their score
type Highscore struct {
	items    *list.List
	maxItems int
}

// NewHighscore creates a new highscore list with maxItems entries
func NewHighscore(maxItems int) *Highscore {
	var highscore Highscore
	highscore.items = list.New()
	highscore.maxItems = maxItems
	return &highscore
}

// Insert insters a new node into the highscore list
func (highscore *Highscore) Insert(node *Node) {
	if highscore.Len() == highscore.maxItems-1 {
		last := highscore.items.Back()
		if last.Value.(*Node).score > node.score {
			return
		}
	}

	itemInserted := false
	for item := highscore.items.Front(); item != nil; item = item.Next() {
		if item.Value == nil {
			panic("Insert: Value is nil")
		}
		if item.Value.(*Node).score < node.score {
			highscore.items.InsertBefore(node, item)
			itemInserted = true
			if highscore.items.Len() >= highscore.maxItems {
				highscore.items.Remove(highscore.items.Back())
			}
			break
		}
	}

	if !itemInserted {
		highscore.items.PushBack(node)
	}
}

// Len returns the number of items in the highscore list
func (h *Highscore) Len() int {
	return h.items.Len()
}

// BestScore returns the best score
func (h *Highscore) BestScore() float64 {
	front := h.items.Front()

	if front == nil {
		return 0.0
	}
	return front.Value.(*Node).score
}
