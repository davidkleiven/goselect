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
func (h *Highscore) Insert(node *Node) {
	if h.Len() == h.maxItems-1 {
		last := h.items.Back()
		if last.Value.(*Node).Score > node.Score {
			return
		}
	}

	itemInserted := false
	for item := h.items.Front(); item != nil; item = item.Next() {
		if item.Value == nil {
			panic("Insert: Value is nil")
		}
		if item.Value.(*Node).Score < node.Score {
			h.items.InsertBefore(node, item)
			itemInserted = true
			if h.items.Len() >= h.maxItems {
				h.items.Remove(h.items.Back())
			}
			break
		}
	}

	if !itemInserted {
		h.items.PushBack(node)
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
	return front.Value.(*Node).Score
}

// Scores returns all the scores in the highscore list
func (h *Highscore) Scores() []float64 {
	scores := make([]float64, h.Len())
	i := 0
	for item := h.items.Front(); item != nil; item = item.Next() {
		scores[i] = item.Value.(*Node).Score
		i++
	}
	return scores
}
