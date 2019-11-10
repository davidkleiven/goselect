package featselect

import (
	"container/list"
	"encoding/json"
)

// Highscore is a structure that holds a list of Nodes sorted by their score
type Highscore struct {
	Items    *list.List
	MaxItems int
}

// NewHighscore creates a new highscore list with maxItems entries
func NewHighscore(maxItems int) *Highscore {
	var highscore Highscore
	highscore.Items = list.New()
	highscore.MaxItems = maxItems
	return &highscore
}

// Insert insters a new node into the highscore list
func (h *Highscore) Insert(node *Node) {
	if h.Len() == h.MaxItems-1 {
		last := h.Items.Back()
		if last.Value.(*Node).Score > node.Score {
			return
		}
	}

	itemInserted := false
	for item := h.Items.Front(); item != nil; item = item.Next() {
		if item.Value == nil {
			panic("Insert: Value is nil")
		}
		if item.Value.(*Node).Score < node.Score {
			h.Items.InsertBefore(node, item)
			itemInserted = true
			if h.Items.Len() >= h.MaxItems {
				h.Items.Remove(h.Items.Back())
			}
			break
		}
	}

	if !itemInserted {
		h.Items.PushBack(node)
	}
}

// Len returns the number of items in the highscore list
func (h *Highscore) Len() int {
	return h.Items.Len()
}

// BestScore returns the best score
func (h *Highscore) BestScore() float64 {
	front := h.Items.Front()

	if front == nil {
		return 0.0
	}
	return front.Value.(*Node).Score
}

// Scores returns all the scores in the highscore list
func (h *Highscore) Scores() []float64 {
	scores := make([]float64, h.Len())
	i := 0
	for item := h.Items.Front(); item != nil; item = item.Next() {
		scores[i] = item.Value.(*Node).Score
		i++
	}
	return scores
}

// MarshalJSON creates a JSON representation of the highscore list
func (h *Highscore) MarshalJSON() ([]byte, error) {
	data := make([]*Node, h.Len())
	counter := 0
	for e := h.Items.Front(); e != nil; e = e.Next() {
		data[counter] = e.Value.(*Node)
		counter++
	}
	return json.Marshal(&struct {
		MaxItems int     `json:"maxItems"`
		Items    []*Node `json:"items"`
	}{
		MaxItems: h.MaxItems,
		Items:    data,
	})
}

// UnmarshalJSON decodes a JSON representation of the highscore list
func (h *Highscore) UnmarshalJSON(data []byte) error {
	var aux struct {
		MaxItems int     `json:"maxItems"`
		Items    []*Node `json:"items"`
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	h.MaxItems = aux.MaxItems

	for _, n := range aux.Items {
		h.Insert(n)
	}
	return nil
}

func (h *Highscore) Equal(h2 *Highscore) bool {
	if h.MaxItems != h2.MaxItems || h.Len() != h2.Len() {
		return false
	}

	for e1, e2 := h.Items.Front(), h2.Items.Front(); e1 != nil; e1, e2 = e1.Next(), e2.Next() {
		if !NodesEqual(e1.Value.(*Node), e2.Value.(*Node)) {
			return false
		}
	}
	return true
}
