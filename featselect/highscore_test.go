package featselect

import (
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestHighscore(t *testing.T) {
	highscore := NewHighscore(4)

	var node1 Node
	node1.Score = 1.0
	highscore.Insert(&node1)

	if highscore.Len() != 1 {
		t.Errorf("Highscore: Expected length 1. Got %v", highscore.Len())
	}

	var node2 Node
	node2.Score = 0.5
	highscore.Insert(&node2)

	expected := []float64{1.0, 0.5}
	scores := getScores(highscore)
	if !floats.EqualApprox(scores, expected, 1e-10) {
		t.Errorf("Highscores: Expected: %v, Got: %v", expected, scores)
	}

	// Insert a node in between
	var node3 Node
	node3.Score = 0.75
	highscore.Insert(&node3)
	scores = getScores(highscore)
	expected = []float64{1.0, 0.75, 0.5}
	if !floats.EqualApprox(scores, expected, 1e-10) {
		t.Errorf("Highscores: Expected: %v, Got: %v", expected, scores)
	}

	// Insert new node at end
	var node4 Node
	node4.Score = 0.6
	highscore.Insert(&node4)
	expected = []float64{1.0, 0.75, 0.6}
	scores = getScores(highscore)
	if !floats.EqualApprox(scores, expected, 1e-10) {
		t.Errorf("Highscores: Expected: %v, Got: %v", expected, scores)
	}
}

func getScores(h *Highscore) []float64 {
	scores := make([]float64, h.Len())
	i := 0
	for item := h.items.Front(); item != nil; item = item.Next() {
		scores[i] = item.Value.(*Node).Score
		i++
	}
	return scores
}
