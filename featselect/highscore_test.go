package featselect

import (
	"bytes"
	"encoding/json"
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

func TestSaveHighscore(t *testing.T) {
	n1 := NewNode(0, []bool{true, false, false})
	n1.Score = 1.0
	n2 := NewNode(1, []bool{false, true, true})
	n2.Score = 0.5

	h := NewHighscore(100)
	h.Insert(n1)
	h.Insert(n2)

	buf := bytes.NewBufferString("")
	json.NewEncoder(buf).Encode(h)
	res := buf.String()
	expected := "{\"maxItems\":100,\"items\":["

	buf1 := bytes.NewBufferString("")
	json.NewEncoder(buf1).Encode(&n1)
	buf2 := bytes.NewBufferString("")
	json.NewEncoder(buf2).Encode(&n2)
	n1Str := buf1.String()
	n2Str := buf2.String()
	expected += n1Str[:len(n1Str)-1] + "," + n2Str[:len(n2Str)-1] + "]}\n"

	if res != expected {
		t.Errorf("SaveHighscore:\nExpected.\n%v\nGot\n%v\n", expected, res)
	}

	h2 := NewHighscore(2)
	json.NewDecoder(buf).Decode(h2)

	if !h.Equal(h2) {
		t.Errorf("Expected:\n%+v\nGot:\n%+v\n", h, h2)
	}
}

func getScores(h *Highscore) []float64 {
	scores := make([]float64, h.Len())
	i := 0
	for item := h.Items.Front(); item != nil; item = item.Next() {
		scores[i] = item.Value.(*Node).Score
		i++
	}
	return scores
}
