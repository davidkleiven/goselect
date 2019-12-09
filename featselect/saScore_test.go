package featselect

import (
	"math"
	"testing"
)

func TestSAItemInit(t *testing.T) {
	for _, test := range []struct {
		model     []bool
		selection []int
	}{
		{
			model:     nil,
			selection: nil,
		},
		{
			model:     []bool{true, false, false},
			selection: []int{0},
		},
	} {
		item := NewSAItem(test.model)

		if test.model == nil && item.Selection != nil {
			t.Errorf("Expected nil, got %v", item.Selection)
		} else {
			for j := range test.selection {
				if test.selection[j] != item.Selection[j] {
					t.Errorf("Expected\n%v\ngot\n%v\n", test.selection, item.Selection)
				}
			}
		}
	}
}

func TestSAScore(t *testing.T) {
	score := NewSAScore(3)

	if score.Cap != 3 {
		t.Errorf("Incorrect capacity after initialisation")
	}

	node1 := NewSAItem([]bool{true, false, true, false, false, false})
	node1.Score = 1.0

	node2 := NewSAItem([]bool{false, true})
	node2.Score = 2.0

	node3 := NewSAItem([]bool{true, true})
	node3.Score = 0.9

	node4 := NewSAItem([]bool{true, false, false})
	node4.Score = 1.5

	node5 := NewSAItem([]bool{true, true, true})
	node5.Score = 2.5

	insertSequence := []struct {
		item       *SAItem
		best       *SAItem
		worst      *SAItem
		length     int
		worstScore float64
		bestScore  float64
	}{
		{
			item:       node1,
			best:       node1,
			worst:      node1,
			length:     1,
			worstScore: 1.0,
			bestScore:  1.0,
		},
		{
			item:       node2,
			best:       node2,
			worst:      node1,
			length:     2,
			worstScore: 1.0,
			bestScore:  2.0,
		},
		{
			item:       node3,
			best:       node2,
			worst:      node3,
			length:     3,
			worstScore: 0.9,
			bestScore:  2.0,
		},
		{
			item:       node4,
			best:       node2,
			worst:      node1,
			length:     3,
			worstScore: 1.0,
			bestScore:  2.0,
		},
		{
			item:       node5,
			best:       node5,
			worst:      node3,
			length:     3,
			worstScore: 1.5,
			bestScore:  2.5,
		},
	}

	for i, ins := range insertSequence {
		score.Insert(ins.item)

		if len(score.Items) != ins.length {
			t.Errorf("Length different from %d after %d insert(s)", ins.length, i+1)
		}

		if score.BestItem != ins.best {
			t.Errorf("BestItem wrong after %d insert(s)", i+1)
		}

		if math.Abs(score.BestItem.Score-ins.bestScore) > 1e-10 {
			t.Errorf("unexpected best score. Expected %f got %f", ins.bestScore, score.BestItem.Score)
		}

		if score.WorstItem != ins.worst {
			t.Errorf("WorstItem wrong after %d insert(s)", i+1)
		}

		if math.Abs(score.WorstItem.Score-ins.worstScore) > 1e-10 {
			t.Errorf("unexpected worst score. Expected %f got %f", ins.worstScore, score.WorstItem.Score)
		}
	}
}
