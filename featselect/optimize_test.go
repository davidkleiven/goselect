package featselect

import (
	"testing"
)

func TestNumFeatures(t *testing.T) {
	for _, test := range []struct {
		model []bool
		num   int
	}{
		{
			model: []bool{false, false, false},
			num:   0,
		},
		{
			model: []bool{false, false, true},
			num:   1,
		},
	} {
		num := numFeatures(test.model)

		if num != test.num {
			t.Errorf("Expect: %v, Got: %v", num, test.num)
		}
	}
}

func TestGCS(t *testing.T) {
	for _, test := range []struct {
		model []bool
		gcs   []bool
		start int
	}{
		{
			model: []bool{false, false, false},
			start: 1,
			gcs:   []bool{false, true, true},
		},
		{
			model: []bool{false, false, false},
			start: 0,
			gcs:   []bool{true, true, true},
		},
		{
			model: []bool{false, false, false},
			start: 2,
			gcs:   []bool{false, false, true},
		},
		{
			model: []bool{false, false, false},
			start: 3,
			gcs:   []bool{false, false, false},
		},
	} {
		gcs := gcs(test.model, test.start)
		for i := 0; i < len(test.model); i++ {
			if gcs[i] != test.gcs[i] {
				t.Errorf("GCS failed. Expected: %v, got %v", test.gcs, gcs)
			}
		}
	}
}

func TestLCS(t *testing.T) {
	for _, test := range []struct {
		model []bool
		lcs   []bool
		start int
	}{
		{
			model: []bool{false, false, false},
			start: 1,
			lcs:   []bool{false, false, false},
		},
		{
			model: []bool{false, false, false},
			start: 0,
			lcs:   []bool{false, false, false},
		},
		{
			model: []bool{true, true, false},
			start: 2,
			lcs:   []bool{true, true, false},
		},
		{
			model: []bool{true, false, false},
			start: 3,
			lcs:   []bool{true, false, false},
		},
	} {
		lcs := lcs(test.model, test.start)
		for i := 0; i < len(test.model); i++ {
			if lcs[i] != test.lcs[i] {
				t.Errorf("LCS failed. Expected: %v, got %v", test.lcs, lcs)
			}
		}
	}
}
