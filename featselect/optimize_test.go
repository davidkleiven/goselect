package featselect

import (
	"math"
	"sort"
	"testing"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
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
		num := NumFeatures(test.model)

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
		gcs := Gcs(test.model, test.start)
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
		lcs := Lcs(test.model, test.start)
		for i := 0; i < len(test.model); i++ {
			if lcs[i] != test.lcs[i] {
				t.Errorf("LCS failed. Expected: %v, got %v", test.lcs, lcs)
			}
		}
	}
}

func TestSelectModel(t *testing.T) {
	X := mat.NewDense(7, 5, []float64{1.0, 0.0, 0.0, 0.0, 0.0,
		1.0, 1.0, 1.0, 1.0, 1.0,
		1.0, 2.0, 4.0, 8.0, 16.0,
		1.0, 3.0, 9.0, 15.0, 40.0,
		1.0, 4.0, 9.0, 30.0, 6.0,
		1.0, 2.0, 3.0, 6.0, 12.0,
		1.0, -2.0, 5.0, 4.0, 3.0})

	y := []float64{1.0, 2.0, 5.0, 7.0, 10.0, 8.0, 15.0}
	bAndB := SelectModel(X, y)
	brute := BruteForceSelect(X, y)

	if brute.Len() != (1<<5)-1 {
		t.Errorf("SelectModel: Brute force has not explored all models. Expected: %v. Got: %v", (1<<5)-1, brute.Len())
	}

	if math.Abs(bAndB.BestScore()-brute.BestScore()) > 1e-10 {
		t.Errorf("SelectModel: BestScore differ. Brute force: %v, BandB: %v", brute.BestScore(), bAndB.BestScore())
	}

	// Make sure that we don't have duplicates in the highscore list
	for item := bAndB.items.Front(); item != nil; item = item.Next() {
		for item2 := bAndB.items.Front(); item2 != item; item2 = item2.Next() {
			if floats.EqualApprox(item.Value.(*Node).Coeff, item2.Value.(*Node).Coeff, 1e-10) {
				t.Errorf("SelectModel: Duplicates in highscore list")
			}
		}
	}
}

func TestBruteForceSelect(t *testing.T) {
	for testnum, test := range []struct {
		X      *mat.Dense
		y      []float64
		models [][]bool
	}{
		{
			X:      mat.NewDense(2, 2, []float64{1.0, 2.0, -1.0, 3.0}),
			y:      []float64{1.0, 0.0},
			models: [][]bool{{true, false}, {false, true}, {true, true}},
		},
		{
			X: mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
			y: []float64{1.0, 2.0},
			models: [][]bool{{true, false, false}, {false, true, false},
				{false, false, true}, {true, false, true},
				{false, true, true}, {true, true, true},
				{true, true, false}},
		},
	} {
		scores := BruteForceSelect(test.X, test.y).Scores()
		expScores := make([]float64, len(test.models))
		for i, model := range test.models {
			design := GetDesignMatrix(model, test.X)
			coeff := Fit(design, test.y)
			rss := Rss(design, coeff, test.y)
			expScores[i] = -Aicc(NumFeatures(model), len(test.y), rss)
		}
		sort.Float64s(expScores)
		reverse(expScores)

		if !floats.EqualApprox(scores, expScores, 1e-10) {
			t.Errorf("Test #%v failed: Expected %v. Got %v", testnum, expScores, scores)
		}

	}
}

func reverse(a []float64) {
	for left, right := 0, len(a)-1; left < right; left, right = left+1, right-1 {
		a[left], a[right] = a[right], a[left]
	}
}
