package featselect

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// SARes holds the solution of SA search
type SARes struct {
	Selected []int
	Coeff    []float64
	Scores   *SAScore
}

// SelectModelSA uses simmulated annealing to select the model
func SelectModelSA(X mat.Matrix, y []float64, nSweeps int, cost crit) *SARes {
	var res SARes
	res.Scores = NewSAScore(10)

	nr, nc := X.Dims()
	current := make([]bool, nc)
	currentScore := math.MaxFloat64
	current[0] = true
	coeff := make([]float64, nc)
	temp := 500.0

	numAccept := 0
	numSteps := 0
	hasReached50 := false

	for {
		index := rand.Intn(nc)
		current[index] = !current[index]
		N := NumFeatures(current)
		if N == 0 {
			current[index] = !current[index]
			N = 1
		} else if N >= nr/2 {
			current[index] = !current[index]
			continue
		}

		design := GetDesignMatrix(current, X)
		coeffTemp := Fit(design, y)
		score := cost(N, len(y), math.Max(Rss(design, coeffTemp, y), RssTol))

		accept := score < currentScore || math.Exp(-(score-currentScore)/temp) > rand.Float64()

		if accept {
			numAccept++
			currentScore = score
			copy(coeff, coeffTemp)
			item := NewSAItem(current)
			item.Score = -score // Change sign since the highscore list keeps only the larges
			copy(item.Coeff, coeffTemp)
			res.Scores.Insert(item)
		} else {
			current[index] = !current[index]
		}
		numSteps++

		// Check if we have 50% acceptance once per sweep
		if numSteps%nc == 0 {
			hasReached50 = hasReached50 || float64(numAccept)/float64(numSteps) > 0.5

			if !hasReached50 {
				temp *= 2.0
			}
		}

		if numSteps >= nc*nSweeps {
			accRate := float64(numAccept) / float64(numSteps)

			if numAccept == 0 && hasReached50 {
				break
			}

			numAccept = 0
			numSteps = 0

			if hasReached50 {
				temp *= 0.5
			}

			if temp < 1e-12 {
				break
			}
			fmt.Printf("Current temperature: %f. Acc.rate: %f. Current score: %.2e\n", temp, accRate, currentScore)
		}
	}

	res.Coeff = coeff[:NumFeatures(current)]
	res.Selected = make([]int, NumFeatures(current))
	counter := 0
	for i, v := range current {
		if v {
			res.Selected[counter] = i
			counter++
		}
	}
	return &res
}
