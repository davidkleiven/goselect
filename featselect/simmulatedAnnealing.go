package featselect

import (
	"fmt"
	"math"
	"math/rand"
)

// SARes holds the solution of SA search
type SARes struct {
	Selected []int
	Coeff    []float64
}

// SelectModelSA uses simmulated annealing to select the model
func SelectModelSA(X DesignMatrix, y []float64, nSweeps int, cost crit) *SARes {
	var res SARes
	_, nc := X.Dims()
	current := make([]bool, nc)
	currentScore := math.MaxFloat64
	current[0] = true
	coeff := make([]float64, nc)
	temp := 1000.0

	numAccept := 0
	numSteps := 0
	hasReached50 := false

	seedSource := rand.NewSource(42)
	seedRng := rand.New(seedSource)
	rngSource := rand.NewSource(int64(seedRng.Intn(math.MaxInt64)))
	rng := rand.New(rngSource)

	for {
		index := rng.Intn(nc)
		current[index] = !current[index]
		N := NumFeatures(current)
		if N == 0 {
			current[index] = !current[index]
			N = 1
		}

		design := GetDesignMatrix(current, X)
		coeffTemp := Fit(design, y)
		score := cost(N, len(y), math.Max(Rss(design, coeffTemp, y), RssTol))

		accept := score < currentScore || math.Exp(-(score-currentScore)/temp) > rng.Float64()

		if accept {
			numAccept++
			currentScore = score
			copy(coeff, coeffTemp)
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
			fmt.Printf("Current temperature: %f. Acc.rate: %f\n", temp, accRate)
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
