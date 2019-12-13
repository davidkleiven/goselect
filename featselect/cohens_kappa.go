package featselect

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// CohensKappaTarget is a interface that can be used to sample the cohens kappa
type CohensKappaTarget interface {
	// GetX returns the full design matrix
	GetX() mat.Matrix

	// GetSelection returns the selected features when using the data points given by indices
	GetSelection(indices []int) []int

	// HyperParameters returns the current hyper parameters
	HyperParameters() map[string]float64

	// StringRep returns a string represenation that is used for logging
	StringRep() string
}

// GetCohensKappa calculates the expected kappa value as by random partitioning
func GetCohensKappa(numSamples int, target CohensKappaTarget) float64 {
	kappa := 0.0
	X := target.GetX()
	nr, nc := X.Dims()

	indices := make([]int, nr)
	for i := 0; i < nr; i++ {
		indices[i] = i
	}
	for i := 0; i < numSamples; i++ {
		rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		n := nr / 2
		selection1 := target.GetSelection(indices[:n])
		selection2 := target.GetSelection(indices[n:])
		kappa += CohenKappa(selection1, selection2, nc)
	}
	return kappa / float64(numSamples)
}

// CohensKappaWorkload is a struct used to calculate cohens kapps
type CohensKappaWorkload struct {
	numSamples int
	target     CohensKappaTarget
	kappa      float64
}

// CohensKappaWorker reads from a channel and pass the result to a new channel
func CohensKappaWorker(work <-chan CohensKappaWorkload, res chan<- CohensKappaWorkload) {
	for w := range work {
		w.kappa = GetCohensKappa(w.numSamples, w.target)
		res <- w
	}
}

// CalculateCohenSequence calculates cohens kappa for a collection of values
func CalculateCohenSequence(numSamples int, targets []CohensKappaTarget) map[string]float64 {
	numWorkers := 16
	workChannel := make(chan CohensKappaWorkload)
	resChannel := make(chan CohensKappaWorkload)
	nextTarget := 0
	for i := 0; i < numWorkers; i++ {
		go CohensKappaWorker(workChannel, resChannel)

		if nextTarget < len(targets) {
			work := CohensKappaWorkload{
				numSamples: numSamples,
				target:     targets[nextTarget],
			}
			nextTarget++
			workChannel <- work
		}
	}

	numReceives := 0
	bestKappa := 0.0
	var bestHyper map[string]float64

	allKappas := make([]float64, len(targets))
	allStrRep := make([]string, len(targets))

	for numReceives < len(targets) {
		res := <-resChannel

		allKappas[numReceives] = res.kappa
		allStrRep[numReceives] = res.target.StringRep()

		numReceives++
		hyper := res.target.HyperParameters()
		fmt.Printf(res.target.StringRep()+" kappa: %3.3f\n", res.kappa)

		if res.kappa > bestKappa {
			bestKappa = res.kappa
			bestHyper = hyper
		}

		if nextTarget < len(targets) {
			work := CohensKappaWorkload{
				numSamples: numSamples,
				target:     targets[nextTarget],
			}
			nextTarget++
			workChannel <- work
		}
	}

	srt := Argsort(allKappas)
	fmt.Printf("-------------------------------------------------------------------\n")
	fmt.Printf("                    COHEN'S KAPPA SUMMARY                          \n")
	fmt.Printf("-------------------------------------------------------------------\n")
	for i := range allKappas {
		fmt.Printf("%54s Kappa: %3.3f\n", allStrRep[srt[i]], allKappas[srt[i]])
	}
	fmt.Printf("-------------------------------------------------------------------\n")
	return bestHyper
}
