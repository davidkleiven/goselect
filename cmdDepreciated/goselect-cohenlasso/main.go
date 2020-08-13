package main

import (
	"flag"

	"github.com/davidkleiven/goselect/featselect"
)

func cohensKappaPureLasso(csvfile string, lambMin float64, lambMax float64, numLamb int, target int, numSamples int, tol float64) {
	dset := featselect.ReadCSV(csvfile, target)
	normDset := featselect.NewNormalizedData(dset.X, dset.Y)

	targets := make([]featselect.CohensKappaTarget, numLamb)
	lambs := featselect.Logspace(lambMin, lambMax, numLamb)
	var cov featselect.Empirical
	var corr featselect.PureLasso
	for i := range targets {
		cohenTarget := featselect.NewPureLassoCohen()
		cohenTarget.Dset = normDset
		cohenTarget.Lamb = lambs[len(lambs)-i-1]
		cohenTarget.Cov = &cov
		cohenTarget.MaxIter = 100000
		cohenTarget.Tol = tol
		cohenTarget.Correction = &corr
		targets[i] = cohenTarget
	}
	featselect.CalculateCohenSequence(numSamples, targets)
}

func main() {
	cohenLassoCsv := flag.String("csvfile", "", "CSV file with data")
	cohenTarget := flag.Int("target", -1, "Target column. If negative, it is counted from the end")
	cohenNumSamp := flag.Int("numSamp", 100, "Number of samples to calculate Cohens kappa")
	cohenLambMin := flag.Float64("lambMin", 1e-5, "Minimum value for the regularization parameter")
	cohenLambMax := flag.Float64("lambMax", 1.0, "Maximum value for the regularization parameter")
	cohenNumLam := flag.Int("numLamb", 20, "Number of regularization parameters")
	cohenTol := flag.Float64("tol", 1e-4, "Tolerance in coordinate descent lasso")
	flag.Parse()

	cohensKappaPureLasso(*cohenLassoCsv, *cohenLambMin, *cohenLambMax, *cohenNumLam, *cohenTarget, *cohenNumSamp, *cohenTol)
}
