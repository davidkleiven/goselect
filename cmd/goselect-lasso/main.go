package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/davidkleiven/goselect/featselect"
	"gonum.org/v1/gonum/mat"
)

func lassoFit(csvfile string, targetCol int, out string, lambMin float64, lambMax float64, num int, lassoType string, covType string, tol float64) {
	dset := featselect.ReadCSV(csvfile, targetCol)
	y := make([]float64, len(dset.Y))
	copy(y, dset.Y)
	normDset := featselect.NewNormalizedData(mat.DenseCopyOf(dset.X), y)

	larspath := []*featselect.LassoLarsNode{}
	if lassoType == "lars" {
		var estimator featselect.MorsePenroseCD
		larspath = featselect.LassoLars(normDset, lambMin, &estimator)
	} else if lassoType == "cd" {
		var cov featselect.CovMat
		var corr featselect.PureLasso
		if covType == "empirical" {
			var emp featselect.Empirical
			cov = &emp
		} else if covType == "identity" {
			var id featselect.Identity
			cov = &id
		} else if covType == "threshold" {
			cov = featselect.NewSparseThreshold(normDset.X)
		} else {
			fmt.Printf("Unknown covariance type %s\n", covType)
			return
		}
		lambs := featselect.Logspace(lambMin, lambMax, num)
		larspath = featselect.LassoCrdDescPath(normDset, cov, lambs, 100000, tol, &corr)
	}

	featselect.Path2Unnormalized(normDset, larspath)
	fmt.Printf("LASSO-LARS solution finished. Number of nodes in path %d.\n", len(larspath))

	var path featselect.LassoLarsPath
	path.Dset = dset
	path.LassoLarsNodes = larspath

	aicc := path.GetCriteria(featselect.Aicc)
	bic := path.GetCriteria(featselect.Bic)
	path.Aicc = aicc
	path.Bic = bic
	featselect.PrintHighscore(&path, aicc, bic, 20)

	js, err := json.Marshal(path)

	if err != nil {
		fmt.Printf("Error: %s", err)
		return
	}

	file, _ := os.Create(out)
	defer file.Close()

	ioutil.WriteFile(out, js, 0644)
	fmt.Printf("LASSO-LARS results written to %s\n", out)
}

func main() {
	lassoCsv := flag.String("csvfile", "", "CSV file with data")
	lassoTarget := flag.Int("target", -1, "Target column, if negative the column is counted from the end")
	lassoOut := flag.String("out", "", "JSON file where the output will be stored")
	lambMin := flag.Float64("lambMin", 1e-10, "Minimum value of the regularization parameter")
	lambMax := flag.Float64("lambMax", 1.0, "Maximum value of the regularization parameter")
	numLamb := flag.Int("num", 50, "Number of regularization (only with coordinate descent)")
	lassoType := flag.String("type", "cd", "Algorithm lars|cd")
	lassoCov := flag.String("cov", "empirical", "Estimator for the covariane matrix")
	lassoTol := flag.Float64("tol", 1e-4, "Tolerance in LASSO coordinate descent")

	flag.Parse()

	if *lassoCsv == "" {
		fmt.Printf("No CSV file given!\n")
		return
	}
	lassoFit(*lassoCsv, *lassoTarget, *lassoOut, *lambMin, *lambMax, *numLamb, *lassoType, *lassoCov, *lassoTol)
}
