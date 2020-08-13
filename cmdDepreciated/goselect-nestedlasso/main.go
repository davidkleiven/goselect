package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"

	"github.com/davidkleiven/goselect/featselect"
)

func nestedLasso(csvfile string, targetCol int, out string, lambMin float64, keep float64) {
	dset := featselect.ReadCSV(csvfile, targetCol)
	var estimator featselect.MorsePenroseCD
	res := featselect.NestedLasso(dset, lambMin, keep, &estimator)
	js, err := json.Marshal(res)

	if err != nil {
		fmt.Printf("Error %s", err)
		return
	}

	ioutil.WriteFile(out, js, 0644)
	fmt.Printf("Nested LASSO-LARS results written to %s\n", out)
}

func main() {
	nestedCsv := flag.String("csvfile", "", "CSV file with data")
	nestedTarget := flag.Int("target", -1, "Column in the CSV file with the data. Negative values wraps around")
	nestedOut := flag.String("out", "nesteLasso.json", "Outfile with the output")
	nestedLamb := flag.Float64("lambMin", 1e-10, "Minimum value for the lambda")
	nestedKeep := flag.Float64("keep", 0.8, "Fraction of features that are kept")

	flag.Parse()
	nestedLasso(*nestedCsv, *nestedTarget, *nestedOut, *nestedLamb, *nestedKeep)
}
