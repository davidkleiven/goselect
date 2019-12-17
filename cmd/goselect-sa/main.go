package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/davidkleiven/goselect/featselect"
)

func saSearch(csvfile string, targetCol int, out string, sweeps int) {
	dset := featselect.ReadCSV(csvfile, targetCol)
	res := featselect.SelectModelSA(dset.X, dset.Y, sweeps, featselect.Aicc)
	file, _ := os.Open(out)
	defer file.Close()

	highscoreJSON, _ := json.Marshal(res.Scores)
	ioutil.WriteFile(out, highscoreJSON, 0644)

	fmt.Printf("SA highscore list written to %s", out)
}

func main() {
	saCsv := flag.String("csvfile", "", "CSV file with the data")
	saTarget := flag.Int("target", -1, "Column where the target values are placed. If negative it is counted from the last column.")
	saOut := flag.String("out", "saSearch.json", "JSON file where the final result will be stored")
	saSweeps := flag.Int("sweeps", 100, "Number of sweeps")

	flag.Parse()
	saSearch(*saCsv, *saTarget, *saOut, *saSweeps)
}
