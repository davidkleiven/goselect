package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"time"

	"github.com/davidkleiven/goselect/featselect"
)

func saveHighscoreList(fname string, h *featselect.Highscore) {
	file, _ := os.Open(fname)
	defer file.Close()

	highscoreJSON, _ := json.Marshal(h)
	ioutil.WriteFile(fname, highscoreJSON, 0644)
}

func setSearchFinished(finished chan int) {
	finished <- 0
}

func findOptimalSolution(csvfile string, targetCol int, cutoff float64, outfile string) {
	dset := featselect.ReadCSV(csvfile, targetCol)
	fmt.Printf("First few items of target column\n%v\n", dset.Y[:10])

	var wg sync.WaitGroup
	var progress featselect.SearchProgress
	searchFinished := make(chan int)
	highscore := featselect.NewHighscore(1000)

	// Get a good initial model from SA
	fmt.Printf("Searching for good initial model with SA\n")
	res := featselect.SelectModelSA(dset.X, dset.Y, 100, featselect.Aicc)
	_, nFeat := dset.X.Dims()

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer setSearchFinished(searchFinished)
		featselect.SelectModel(dset.X, dset.Y, highscore, &progress, cutoff, featselect.Selected2Model(res.Selected, nFeat))
	}()

	c := time.Tick(60 * time.Second)
	fmt.Printf("Saving highscore list periodically to " + outfile + "\n")

timeloop:
	for {
		select {
		case <-c:
			score, numChecked, log2Pruned := progress.Get()
			fmt.Printf("%v: Score: %f, Num. checked: %d, Log2 pruned: %f\n", time.Now().Format(time.RFC3339), score, numChecked, log2Pruned)
			saveHighscoreList(outfile, highscore)
		case <-searchFinished:
			break timeloop
		}
	}
	wg.Wait()
	fmt.Printf("Selection finished\n")
}

func standardizeColumns(infile string, outfile string) {
	dset := featselect.ReadCSV(infile, 0)
	featselect.NormalizeCols(dset.X)
	featselect.NormalizeArray(dset.Y)
	dset.Save(outfile)
	fmt.Printf("Normalised features written to " + outfile)
}

func main() {
	searchCommand := flag.NewFlagSet("search", flag.ExitOnError)
	stdColCommand := flag.NewFlagSet("std", flag.ExitOnError)

	// Optimal solution search
	csvfile := searchCommand.String("csvfile", "", "csv file with data")
	targetCol := searchCommand.Int("target", 0, "column with the y-values where the remaining features should predict")
	outfile := searchCommand.String("out", "", "json file used to store the result")
	cutoff := searchCommand.Float64("cutofff", 0.0, "cutoff added to the cost function")

	// Standardiize stdColCommand
	stdIn := stdColCommand.String("csvfile", "", "csv file with data")
	stdOut := stdColCommand.String("out", "", "outfile where the standardized features are placed")

	if len(os.Args) < 2 {
		fmt.Printf("No subcommand specifyied\n")
		return
	}

	switch os.Args[1] {
	case "search":
		searchCommand.Parse(os.Args[2:])
	case "std":
		stdColCommand.Parse(os.Args[2:])
	default:
		flag.PrintDefaults()
		fmt.Printf("No subcommands specified: search,std\n")
		return
	}

	if searchCommand.Parsed() {
		findOptimalSolution(*csvfile, *targetCol, *cutoff, *outfile)
	} else if stdColCommand.Parsed() {
		standardizeColumns(*stdIn, *stdOut)
	}
}
