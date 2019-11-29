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

func findOptimalSolution(csvfile string, targetCol int, cutoff float64, outfile string, maxQueueSize int) {
	dset := featselect.ReadCSV(csvfile, targetCol)
	params := featselect.NewSelectModelOptParams()
	params.Cutoff = cutoff
	params.MaxQueueSize = maxQueueSize

	num := 10
	if len(dset.Y) < num {
		num = len(dset.Y)
	}
	fmt.Printf("First few items of target column\n%v\n", dset.Y[:num])

	var wg sync.WaitGroup
	var progress featselect.SearchProgress
	searchFinished := make(chan int)
	highscore := featselect.NewHighscore(1000)

	// Get a good initial model from SA
	fmt.Printf("Searching for good initial model with SA\n")
	res := featselect.SelectModelSA(dset.X, dset.Y, 100, featselect.Aicc)
	_, nFeat := dset.X.Dims()

	params.RootModel = featselect.Selected2Model(res.Selected, nFeat)
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer setSearchFinished(searchFinished)
		featselect.SelectModel(dset.X, dset.Y, highscore, &progress, params)
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
	saveHighscoreList(outfile, highscore)
	wg.Wait()
	fmt.Printf("Selection finished\n")
}

func standardizeColumns(infile string, outfile string) {
	dset := featselect.ReadCSV(infile, 0)
	featselect.NormalizeCols(dset.X)
	featselect.NormalizeArray(dset.Y)
	dset.Save(outfile)
	fmt.Printf("Normalised features written to " + outfile + "\n")
}

func estimateMaxQueueBuffer(memory int, maxFeat int) {
	model := make([]bool, maxFeat)
	for i := 0; i < len(model); i++ {
		model[i] = true
	}

	n := featselect.NewNode(0, model)
	size := n.EstimateMemory()

	numInQueue := memory * 1000000000 / size
	fmt.Printf("Buffer size for %d GB: %d\n", memory, numInQueue)
}

func main() {
	searchCommand := flag.NewFlagSet("search", flag.ExitOnError)
	stdColCommand := flag.NewFlagSet("std", flag.ExitOnError)
	memEstCommand := flag.NewFlagSet("bufferSize", flag.ExitOnError)

	// Optimal solution search
	csvfile := searchCommand.String("csvfile", "", "csv file with data")
	targetCol := searchCommand.Int("target", 0, "column with the y-values where the remaining features should predict")
	outfile := searchCommand.String("out", "", "json file used to store the result")
	cutoff := searchCommand.Float64("cutoff", 0.0, "cutoff added to the cost function")
	maxQueueSize := searchCommand.Int("maxQueueSize", 10000000, "maximum number of nodes in the queue")

	// Standardiize stdColCommand
	stdIn := stdColCommand.String("csvfile", "", "csv file with data")
	stdOut := stdColCommand.String("out", "", "outfile where the standardized features are placed")

	// Buffersize vs memory
	memUse := memEstCommand.Int("mem", 0, "max memory to use for the queue")
	maxFeat := memEstCommand.Int("nfeat", 1, "maximum number of features")

	subcmds := "search, std, bufferSize"
	if len(os.Args) < 2 {
		fmt.Printf("No subcommand specifyied. Has to be one of %s\n", subcmds)
		return
	}

	switch os.Args[1] {
	case "search":
		searchCommand.Parse(os.Args[2:])
	case "std":
		stdColCommand.Parse(os.Args[2:])
	case "bufferSize":
		memEstCommand.Parse(os.Args[2:])
	default:
		flag.PrintDefaults()
		fmt.Printf("No subcommands specified: %s\n", subcmds)
		return
	}

	if searchCommand.Parsed() {
		findOptimalSolution(*csvfile, *targetCol, *cutoff, *outfile, *maxQueueSize)
	} else if stdColCommand.Parsed() {
		standardizeColumns(*stdIn, *stdOut)
	} else if memEstCommand.Parsed() {
		estimateMaxQueueBuffer(*memUse, *maxFeat)
	}
}
