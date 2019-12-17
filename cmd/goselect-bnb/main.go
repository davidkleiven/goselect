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

func main() {
	csvfile := flag.String("csvfile", "", "CSV file containing the data")
	targetCol := flag.Int("target", -1, "Column in the CSV file with the target data. If negative it wraps around.")
	outfile := flag.String("out", "", "Outfile for the search")
	cutoff := flag.Float64("cutoff", 0.0, "Cutoff that will be added to the cost function when when branches are pruned")
	maxQueueSize := flag.Int("maxQueueSize", 10000000, "Maximum size of the queue. If this limit is reached, subtrees will be removed.")

	flag.Parse()

	findOptimalSolution(*csvfile, *targetCol, *cutoff, *outfile, *maxQueueSize)
}
