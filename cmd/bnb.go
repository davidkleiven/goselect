/*
Copyright © 2020 NAME HERE <EMAIL ADDRESS>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package cmd

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sync"
	"time"

	"github.com/davidkleiven/goselect/featselect"
	"github.com/spf13/cobra"
)

// bnbCmd represents the bnb command
var bnbCmd = &cobra.Command{
	Use:   "bnb",
	Short: "Optimized models via branch and bound",
	Long: `Minimizes AICc by branch and bound.
	
Example:
goselect bnb -csv mydataset.csv -target -1 -out result.json
	`,
	Run: func(cmd *cobra.Command, args []string) {
		csvfile, err := cmd.Flags().GetString("csv")
		if err != nil {
			log.Print(err)
			return
		}

		target, _ := cmd.Flags().GetInt("target")
		cutoff, _ := cmd.Flags().GetFloat64("cutoff")
		outfile, _ := cmd.Flags().GetString("out")
		maxQueue, _ := cmd.Flags().GetInt("maxqueue")

		findOptimalSolution(csvfile, target, cutoff, outfile, maxQueue)
	},
}

func init() {
	rootCmd.AddCommand(bnbCmd)

	bnbCmd.Flags().String("csv", "", "CSV file containing the data")
	bnbCmd.Flags().Int("target", -1, "Column in the CSV file with the target data. If negative it wraps around.")
	bnbCmd.Flags().String("out", "bnbSearch.json", "Outfile for the search")
	bnbCmd.Flags().Float64("cutoff", 0.0, "Cutoff that will be added to the cost function when when branches are pruned")
	bnbCmd.Flags().Int("maxqueue", 10000000, "Maximum size of the queue. If this limit is reached, subtrees will be removed. If you run out of memory, this number should be lowered.")
}

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
	highscore := featselect.NewHighscore(10)

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
