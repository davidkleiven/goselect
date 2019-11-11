package main

import (
	"encoding/json"
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

func main() {
	args := featselect.ParseCommandLineArgs(os.Args[1:])
	dset := featselect.ReadCSV(args.Csvfile, args.TargetCol)

	fmt.Println("First few items of target column\n%v\n", dset.Y[:10])

	var wg sync.WaitGroup
	var progress featselect.SearchProgress
	highscore := featselect.NewHighscore(1000)

	wg.Add(1)
	go func() {
		defer wg.Done()
		featselect.SelectModel(dset.X, dset.Y, highscore, &progress)
	}()

	c := time.Tick(60 * time.Second)
	fmt.Printf("Saving highscore list periodically to " + args.Outfile + "\n")
	for {
		select {
		case <-c:
			score, numChecked, log2Pruned := progress.Get()
			fmt.Printf("%v: Score: %f, Num. checked: %d, Log2 pruned: %d\n", time.Now(), score, numChecked, log2Pruned)
			saveHighscoreList(args.Outfile, highscore)
		}
	}
	wg.Wait()
}
