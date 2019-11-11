package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/davidkleiven/goselect/featselect"
)

func main() {
	args := featselect.ParseCommandLineArgs(os.Args[1:])
	dset := featselect.ReadCSV(args.Csvfile, args.TargetCol)

	var wg sync.WaitGroup
	var progress featselect.SearchProgress
	highscore := featselect.NewHighscore(100)

	wg.Add(1)
	go func() {
		defer wg.Done()
		featselect.SelectModel(dset.X, dset.Y, highscore, &progress)
	}()

	c := time.Tick(60 * time.Second)
	for {
		select {
		case <-c:
			score, numChecked, log2Pruned := progress.Get()
			fmt.Printf("%v: Score: %f, Num. checked: %d, Log2 pruned: %d", time.Now(), score, numChecked, log2Pruned)
		}
	}

	wg.Wait()
	file, _ := os.Open(args.Outfile)
	defer file.Close()

	json.NewEncoder(file).Encode(highscore)
	fmt.Printf("Best models written to " + args.Outfile)

}
