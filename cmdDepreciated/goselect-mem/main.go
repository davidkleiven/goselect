package main

import (
	"flag"
	"fmt"

	"github.com/davidkleiven/goselect/featselect"
)

func main() {
	memUse := flag.Int("mem", 0, "Maximum memory to use for the B&B queue")
	maxFeat := flag.Int("nfeat", 1, "Total number of features")
	flag.Parse()

	model := make([]bool, *maxFeat)
	for i := 0; i < len(model); i++ {
		model[i] = true
	}

	n := featselect.NewNode(0, model)
	size := n.EstimateMemory()

	memory := *memUse
	numInQueue := memory * 1000000000 / size
	fmt.Printf("Buffer size for %d GB: %d\n", memory, numInQueue)
}
