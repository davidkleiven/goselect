package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sync"
	"time"

	"github.com/davidkleiven/goselect/featselect"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot/vg"
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

func lassoFit(csvfile string, targetCol int, out string, lambMin float64) {
	dset := featselect.ReadCSV(csvfile, targetCol)
	y := make([]float64, len(dset.Y))
	copy(y, dset.Y)
	normDset := featselect.NewNormalizedData(mat.DenseCopyOf(dset.X), y)
	larspath := featselect.LassoLars(normDset, lambMin)
	featselect.Path2Unnormalized(normDset, larspath)
	fmt.Printf("LASSO-LARS solution finished. Number of nodes in path %d.\n", len(larspath))

	var path featselect.LassoLarsPath
	path.Dset = dset
	path.LassoLarsNodes = larspath
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

func analyseLasso(jsonfile string, prefix string, ext string, coeffRng *featselect.AxisRange) {
	path := featselect.LassoLarsPathFromJSON(jsonfile)
	entrance := path.PlotEntranceTimes()
	fname := prefix + "_entrance." + ext
	w := 4 * vg.Inch
	h := 4 * vg.Inch
	entrance.Save(w, h, fname)
	fmt.Printf("Entrance times written to %s\n", fname)

	quality := path.PlotQualityScores()
	fname = prefix + "_quality." + ext
	quality.Save(w, h, fname)
	fmt.Printf("Quality score written to %s\n", fname)

	dev := path.PlotDeviations()
	fname = prefix + "_dev." + ext
	dev.Save(w, h, fname)
	fmt.Printf("Dev. score written to %s\n", fname)

	coeff := path.PlotPath(coeffRng)
	fname = prefix + "_path." + ext
	coeff.Save(w, h, fname)
	fmt.Printf("LASSO-LARS path written to %s\n", fname)
}

func aicAverageLassoPath(jsonfile string, out string) {
	path := featselect.LassoLarsPathFromJSON(jsonfile)
	nr, nFeat := path.Dset.X.Dims()
	aicc := make([]float64, len(path.LassoLarsNodes))

	for i, v := range path.LassoLarsNodes {
		coeff := featselect.FullCoeffVector(nFeat, v.Selection, v.Coeff)
		rss := featselect.Rss(path.Dset.X, coeff, path.Dset.Y)
		aicc[i] = featselect.Aicc(len(v.Selection), nr, rss)
	}

	w := featselect.WeightsFromAIC(aicc)
	sp := featselect.LassoNodesSlice2SparsCoeff(path.LassoLarsNodes)
	avgCoeff := featselect.WeightedAveragedCoeff(nFeat, w, sp)

	// Export result
	dict := make(map[string]float64)
	numFeatInAvg := 0
	for i := range avgCoeff {
		if math.Abs(avgCoeff[i]) > 1e-16 {
			dict[path.Dset.GetFeatName(i)] = avgCoeff[i]
			numFeatInAvg++
		}
	}

	js, err := json.Marshal(dict)

	if err != nil {
		fmt.Printf("Error: %s", err)
		return
	}

	file, _ := os.Create(out)
	defer file.Close()

	ioutil.WriteFile(out, js, 0644)
	fmt.Printf("Num. features in AIC averaged model %d\n", numFeatInAvg)
	fmt.Printf("AIC averaged coefficients written to %s\n", out)
}

func main() {
	searchCommand := flag.NewFlagSet("search", flag.ExitOnError)
	stdColCommand := flag.NewFlagSet("std", flag.ExitOnError)
	memEstCommand := flag.NewFlagSet("bufferSize", flag.ExitOnError)
	lassoCommand := flag.NewFlagSet("lasso", flag.ExitOnError)
	plotLassoCommand := flag.NewFlagSet("plotlasso", flag.ExitOnError)
	lassoAicAverage := flag.NewFlagSet("lassoavg", flag.ExitOnError)

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

	// Lasso command
	lassoCsv := lassoCommand.String("csvfile", "", "csv file with data")
	lassoTarget := lassoCommand.Int("target", 0, "column with the y-values that the remaining features should predict")
	lassoOut := lassoCommand.String("out", "lassoOut.json", "JSON file where the result will be stored")
	lambMin := lassoCommand.Float64("lambMin", 1e-10, "minimum value of the regularization parameter")

	// Plot lasso command
	lassoPathJSON := plotLassoCommand.String("json", "", "JSON file with the Lasso-Lars path")
	lassoPathOutPrefix := plotLassoCommand.String("prefix", "lassolars", "Prefix that will be used for all figures generated")
	lassoPathType := plotLassoCommand.String("ext", "png", "Type to use when save figures has to be png|pdf|svg|jpeg|tiff|")
	lassoPathCoeffMin := plotLassoCommand.Float64("coeffmin", 0.0, "ymin on path plot")
	lassoPathCoeffMax := plotLassoCommand.Float64("coeffmax", 0.0, "ymax on path plot. If equal to coeffmin the axis will be autoscaled.")

	// AIC averaged lasso
	lassoAicPath := lassoAicAverage.String("json", "", "JSON file with the Lasso-Lars path")
	lassoAicOut := lassoAicAverage.String("out", "lasso_aic_avg.json", "Text file where the averaged coefficients will be placed")

	subcmds := "search, std, bufferSize, lasso, plotlasso, lassoavg"
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
	case "lasso":
		lassoCommand.Parse(os.Args[2:])
	case "plotlasso":
		plotLassoCommand.Parse(os.Args[2:])
	case "lassoavg":
		lassoAicAverage.Parse(os.Args[2:])
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
	} else if lassoCommand.Parsed() {
		lassoFit(*lassoCsv, *lassoTarget, *lassoOut, *lambMin)
	} else if plotLassoCommand.Parsed() {
		coeffMin := *lassoPathCoeffMin
		coeffMax := *lassoPathCoeffMax
		var coeffRng featselect.AxisRange
		coeffRngPtr := &coeffRng
		if math.Abs(coeffMax-coeffMin) > 1e-10 {
			coeffRng.Min = coeffMin
			coeffRng.Max = coeffMax
			coeffRngPtr = &coeffRng
		} else {
			coeffRngPtr = nil
		}
		analyseLasso(*lassoPathJSON, *lassoPathOutPrefix, *lassoPathType, coeffRngPtr)
	} else if lassoAicAverage.Parsed() {
		aicAverageLassoPath(*lassoAicPath, *lassoAicOut)
	}
}
