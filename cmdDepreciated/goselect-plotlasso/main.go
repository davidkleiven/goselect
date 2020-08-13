package main

import (
	"flag"
	"fmt"
	"math"

	"github.com/davidkleiven/goselect/featselect"
	"gonum.org/v1/plot/vg"
)

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

func main() {
	lassoPathJSON := flag.String("json", "", "JSON file with the LASSO path")
	lassoPathOutPrefix := flag.String("prefix", "lassofig", "Prefix used in front of all figures")
	lassoPathType := flag.String("ext", "png", "File extension. Support all from gonum plot")
	lassoPathCoeffMin := flag.Float64("coeffmin", 0.0, "Minumum coefficient value in plot")
	lassoPathCoeffMax := flag.Float64("coeffmax", 0.0, "Maximum coefficient value in plot. If equal to the minimum value, the range is autoscaled")

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

	flag.Parse()
	analyseLasso(*lassoPathJSON, *lassoPathOutPrefix, *lassoPathType, coeffRngPtr)
}
