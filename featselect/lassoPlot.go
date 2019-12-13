package featselect

import (
	"encoding/json"
	"io/ioutil"
	"math"
	"os"

	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// LassoLarsPath is a datatype that reads the data from json
type LassoLarsPath struct {
	Dset           *Dataset
	LassoLarsNodes []*LassoLarsNode
	Aicc           []float64
	Bic            []float64
}

// LassoLarsPathFromJSON loads the lasso lars path from a JSON file
func LassoLarsPathFromJSON(fname string) *LassoLarsPath {
	file, err := os.Open(fname)
	if err != nil {
		panic("lassolarspathfromjson: Error when opening file!")
	}
	defer file.Close()
	byteValue, _ := ioutil.ReadAll(file)
	return LassoLarsPathFromBytes(byteValue)
}

// LassoLarsPathFromBytes loads the lasso lars path from a bytes array
func LassoLarsPathFromBytes(data []byte) *LassoLarsPath {
	var lassoPath LassoLarsPath
	json.Unmarshal(data, &lassoPath)
	return &lassoPath
}

// MaxMinFeatNo reeturns the minimum and maximum feature in the set
func (p *LassoLarsPath) MaxMinFeatNo() (int, int) {
	minVal := math.MaxInt32
	maxVal := 0
	for _, n := range p.LassoLarsNodes {
		max := MaxInt(n.Selection)
		min := MinInt(n.Selection)

		if max > maxVal {
			maxVal = max
		}

		if min < minVal {
			minVal = min
		}
	}
	return minVal, maxVal
}

// PlotEntranceTimes plots the lambda values when a feature is selected
func (p *LassoLarsPath) PlotEntranceTimes() *plot.Plot {
	plt, err := plot.New()

	if err != nil {
		panic(err)
	}

	minFeat, maxFeat := p.MaxMinFeatNo()
	firstEntrance := make(plotter.XYs, maxFeat-minFeat+1)
	featureDidEnter := make([]bool, maxFeat-minFeat+1)

	for i := range firstEntrance {
		firstEntrance[i].Y = -math.MaxFloat64
		firstEntrance[i].X = float64(i + minFeat)
	}

	var entranceTimes plotter.XYs

	for _, v := range p.LassoLarsNodes {
		logLamb := math.Log10(v.Lamb)
		for _, feat := range v.Selection {
			entranceTimes = append(entranceTimes, plotter.XY{X: float64(feat), Y: logLamb})

			if logLamb > firstEntrance[feat-minFeat].Y {
				firstEntrance[feat-minFeat].Y = logLamb
				featureDidEnter[feat-minFeat] = true
			}
		}
	}

	s, err := plotter.NewScatter(entranceTimes)

	if err != nil {
		panic(err)
	}

	// Remove all features that never entered
	var filteredFirstEntrance plotter.XYs
	for i, v := range firstEntrance {
		if featureDidEnter[i] {
			filteredFirstEntrance = append(filteredFirstEntrance, v)
		}
	}

	l, err := plotter.NewLine(filteredFirstEntrance)
	l.Color = color.RGBA{R: 85, G: 122, B: 149, A: 255}
	l.Width = vg.Points(2)
	if err != nil {
		panic(err)
	}

	plt.Add(s, l)
	plt.X.Label.Text = "Feat. No."
	plt.Y.Label.Text = "Regularization"
	return plt
}

// GetCriteria returns the value of the passed criteria along the path
func (p *LassoLarsPath) GetCriteria(criteria crit) []float64 {
	nData, nFeat := p.Dset.X.Dims()
	values := make([]float64, len(p.LassoLarsNodes))

	for i, n := range p.LassoLarsNodes {
		model := Selected2Model(n.Selection, nFeat)
		design := GetDesignMatrix(model, p.Dset.X)
		rss := Rss(design, n.Coeff, p.Dset.Y)
		num := NumFeatures(model)
		values[i] = criteria(num, nData, rss)
	}
	return values
}

// PlotQualityScores plot the AICC value of the path
func (p *LassoLarsPath) PlotQualityScores() *plot.Plot {
	plt, err := plot.New()

	if err != nil {
		panic(err)
	}

	aicValuesSlice := p.GetCriteria(Aicc)
	aiccVals := make(plotter.XYs, len(p.LassoLarsNodes))
	for i, n := range p.LassoLarsNodes {
		aiccVals[i] = plotter.XY{X: math.Log10(n.Lamb), Y: aicValuesSlice[i]}
	}

	l, err := plotter.NewLine(aiccVals)
	if err != nil {
		panic(err)
	}

	plt.Add(l)
	plt.X.Label.Text = "Regularization"
	plt.Y.Label.Text = "AICC"
	return plt
}

// PlotDeviations plots RMSE error and GCV
func (p *LassoLarsPath) PlotDeviations() *plot.Plot {
	plt, err := plot.New()

	if err != nil {
		panic(err)
	}

	nData, nFeat := p.Dset.X.Dims()
	rmseVals := make(plotter.XYs, len(p.LassoLarsNodes))
	for i, n := range p.LassoLarsNodes {
		model := Selected2Model(n.Selection, nFeat)
		design := GetDesignMatrix(model, p.Dset.X)
		rss := Rss(design, n.Coeff, p.Dset.Y)
		rmse := math.Sqrt(rss / float64(nData))
		rmseVals[i] = plotter.XY{X: math.Log10(n.Lamb), Y: math.Log10(rmse)}
	}

	l, err := plotter.NewLine(rmseVals)
	l.Color = color.RGBA{R: 109, B: 108, G: 81, A: 255}
	if err != nil {
		panic(err)
	}

	plt.Add(l)
	plt.X.Label.Text = "Regularization"
	plt.Y.Label.Text = "Deviation"
	return plt
}

// ExtractPath extracts the path of one coefficient
func (p *LassoLarsPath) ExtractPath(featNo int) plotter.XYs {
	xys := make(plotter.XYs, len(p.LassoLarsNodes))

	for i, n := range p.LassoLarsNodes {
		found := false
		for j := range n.Selection {
			if n.Selection[j] == featNo {
				xys[i] = plotter.XY{X: math.Log10(n.Lamb), Y: n.Coeff[j]}
				found = true
				break
			}
		}

		if !found {
			xys[i] = plotter.XY{X: math.Log10(n.Lamb), Y: 0.0}
		}
	}
	return xys
}

// AxisRange is used to pass max and min information for an axis
type AxisRange struct {
	Min, Max float64
}

// PlotPath plots the LassoLarsPath
func (p *LassoLarsPath) PlotPath(cr *AxisRange) *plot.Plot {
	plt, err := plot.New()

	if err != nil {
		panic(err)
	}

	_, numFeat := p.MaxMinFeatNo()

	for featNo := 0; featNo < numFeat; featNo++ {
		xy := p.ExtractPath(featNo)
		line, err := plotter.NewLine(xy)

		if err != nil {
			panic(err)
		}

		plt.Add(line)
	}

	// Add zero line
	lambMin := math.Log10(p.LassoLarsNodes[len(p.LassoLarsNodes)-1].Lamb)
	lambMax := math.Log10(p.LassoLarsNodes[0].Lamb)
	zeroXY := plotter.XYs{{X: lambMin, Y: 0.0}, {X: lambMax, Y: 0.0}}

	zeroLine, err := plotter.NewLine(zeroXY)
	if err != nil {
		panic(err)
	}
	zeroLine.Width = 2
	zeroLine.Color = color.RGBA{R: 85, G: 122, B: 149, A: 255}
	plt.Add(zeroLine)

	plt.X.Label.Text = "Regularization"
	plt.Y.Label.Text = "Coefficients"

	if cr != nil {
		plt.Y.Min = cr.Min
		plt.Y.Max = cr.Max
	}

	return plt
}

// PickMostRelevantFeatures picks out a subset of features based on when they
// entered the lasso path
func (p *LassoLarsPath) PickMostRelevantFeatures(numFeat int) []int {
	mostRelevant := []int{}
	for _, node := range p.LassoLarsNodes {
		for _, v := range node.Selection {
			if !ExistInt(mostRelevant, v) {
				mostRelevant = append(mostRelevant, v)
			}
		}
		if len(mostRelevant) >= numFeat {
			return mostRelevant
		}
	}
	return mostRelevant
}
