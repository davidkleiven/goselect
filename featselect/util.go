package featselect

import (
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// GetDesignMatrix returns the design matrix corresponding to the passed model
func GetDesignMatrix(model []bool, X DesignMatrix) *mat.Dense {
	n, _ := X.Dims()
	numFeat := NumFeatures(model)

	if numFeat == 0 {
		panic("getDesignMatrix: No features in model")
	}
	design := mat.NewDense(n, numFeat, nil)

	col := 0
	for i := 0; i < len(model); i++ {
		if model[i] {
			colView := X.ColView(i)

			for j := 0; j < n; j++ {
				design.Set(j, col, colView.At(j, 0))
			}
			col++
		}
	}
	return design
}

func All(a []int, value int) bool {
	for i := 0; i < len(a); i++ {
		if a[i] != value {
			return false
		}
	}
	return true
}

func IterProduct(values []int, repeat int) [][]int {
	res := make([][]int, 1)

	for r := 0; r < repeat; r++ {
		nItems := len(res)
		updatedRes := make([][]int, 0)
		for j := 0; j < nItems; j++ {
			for i := 0; i < len(values); i++ {
				newRow := append(res[j], values[i])
				updatedRes = append(updatedRes, newRow)
			}
		}
		res = updatedRes
	}
	return res
}

func Sum(a []int) int {
	s := 0
	for _, v := range a {
		s += v
	}
	return s
}

type CommandLineOptions struct {
	Csvfile   string
	Outfile   string
	TargetCol int
}

func ParseCommandLineArgs(args []string) *CommandLineOptions {
	var options CommandLineOptions
	options.Outfile = "defaultGoSelectOut.json"
	for _, v := range args {
		if strings.HasPrefix(v, "--csv=") {
			options.Csvfile = strings.SplitAfter(v, "--csv=")[1]
		} else if strings.HasPrefix(v, "--target=") {
			value := strings.SplitAfter(v, "--target=")[1]
			if num, err := strconv.ParseInt(value, 10, 0); err == nil {
				options.TargetCol = int(num)
			}
		} else if strings.HasPrefix(v, "--out=") {
			options.Outfile = strings.SplitAfter(v, "--out=")[1]
		}
	}
	return &options
}
