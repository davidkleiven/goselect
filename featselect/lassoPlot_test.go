package featselect

import (
	"encoding/json"
	"testing"

	"gonum.org/v1/gonum/mat"

	"gonum.org/v1/gonum/floats"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
)

func getExamplePath() *LassoLarsPath {
	X, y := testfeatselect.GetExampleXY()
	data := NewNormalizedData(X, y)
	res := LassoLars(data, 1e-10)

	var path LassoLarsPath

	var dset Dataset
	dset.X = X
	dset.Y = y
	_, nc := X.Dims()
	dset.Names = make([]string, nc+1)
	dset.TargetCol = 1
	path.Dset = &dset
	path.LassoLarsNodes = res
	return &path
}

func TestLoadLassoLars(t *testing.T) {
	path1 := getExamplePath()
	js, err := json.Marshal(path1)

	if err != nil {
		t.Errorf("Could not marshal path")
	}

	path2 := LassoLarsPathFromBytes(js)

	if len(path2.LassoLarsNodes) != len(path1.LassoLarsNodes) {
		t.Errorf("unexpected path length. Expected %d got %d", len(path1.LassoLarsNodes), len(path2.LassoLarsNodes))
	}

	for i := range path1.LassoLarsNodes {
		c1 := path1.LassoLarsNodes[i].Coeff
		c2 := path2.LassoLarsNodes[i].Coeff
		if !floats.EqualApprox(c1, c2, 1e-10) {
			t.Errorf("unexpected coeff. Expected \n%v\nGot%v\n", c1, c2)
		}
	}

	if !mat.EqualApprox(path1.Dset.X, path2.Dset.X, 1e-10) {
		t.Errorf("Matrices after JSON load does not match! Expected\n%v\nGot%\nv\n", mat.Formatted(path1.Dset.X), mat.Formatted(path2.Dset.X))
	}

	if !floats.EqualApprox(path1.Dset.Y, path2.Dset.Y, 1e-10) {
		t.Errorf("Y-values differ after reading from JSON")
	}
}

func TestLarsPlots(t *testing.T) {
	path := getExamplePath()

	path.PlotEntranceTimes()
	path.PlotQualityScores()
	path.PlotDeviations()
	path.PlotPath(nil)
}

func TestPickMostRelvant(t *testing.T) {
	var path LassoLarsPath

	n1 := LassoLarsNode{Selection: []int{0, 2, 5}}
	n2 := LassoLarsNode{Selection: []int{0, 2, 6}}
	n3 := LassoLarsNode{Selection: []int{7, 4, 3}}

	path.LassoLarsNodes = []*LassoLarsNode{&n1, &n2, &n3}

	relevant := path.PickMostRelevantFeatures(4)
	expect := []int{0, 2, 5, 6}

	for i := range expect {
		if expect[i] != relevant[i] {
			t.Errorf("unexpected relevant nodes. Expected\n%v\nGot\n%v\n", expect, relevant)
		}
	}
}

func TestGetCriteria(t *testing.T) {
	p := getExamplePath()
	res := p.GetCriteria(Aicc)

	// Just make a simple test to ensure that the lengths matches
	if len(res) != len(p.LassoLarsNodes) {
		t.Errorf("Inconsistent length")
	}
}
