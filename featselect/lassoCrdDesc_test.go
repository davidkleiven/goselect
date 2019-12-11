package featselect

import (
	"fmt"
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
	"gonum.org/v1/gonum/floats"
)

func TestLassoCrdDesc(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()
	data := NewNormalizedData(X, y)
	var cov Empirical
	res := LassoCrdDesc(data, 0.00001, &cov, nil, 100000)
	fmt.Printf("%v\n", res)
	res[0] = data.LinearTransformationBias([]int{1, 2, 3, 4}, res[1:])
	for i := 1; i < 5; i++ {
		res[i] = data.LinearNormalizationTransformation(i, res[i])
	}

	expect := []float64{-3., 3.0, 0.2, 0.0, 0.0}

	if !floats.EqualApprox(res, expect, 1e-4) {
		t.Errorf("LassoCrdDesc: expected\n%v\nGot\n%v\n", expect, res)
	}
}
