package featselect

import (
	"math"
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
)

func TestCohensKappaCalc(t *testing.T) {
	var target testfeatselect.MockCohenTarget
	kappa := GetCohensKappa(20, &target)

	if math.Abs(kappa-1.0) > 1e-10 {
		t.Errorf("Expected 1.0 got %f", kappa)
	}
}

func TestCohensKappaPath(t *testing.T) {
	var t1 testfeatselect.MockCohenTarget
	t1.Mode = 1
	var t2 testfeatselect.MockCohenTarget
	t2.Mode = 0
	var t3 testfeatselect.MockCohenTarget
	t3.Mode = 2
	var t4 testfeatselect.MockCohenTarget
	t4.Mode = 2

	targets := make([]CohensKappaTarget, 4)
	targets[0] = &t1
	targets[1] = &t2
	targets[2] = &t3
	targets[3] = &t4

	hyper := CalculateCohenSequence(20, targets)

	if hyper["mode"] != 0 {
		t.Errorf("Unexpected best kappa. Expected 0 got %v", hyper)
	}
}
