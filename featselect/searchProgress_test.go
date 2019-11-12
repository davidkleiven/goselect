package featselect

import (
	"math"
	"testing"
)

func TestSetGet(t *testing.T) {
	var sp SearchProgress
	sp.Set(1.0, 2, 3.0)
	bs, ne, np := sp.Get()
	tol := 1e-12
	if math.Abs(bs-1.0) > tol || ne != 2 || math.Abs(np-3.0) > tol {
		t.Errorf("SearchProgress: Expected (%v,%v,%v). Got (%v,%v,%v)", 1.0, 2, 3.0, bs, ne, np)
	}
}
