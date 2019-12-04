package featselect

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

const tol = 1e-10

// LassoLarsNode is a structure the result of one of the lasso path
type LassoLarsNode struct {
	Coeff     []float64
	Lamb      float64
	Selection []int
}

// NewLassoLarsNode creates a new lasso-lars node
func NewLassoLarsNode(coeff []float64, lamb float64, selection []int) *LassoLarsNode {

	if len(coeff) != len(selection) {
		panic("lassolars: inconsistent lengths")
	}
	var l LassoLarsNode
	l.Coeff = make([]float64, len(coeff))
	copy(l.Coeff, coeff)
	l.Lamb = lamb
	l.Selection = make([]int, len(selection))
	copy(l.Selection, selection)
	return &l
}

// LassoLarsParams is a convenience struct defined to hold the variable c and d
// in Tibshirani, R.J., 2013. The lasso problem and uniqueness. Electronic Journal of Statistics, 7, pp.1456-1490.
type LassoLarsParams struct {
	c *mat.VecDense
	d *mat.VecDense
}

// LastZeroed holds information about the feature that least left the
type LastZeroed struct {
	feat   int
	coeff  float64
	active bool
}

// LassoLars computes the LASSO solution wiith the LARS algorithm
func LassoLars(data *NormalizedData, lambMin float64) []*LassoLarsNode {
	nr, nc := data.X.Dims()
	allSigns := mat.NewVecDense(nc, nil)
	yVec := mat.NewVecDense(nr, data.y)

	lambJoin := mat.NewVecDense(nc, nil)
	lambJoin.MulVec(data.X.T(), yVec)

	maxTime := 0.0
	feat := 0
	for i := 0; i < lambJoin.Len(); i++ {
		v := math.Abs(lambJoin.AtVec(i))
		if v > maxTime {
			maxTime = v
			feat = i
		}
	}

	allSigns.SetVec(feat, 1.0)

	if lambJoin.AtVec(feat) < 0.0 {
		allSigns.SetVec(feat, -1.0)
	}
	//fmt.Printf("%v\n", mat.Formatted(lambJoin))
	var res []*LassoLarsNode
	activeSet := []int{feat}
	lamb := maxTime

	var llp LassoLarsParams
	var svd mat.SVD

	var last LastZeroed
	last.feat = -1
	for lamb > lambMin {
		if len(activeSet) == 0 {
			panic("lassolars: Active set is empty!")
		}

		// To make it consistent with how the design matrix works, we keep the activeSet
		// in ascending order
		sort.Ints(activeSet)

		signs := NewIndexedColVecView(allSigns, activeSet)
		Xe := NewIndexedColView(data.X, activeSet)
		svd.Factorize(&Xe, mat.SVDThin)

		invD := invDesignMatrix(&svd)
		llp.c = cParameter(&svd, yVec)
		llp.d = dParameter(invD, signs)

		joinTimes := tJoin(data.X, &Xe, yVec, &llp, lamb, activeSet, last)
		crossTimes := tCross(&llp, lamb)

		jt, jFeat := maxJoinTime(joinTimes, activeSet)
		ct, cFeat := maxCrossTime(crossTimes)
		lrs := betaLARS(&llp, lamb)
		res = append(res, NewLassoLarsNode(lrs.RawVector().Data, lamb, activeSet))

		if jt > ct {
			last.feat = -1
			activeSet = append(activeSet, jFeat)
			coeffVec := betaLARS(&llp, jt)
			allSigns.SetVec(jFeat, joinFeatSign(jFeat, data.X, &Xe, yVec, coeffVec))

			if jt > lamb {
				// We are close to numerical tolerance
				break
			}
			lamb = jt
		} else {
			last.coeff = betaLARS(&llp, lamb).AtVec(cFeat)
			last.feat = activeSet[cFeat]
			activeSet[cFeat] = activeSet[len(activeSet)-1]
			activeSet = activeSet[:len(activeSet)-1]

			if ct > lamb {
				// We are close to numerical tolerance
				break
			}
			lamb = ct
		}
	}
	return res
}

func maxJoinTime(v *mat.VecDense, active []int) (float64, int) {
	// Avoid that indices in the active set are selected
	for _, ind := range active {
		v.SetVec(ind, -1.0)
	}

	maxVal := 0.0
	maxIndx := 0
	for i := 0; i < v.Len(); i++ {
		if v.AtVec(i) > maxVal {
			maxVal = v.AtVec(i)
			maxIndx = i
		}
	}
	return maxVal, maxIndx
}

func maxCrossTime(v *mat.VecDense) (float64, int) {
	maxVal := 0.0
	maxIndx := 0
	for k := 0; k < v.Len(); k++ {
		if v.AtVec(k) > maxVal {
			maxVal = v.AtVec(k)
			maxIndx = k
		}
	}
	return maxVal, maxIndx
}

// cParameter calculates the c-value in the algorithm presented in
// this paper
// Tibshirani, R.J., 2013. The lasso problem and uniqueness. Electronic Journal of Statistics, 7, pp.1456-1490.
func cParameter(svd *mat.SVD, y mat.Vector) *mat.VecDense {
	s := svd.Values(nil)
	var v mat.Dense
	var u mat.Dense
	svd.VTo(&v)
	svd.UTo(&u)

	for i := 0; i < len(s); i++ {
		if s[i]*s[i] > tol {
			s[i] = 1.0 / s[i]
		}
	}

	diag := mat.NewDiagDense(len(s), s)
	nr, _ := v.Dims()
	cMat := mat.NewDense(nr, 1, nil)
	cMat.Product(&v, diag, u.T(), y)
	res := mat.NewVecDense(nr, nil)
	for i := 0; i < nr; i++ {
		res.SetVec(i, cMat.At(i, 0))
	}
	return res
}

// dParameter calculates the d-value in the algorithm presented in
// this paper
// Tibshirani, R.J., 2013. The lasso problem and uniqueness. Electronic Journal of Statistics, 7, pp.1456-1490.
func dParameter(invD mat.Matrix, signs mat.Vector) *mat.VecDense {
	nr, _ := invD.Dims()
	res := mat.NewVecDense(nr, nil)
	res.MulVec(invD, signs)
	return res
}

// invDesignMatrix returns the inverse of X^T X
func invDesignMatrix(svd *mat.SVD) *mat.Dense {
	s := svd.Values(nil)
	var v mat.Dense
	svd.VTo(&v)
	for i := 0; i < len(s); i++ {
		if s[i]*s[i] > tol {
			s[i] = 1.0 / (s[i] * s[i])
		} else {
			s[i] = 0.0
		}
	}

	diag := mat.NewDiagDense(len(s), s)
	nr, _ := v.Dims()
	res := mat.NewDense(nr, nr, nil)
	res.Product(&v, diag, v.T())
	return res
}

// tJoin calculates the joining time for all features
func tJoin(X mat.Matrix, Xe mat.Matrix, y mat.Vector, llp *LassoLarsParams, lamb float64, active []int, last LastZeroed) *mat.VecDense {
	_, nc := X.Dims()
	joinTime := mat.NewVecDense(nc, nil)

	// Pre-calculate products
	Xy := mat.NewVecDense(nc, nil)
	Xy.MulVec(X.T(), y)

	xenr, _ := Xe.Dims()
	XXeCtmp := mat.NewVecDense(xenr, nil)
	XXeCtmp.MulVec(Xe, llp.c)
	XXeC := mat.NewVecDense(nc, nil)
	XXeC.MulVec(X.T(), XXeCtmp)

	XXeDtmp := mat.NewVecDense(xenr, nil)
	XXeDtmp.MulVec(Xe, llp.d)
	XXeD := mat.NewVecDense(nc, nil)
	XXeD.MulVec(X.T(), XXeDtmp)

	isActive := make([]bool, nc)
	for _, v := range active {
		isActive[v] = true
	}

	for i := 0; i < nc; i++ {
		tpluss := (Xy.AtVec(i) - XXeC.AtVec(i)) / (1.0 - XXeD.AtVec(i))
		tminus := (Xy.AtVec(i) - XXeC.AtVec(i)) / (-1.0 - XXeD.AtVec(i))

		// Force to use the one with the opposite sign of the coefficient
		if last.feat == i && last.coeff > 0.0 {
			tpluss = -10.0
		} else if last.feat == i && last.coeff < 0.0 {
			tminus = -10.0
		}

		if tpluss >= -tol && tpluss <= lamb+tol {
			joinTime.SetVec(i, tpluss)
		} else if tminus >= -tol && tminus <= lamb+tol {
			joinTime.SetVec(i, tminus)
		} else {
			panic("larslasso: Feature never included")
		}
	}
	return joinTime
}

// tCross calculates the crossing times
func tCross(llp *LassoLarsParams, lamb float64) *mat.VecDense {
	cross := mat.NewVecDense(llp.c.Len(), nil)
	for i := 0; i < cross.Len(); i++ {
		ratio := llp.c.AtVec(i) / llp.d.AtVec(i)

		if ratio+tol > lamb {
			ratio = 0.0
		}
		cross.SetVec(i, ratio)
	}
	return cross
}

// betaLARS calculates the LASSO-LARS coefficients
func betaLARS(llp *LassoLarsParams, lamb float64) *mat.VecDense {
	beta := mat.NewVecDense(llp.c.Len(), nil)

	for i := 0; i < beta.Len(); i++ {
		beta.SetVec(i, llp.c.AtVec(i)-lamb*llp.d.AtVec(i))
	}
	return beta
}

func joinFeatSign(jFeat int, X mat.Matrix, Xe mat.Matrix, y mat.Vector, coeffVec mat.Vector) float64 {
	v := mat.NewVecDense(y.Len(), nil)
	v.MulVec(Xe, coeffVec)
	dotProd := 0.0
	for i := 0; i < v.Len(); i++ {
		dotProd += X.At(i, jFeat) * (y.AtVec(i) - v.AtVec(i))
	}

	if dotProd > 0.0 {
		return 1.0
	}
	return -1.0
}

// Path2Unnormalized converts all the coefficients in the path to unnormalzed values
func Path2Unnormalized(data *NormalizedData, path []*LassoLarsNode) {
	for _, node := range path {
		bias := data.LinearTransformationBias(node.Selection, node.Coeff)

		for i := range node.Selection {
			node.Coeff[i] = data.LinearNormalizationTransformation(node.Selection[i], node.Coeff[i])
		}

		if data.HasBias {
			tmpSelect := make([]int, len(node.Selection)+1)
			tmpSelect[0] = 0
			copy(tmpSelect[1:], node.Selection)
			node.Selection = tmpSelect

			tmpCoeff := make([]float64, len(node.Coeff)+1)
			tmpCoeff[0] = bias
			copy(tmpCoeff[1:], node.Coeff)
			node.Coeff = tmpCoeff
		}
	}
}

// LassoNodesSlice2SparsCoeff converts a slice with LassoLarsNodes into a
// SparseCoeff slice (which is simply to transfer the Coeff and Selection slices)
func LassoNodesSlice2SparsCoeff(nodes []*LassoLarsNode) []SparseCoeff {
	sp := make([]SparseCoeff, len(nodes))
	for i := range nodes {
		sp[i].Coeff = nodes[i].Coeff
		sp[i].Selection = nodes[i].Selection
	}
	return sp
}
