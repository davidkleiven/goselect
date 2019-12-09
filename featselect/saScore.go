package featselect

// SAItem is one item in the SA queue
type SAItem struct {
	Selection []int
	Score     float64
}

// NewSAItem creates a new instane of SAIte
func NewSAItem(model []bool) *SAItem {
	var saItem SAItem
	if model == nil {
		return &saItem
	}

	n := NumFeatures(model)
	saItem.Selection = make([]int, n)
	counter := 0
	for i, v := range model {
		if v {
			saItem.Selection[counter] = i
			counter++
		}
	}
	return &saItem
}

// SAScore is a type that is used to efficiently maintain a highscore list of
// simmulated annealing features
type SAScore struct {
	Cap       int
	Items     []*SAItem
	BestItem  *SAItem
	WorstItem *SAItem
}

// NewSAScore creates a new item with the scores
func NewSAScore(length int) *SAScore {
	var saScore SAScore
	saScore.Cap = length
	return &saScore
}

// Insert a new item in the queue
func (s *SAScore) Insert(item *SAItem) {
	if s.Exists(item) {
		return
	}

	if len(s.Items) == 0 {
		s.Items = append(s.Items, item)
		s.WorstItem = item
		s.BestItem = item
	} else if len(s.Items) < s.Cap {
		s.Items = append(s.Items, item)
		if item.Score > s.BestItem.Score {
			s.BestItem = item
		}

		if item.Score < s.WorstItem.Score {
			s.WorstItem = item
		}
	} else {
		if item.Score > s.WorstItem.Score {
			s.WorstItem.Selection = item.Selection
			s.WorstItem.Score = item.Score

			if item.Score > s.BestItem.Score {
				s.BestItem = item
			}

			// Update the worst item
			minScore := item.Score
			for _, v := range s.Items {
				if v.Score < minScore {
					minScore = v.Score
					s.WorstItem = v
				}
			}
		}
	}
}

// Exists return true if the item already exists in the queue
func (s *SAScore) Exists(item *SAItem) bool {
	for _, v := range s.Items {
		if EqualInt(v.Selection, item.Selection) {
			return true
		}
	}
	return false
}
