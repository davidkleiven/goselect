package featselect

import "sync"

type SearchProgress struct {
	BestScore     float64
	NumExplored   int
	Log2NumPruned float64
	rwlock        sync.RWMutex
}

// Set a new state
func (sp *SearchProgress) Set(bs float64, ne int, np float64) {
	sp.rwlock.Lock()
	defer sp.rwlock.Unlock()
	sp.BestScore = bs
	sp.NumExplored = ne
	sp.Log2NumPruned = np
}

// Get current state
func (sp *SearchProgress) Get() (float64, int, float64) {
	sp.rwlock.RLock()
	defer sp.rwlock.RUnlock()
	return sp.BestScore, sp.NumExplored, sp.Log2NumPruned
}
