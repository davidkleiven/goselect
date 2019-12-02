.PHONY: install
install:
	go get -u gonum.org/v1/gonum/mat

build:
	go build main.go

test:
	go test ./... -cover

testCLI: build testSearch testNormalize testBuffer testLasso

testSearch:
	./main search -csvfile=data/demo.csv -target=1 -out=demo.json -cutoff=0.0 -maxQueueSize=1000

testNormalize:
	./main std -csvfile=data/demo.csv -out=demoNorm.csv
	rm demoNorm.csv

testBuffer:
	./main bufferSize -mem=20 -nfeat=62

testLasso:
	./main lasso -csvfile=data/demo.csv -target=1 --out=lassoPath.json
	rm lassoPath.json
