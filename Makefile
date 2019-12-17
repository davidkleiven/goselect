.PHONY: install
install:
	go get -d -t -v ./...
	go install ./...

build:
	go build main.go

test:
	go test ./... -cover

testCLI: build testSearch testSearch testNormalize testBuffer testLasso

testSearch:
	goselect-bnb -csvfile=data/demo.csv -target=1 -out=demo.json -cutoff=0.0 -maxQueueSize=1000

testSASearch:
	goselect-sa -csvfile=data/demo.csv -target=1 -out=demosa.json -sweeps=2
	rm demosa.json

testNormalize:
	./main std -csvfile=data/demo.csv -out=demoNorm.csv
	rm demoNorm.csv

testBuffer:
	./main bufferSize -mem=20 -nfeat=62

testLasso:
	./main lasso -csvfile=data/demo.csv -target=1 --out=lassoPath.json
	./main lassoavg -json=lassoPath.json -out=lassoAICAvg.json
	rm lassoPath.json
	rm lassoAICAvg.json
