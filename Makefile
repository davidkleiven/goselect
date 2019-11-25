.PHONY: install
install:
	go get -u gonum.org/v1/gonum/mat

build:
	go build main.go

test:
	go test ./... -cover

testCLI: build testSearch testNormalize testBuffer

testSearch:
	./main -csvfile=data/demo.csv -target=1 -out=demo.json

testNormalize:
	./main std -csvfile=data/demo.csv -out=demoNorm.csv
	rm demoNorm.csv

testBuffer:
	./main bufferSize -mem=20 -nfeat=62
