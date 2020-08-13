.PHONY: install
install:
	go get -d -t -v ./...
	go install ./...

test:
	go test ./... -cover

testCLI: testSearch testSearch testBuffer testLasso testNestedLasso testCohenLasso

testSearch:
	goselect bnb --csv data/demo.csv --target 1 --out demo.json --cutoff 0.0 --maxqueue=1000
	rm demo.json

testSASearch:
	goselect sasearch --csv data/demo.csv --target 1 --out demosa.json --sweeps 2
	rm demosa.json

testBuffer:
	goselect-mem -mem=20 -nfeat=62

testLasso:
	goselect lasso --csv data/demo.csv --target=1 --out lassoPath.json
	goselect-plotlasso -json=lassoPath.json -ext=png
	rm lassoPath.json
	rm *.png

testNestedLasso:
	goselect-nestedlasso -csvfile=data/demo.csv -target=1 -out=lassoNested.json
	rm lassoNested.json

testCohenLasso:
	goselect-cohenlasso -csvfile=data/demo.csv -target=1