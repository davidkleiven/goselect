.PHONY: install
install:
	go get -u gonum.org/v1/gonum/mat

test:
	go test ./...
