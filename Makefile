.PHONY: install
install:
	go get -u gonum.org/v1/gonum/...

test:
	go test ./... -cover
