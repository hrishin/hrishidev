.PHONY: build run

build:
	bundle

run: build
	bundle exec jekyll serve

deploy: 
	rm -rf .jekyll-cache
	bundle
	

