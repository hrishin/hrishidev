.PHONY: build run

build:
	bundle

run: build
	bundle exec jekyll serve

