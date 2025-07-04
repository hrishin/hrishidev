.PHONY: build run

build:
	bundle install

run: build
	bundle exec jekyll serve

deploy:
	rm -rf .jekyll-cache
	bundle
	

