.PHONY: build run serve

build:
	bundle install

run: build
	bundle exec jekyll serve

serve:
	podman run --rm -it \
		-v "$(PWD)":/srv/jekyll \
		-p 4000:4000 -p 35729:35729 \
		jekyll/jekyll:4.2.2 \
		jekyll serve --host 0.0.0.0 --port 4000 --livereload

deploy:
	rm -rf .jekyll-cache
	bundle
	

