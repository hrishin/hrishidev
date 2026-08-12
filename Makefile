IMAGE := hrishidev-jekyll:4.1.1

.PHONY: build run image serve

build:
	bundle install

run: build
	bundle exec jekyll serve

image:
	podman build -t $(IMAGE) .

serve: image
	podman run --rm -it \
		-v "$(PWD)":/srv/jekyll:Z \
		-p 4000:4000 -p 35729:35729 \
		$(IMAGE) \
		bundle _2.1.4_ exec jekyll serve --host 0.0.0.0 --port 4000 --livereload --force_polling

deploy:
	rm -rf .jekyll-cache
	bundle
	

