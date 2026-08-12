FROM ruby:3.1-alpine

RUN apk add --no-cache \
        build-base \
        git \
        less \
        bash \
        tzdata \
        libffi-dev \
        zlib-dev \
        libxml2-dev \
        libxslt-dev \
        yaml-dev \
        autoconf \
        automake \
        libtool \
        patch

WORKDIR /srv/jekyll

COPY Gemfile Gemfile.lock ./

RUN gem install bundler:2.1.4 && \
    bundle _2.1.4_ install

EXPOSE 4000 35729

CMD ["bundle", "_2.1.4_", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--port", "4000", "--livereload", "--force_polling"]
