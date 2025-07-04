---
layout: post
title:  "Examples of using Kubernetes client-go"
date:   2020-11-24 15:00:12 +0000
categories: kubernetes, client-go
---

## Background

One of the beauty of the **Kubernetes** is its _extensiblity_. Just like Kubernetes provide the 
Resources/Objects like `Deployment, StatefulSet, Pod, Service` etc to run the application workloads, it allows defining `Custome Resources` to define the custom tailer workloads. e.g. `Pipelines`, `Task` resources to create CI/CD pipeline workloads. That's _extensibility_.

In order to work with resources Kubernetes community provides a awesome  `client-go` library. It provides the collection of methods()/APIS() for the `Golang` to perform certain  operations on resources such as `create`, `get/list`, `update` and so on. So one can use the Kubernetes HTTP APIS to orchestrate
certain workflow for automating repetitive mundane things.

_One of the challenge sometimes I feel using cleint-go is lack of API usage documentation or examples_

Hence I to overcome this challenge I've started documenting examples of frequently used API' and their options.


## Examples
One can find all examples at [https://github.com/hrishin/k8s-client-go-examples](https://github.com/hrishin/k8s-client-go-examples)






