---
layout: post
title:  "Writing a Kubernetes Operator from scratch: PodSet Operator(ReplicaSet controller)"
date:   2020-11-30 10:00:00 +0000
categories: [kubernetes, operators, client-go]
---

## Background

Kubernetes lets you define your own `Resources`/`Objects` on top of the built-in ones like `Deployment`, `StatefulSet`, `Pod` and `Service`. Once you have a `CustomResourceDefinition`, the next question is always: how do you make that resource actually *do* something? That's where the **Operator pattern** comes in — a controller that watches your custom resource and drives the cluster towards the state you declared.

There's plenty of high-level material explaining *what* an operator is, but far less that walks through *building one from nothing* — scaffolding, generating clientsets, writing the reconcile loop, and eventually graduating to shared informers and workqueues. So I put together **PodSet Operator**, a small, deliberately educational operator, to fill that gap.

You can find the full project at [https://github.com/hrishin/podset-operator](https://github.com/hrishin/podset-operator).

## What is a PodSet?

`PodSet` is a toy custom resource, similar in spirit to a `ReplicaSet`, but stripped down to the bare essentials so the controller logic stays easy to follow. You declare how many pods you want:

```yaml
apiVersion: demo.k8s.io/v1alpha1
kind: PodSet
metadata:
  name: three-podset
spec:
  replicas: 3
```

Apply it, and the controller creates the requested number of pods. Delete pods out-of-band, and the controller reconciles the count back to what's declared in `spec.replicas`. Simple to describe, but implementing it properly touches almost everything you need to know to write a real operator.

## Learning path

Rather than dumping a finished controller, the repo is broken into branches (`step-1` through `step-5`) that build up incrementally:

- **Step 1** — Basic project scaffolding and code structure.
- **Step 2** — Defining the `PodSet` CRD types, registering the CRD, and generating client APIs using `client-go` and the Kubernetes code generators (`deepcopy-gen`, `client-gen`, `informer-gen`, `lister-gen`).
- **Step 3** — A functional controller demonstrating `watch` requests and a basic reconciliation loop.
- **Step 4** — A production-pattern controller using shared informers, listers, and workqueues — the same building blocks `kube-controller-manager` itself is built from.
- **Step 5** — The same production-pattern controller rebuilt on [kubebuilder](https://github.com/kubernetes-sigs/kubebuilder), which scaffolds the informers, listers, and workqueue wiring for you.

Each step is meant to be checked out and read on its own, so you can see exactly what changes (and why) as the controller matures from "watch and react" to something closer to what you'd actually run in production.

The project borrows heavily from the Kubernetes `sample-controller` and from *Programming Kubernetes* by Stefan Schimanski and Michael Hausenblas — both excellent references if you want to go deeper after working through this.

**Disclaimer:** this codebase optimizes for educational value, not production-readiness. Error handling, edge cases and testing are deliberately kept light so the core controller pattern stays visible.

## Why I built this

Most of the confusion I saw when people first approach operators wasn't about *why* operators exist — it was the mechanics: what a shared informer actually buys you over a naive `watch`, why you need a workqueue instead of reacting to events inline, and how generated clientsets/listers fit together with your own reconcile code. Stepping through that incrementally, branch by branch, made those pieces click in a way that reading a finished, fully-loaded controller never did for me.

I've since used this material to run hands-on sessions for engineers at both **Red Hat** and **JP Morgan**, walking through the same steps, and it's also formed the basis of a couple of community talks at the [Kubernetes India Meetup](https://www.meetup.com/kubernetes-india-meetup/). It's been a good way to take people from "I've used `kubectl apply` on a CRD" to "I understand what my controller is actually doing when it wakes up."

If you're getting started with writing your own operator, clone the repo, check out `step-1`, and work your way up. Happy to hear feedback or questions from anyone working through it.

[![Presenting PodSet custom resource at the Kubernetes India Meetup](/assets/image1.jpg)](https://www.meetup.com/kubernetes-india-meetup/)

[![Preparing for the meetup](/assets/image2.jpeg)](https://www.meetup.com/kubernetes-india-meetup/)

[![Kubernetes India Meetup audience, wider view](/assets/iamge3.jpeg)](https://www.meetup.com/kubernetes-india-meetup/)

*Photo courtesy of friends at [K8SBLR](https://x.com/k8sBLR).*
