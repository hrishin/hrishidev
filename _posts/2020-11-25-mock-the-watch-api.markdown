---
layout: post
title:  "Unit test client-go watch API to mock the watch events in Kubernetes"
date:   2020-11-24 21:00:12 +0000
categories: Kubernetes, client-go
---

# Unit test client-go watch API to mock the watch events using client-go testing package

Sometimes we encounter the case where we need to simulate the watch events in 
order to test code that uses client-go `watch` API in the Kubernetes world.

In this example, we will see how to mock `watch` API events sequence as part of the unit testing.

Before that, let's see how to use the `fake` package to `client-go` APIs.

In this case, we will use the `pod()` API. You can find the complete example at [https://github.com/hrishin/k8s-client-go-examples/tree/main/examples/mock-watch-events](https://github.com/hrishin/k8s-client-go-examples/tree/main/examples/mock-watch-events)

## Test get pod by name

The following snippet initialize the fake client by feeding a  `pod` resource.
```
import(
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

client := fake.NewSimpleClientset(&v1.Pod{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "fake",
		Namespace: "fake",
	},
})
```

Test it by running `make test-get-pod`

```
➜  mock-watch-events git:(mock-watch) ✗ make test-get-pod
go test -run Test_get_pod_using_fake_client -v
=== RUN   Test_get_pod_using_fake_client
    fake_client_test.go:28: Fetch the pod by pod name using the client-go API 
    fake_client_test.go:31: 	Test 0: checking the error code response
    fake_client_test.go:36: 	✓	client go has return no error.
    fake_client_test.go:39: 	Test 1: verifying the retrived pod from client-go get pod API
    fake_client_test.go:44: 	✓	client go has returned the expected pod
--- PASS: Test_get_pod_using_fake_client (0.00s)
PASS
ok  	github.com/hrishin/k8s-client-go-examples/examples/mock-watch-events	0.571s
```

# Test pod watch events mocking

In this scenario, we will simulate pod life cycle events i.e. `pod.status.phase` -> {PodPending,  PodUnknown, PodRunning}.
Usually, we encounter such code to wait for the pod to become up & running.

To feed such events `client-go` provides the `testing` package. Following example snippet to feed the mock events for the `watch` API.

```
clients := fake.NewSimpleClientset()
watcher := watch.NewFake()
clients.PrependWatchReactor("pods", k8stest.DefaultWatchReactor(watcher, nil))

go func() {
	defer watcher.Stop()

	for i, _ := range pods {
		time.Sleep(300 * time.Millisecond)
		watcher.Add(&v1.Pod{
		   ..... // your pod definitions
		})
	}
}()
```

Important to note here that in `clients.PrependWatchReactor("pods", k8stest.DefaultWatchReactor(watcher, nil))` method `pods` is the plural resource name. Giving the wrong resource name would fail mocking watch events. One of the way to get the resource name is using 
```
kubectl api-resources | grep -i pod

NAME  SHORTNAMES APIGROUP NAMESPACED KIND
pods  po  				  true 		 Pod
```

Test the example by running `make test-watch-pod`

```
➜  mock-watch-events git:(mock-watch) ✗ make test-watch-pod
go test -run Test_watch_pod_using_fake_client -v
=== RUN   Test_watch_pod_using_fake_client
    fake_client_test.go:91: Watch pod updates by pod name using the client-go API 
    fake_client_test.go:97:     Test 0: checking the error code response
    fake_client_test.go:102:    ✓       client go has return no error.
    fake_client_test.go:105:    Test 1: checking watch event updates
    fake_client_test.go:115:    ✓       got a pod update event
    fake_client_test.go:122:    ✓       expecting pod phase Pending and got Pending
    fake_client_test.go:115:    ✓       got a pod update event
    fake_client_test.go:122:    ✓       expecting pod phase Unknown and got Unknown
    fake_client_test.go:115:    ✓       got a pod update event
    fake_client_test.go:122:    ✓       expecting pod phase Running and got Running
--- PASS: Test_watch_pod_using_fake_client (0.91s)
PASS
ok      github.com/hrishin/k8s-client-go-examples/examples/mock-watch-events    1.490s
```

I hope this post will be useful. Would like to hear your reviews, feedback or your experience. 
Happy programming with Kubernetes!
