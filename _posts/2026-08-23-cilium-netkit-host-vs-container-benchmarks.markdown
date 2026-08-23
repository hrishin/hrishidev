---
layout: post
title:  "Linux Bare Host vs. Container with Cilium netkit: What Actually Wins"
date:   2026-08-23 20:00:00 +0000
categories: [Linux, Kubernetes, Networking, netkit, Benchmarking]
---

*Six workloads, one self-managed cluster, and a question I kept getting wrong until I measured it: does the extra hop through a container's CNI add real network overhead versus the same binary running as a plain Linux process on the bare host, or does Cilium's netkit datapath actually eliminate that bottleneck?*

netkit, for anyone new to it: a Linux kernel network device type, merged into mainline Linux 6.7, contributed largely by Cilium/Isovalent engineers and currently used as Cilium's container networking datapath. No `veth` pair, no virtual switch in the middle, just a lightweight in-kernel construct with eBPF hooks (`netkit/primary` and `netkit/peer`) on the send and receive path[^netkit-isovalent].

## The setup

A self-managed Kubernetes cluster on a small cloud VM instance type throughout (4 vCPUs, ~8GB RAM, ~8GB disk per node, Ubuntu 26.04 LTS, kernel 7.0.0-14-generic), Cilium in **native routing** mode (no VXLAN, no encapsulation overhead) with `bpf.datapathMode=netkit` and `loadBalancer.mode=dsr`. Every test below runs the *same binary* two ways: as a **bare Linux process** on the node (a normal `systemd`-managed process in the node's own network namespace, no container, no CNI, no `netkit` in the path at all) and as a **container behind Cilium's `netkit` device**, the same node, the same kernel, just routed through Cilium's datapath instead of a plain socket. "Bare host" here means bare of any container runtime and CNI, not bare-metal hardware: every node is a cloud VM, so both sides of every comparison share the same hypervisor-level virtualization; the variable under test is strictly the Linux networking path from that point down, host netns vs. `netkit`. No test ever uses `hostNetwork: true` to fake that comparison. Where a test involves two nodes, the host side talks over the cloud provider's private network and the container side talks over Cilium's pod network, both real, both production-shaped.

Two cluster shapes appear in this post, same instance type, different split. HTTP, Redis, and north-south ran on 3 control-plane + 2 workers (5 nodes). The quorum-based results (etcd, PostgreSQL, Kafka) and the MPI ping-pong lead with numbers from a **rebuilt 1 control-plane + 3 worker** cluster (4 nodes): a 2-node "quorum" is degenerate (majority of 2 is 2, so it's actually unanimity, not a majority vote), and real quorum-based systems don't run that way in production; MPI's re-run there isn't a quorum result, but it reuses the same 3-worker layout to isolate the Slurm controller (`worker-01`) from the two nodes actually carrying MPI traffic. Getting a genuine 3-of-3-worker majority meant trading two control-plane nodes for one extra worker, since the control-plane doesn't need HA for anything measured here.

```mermaid!
graph TD
    subgraph "Control plane (1 node)"
        CP1["control-plane-01<br/>(north-south client, earlier tests)"]
    end
    subgraph "Workers: quorum tests run here"
        W1["worker-01<br/>bare Linux process + matching netkit pod"]
        W2["worker-02<br/>bare Linux process + matching netkit pod"]
        W3["worker-03<br/>bare Linux process + matching netkit pod"]
    end
    W1 <==> W2
    W2 <==> W3
    W1 <==> W3
```

Every quorum result compares the same three workers: a bare Linux process on each versus a netkit-backed pod on each, never a mix. The earlier 3-control-plane + 2-worker shape (referenced for HTTP, Redis, and north-south, and for the 2-node numbers shown alongside each 3-node quorum result below) followed the identical rule, just with `worker-01`/`worker-02` only. MPI is the one non-quorum test that also moved to the 3-worker shape, for the isolation reason noted above, not because of quorum semantics.

Almost everything below is **east-west** traffic: pod-to-pod (or host-to-host) inside the cluster, the shape most inter-service and database traffic actually takes. There's one **north-south** data point too: an external client (`control-plane-01`, itself a plain host, no Kubernetes networking on its own side) hitting the worker directly for the host case, and hitting the same worker's Service `ClusterIP` (resolved via Cilium's host-reachable-services, an eBPF hook at the client's own socket layer, no `kube-proxy` involved) for the container case. Worth keeping separate: it's a different traffic shape from the paired host-vs-host / pod-vs-pod comparisons everywhere else in this post.

Every system here runs on its distribution's default configuration: default `sysctl` values, default TCP buffer sizes, no CPU pinning or NUMA tuning, no custom kernel parameters, stock package configs for Redis/PostgreSQL/Kafka/etcd beyond the minimum needed to get replication or clustering working at all. Nothing here is tuned for the benchmark, on either side of the comparison, deliberately: the question is what happens to an unmodified deployment, not what's achievable after a tuning pass.

Worth being explicit that this is a limitation, not just a methodology choice: an untuned Linux host is not the host's best case. Kernel-parameter tuning (`sysctl` network buffer sizes, IRQ/CPU affinity, NUMA pinning, and similar) is exactly the kind of work a latency-sensitive production deployment would actually do, and it could move the host-side numbers in this post, possibly enough to change which side wins some of the closer results. Nothing here says a tuned bare-metal host would still lose to netkit; it says an out-of-the-box one does, on this hardware, for these workloads.

One methodological note worth keeping in mind while reading: not every "container wins" result is actually about Cilium. An early `iperf3` throughput test showed container traffic 2.4x faster than host-to-host, until holding the network path constant (private network on both sides) showed the *host* hitting the identical ceiling: 1,270 Mbit/s either way. That gap was the cloud provider's private network having a higher bandwidth ceiling than its public network, nothing to do with the CNI. Every result below has had that kind of confound checked for.

## Where the container with netkit wins: concurrent, dispatch-bound traffic

First, east-west: `wrk` (20 connections) runs on `worker-02` and hits the identical `whoami` binary on `worker-01`, two ways. Host case: a direct socket connection, no container, no CNI. Container case: through the `whoami` Service's `ClusterIP` (Cilium's DSR path), not bypassed to the pod IP directly:

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="http-throughput-host-vs-container-wrk-20-connections-title http-throughput-host-vs-container-wrk-20-connections-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="http-throughput-host-vs-container-wrk-20-connections-title">HTTP throughput, host vs container (wrk, 20 connections)</title>
  <desc id="http-throughput-host-vs-container-wrk-20-connections-desc">HTTP throughput, host vs container (wrk, 20 connections). grouped bar data: Requests/sec: Host 3014, Container 14582.Source: Author benchmark, whoami on Cilium netkit, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">HTTP throughput, host vs container (wrk, 20 connections)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="264.0" y="238.8" width="16.0" height="38.2" fill="#f97316" />
<text x="272.0" y="234.8" text-anchor="middle" font-size="10" fill="currentColor">3,014</text>
<rect x="282.0" y="92.0" width="16.0" height="185.0" fill="#38bdf8" />
<text x="290.0" y="88.0" text-anchor="middle" font-size="10" fill="currentColor">14,582</text>
<text x="282.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="282.0" dy="0">Requests/sec</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, whoami on Cilium netkit, 2026</text>
</svg>
</figure>

4.8x. I didn't trust that number until I isolated it: swapping which side (client or server) ran in a container, holding the other constant, showed client location barely moved the result: the win is entirely server-side. `mpstat` during the load test confirmed why:

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="cpu-time-per-http-request-ms-title cpu-time-per-http-request-ms-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="cpu-time-per-http-request-ms-title">CPU time per HTTP request (ms)</title>
  <desc id="cpu-time-per-http-request-ms-desc">CPU time per HTTP request (ms). horizontal bar data: Host (plain socket) 1.1; Container (netkit) 0.23.Source: Author benchmark, mpstat under wrk load, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">CPU time per HTTP request (ms)</text>

  <line x1="145" y1="298.0" x2="510" y2="298.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="243.0" x2="510" y2="243.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="188.0" x2="510" y2="188.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="133.0" x2="510" y2="133.0" stroke="currentColor" opacity="0.08" />
<line x1="145" y1="78.0" x2="510" y2="78.0" stroke="currentColor" opacity="0.08" />
<text x="136.0" y="146.9" text-anchor="end" font-size="11" fill="currentColor" opacity="0.8"><tspan x="136.0" dy="0">Host (plain</tspan><tspan x="136.0" dy="12">socket)</tspan></text>
<rect x="145" y="78.0" width="365.0" height="106.0" rx="4" fill="#f97316" />
<text x="518.0" y="146.9" font-size="11" fill="currentColor">1.1</text>
<text x="136.0" y="260.9" text-anchor="end" font-size="11" fill="currentColor" opacity="0.8"><tspan x="136.0" dy="0">Container</tspan><tspan x="136.0" dy="12">(netkit)</tspan></text>
<rect x="145" y="192.0" width="76.3" height="106.0" rx="4" fill="#38bdf8" />
<text x="229.3" y="260.9" font-size="11" fill="currentColor">0.23</text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, mpstat under wrk load, 2026</text>
</svg>
</figure>

The `mpstat` summary lines behind that chart, captured on `worker-01` (4 vCPUs) while `wrk` was running:

```
# server = host (whoami-host, plain socket), 3,014 req/s
Average:     CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle
Average:     all   23.20    0.00   57.31    0.00    0.00    2.26    0.00    0.00    0.00   17.06

# server = container (whoami via Service, netkit), 15,588 req/s
Average:     CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle
Average:     all   25.64    0.00   49.02    0.00    0.00   14.12    0.00    0.00    0.00   10.88
```

Converting busy-% into CPU-time per request: `4 × (1 − 0.1706) = 3.32` CPUs busy ÷ 3,014 req/s = **1.10ms** on the host; `4 × (1 − 0.1088) = 3.56` CPUs busy ÷ 15,588 req/s = **0.23ms** in the container. 4.81x more CPU per request on the host path, almost exactly matching the throughput gap. The mechanism, per Isovalent's own writeup on netkit[^netkit-isovalent]: pod egress traffic is redirected straight to the physical device from a BPF program running on the netkit device itself, **skipping the per-CPU backlog queue** (the same queue a `veth`-based pod would have to cross via a full namespace hop), plus `netkit` defaults to L3 (L2 is a supported option, just not the default), removing ARP overhead in that default mode. That's the documented source of the saving; I was wrong to reach for virtio and ring buffers, which nothing in netkit's own design touches.

Worth being precise about what Isovalent's own numbers actually claim, though: their stated goal, and result, is netkit reaching **parity** with host networking, not beating it. ByteDance reported netkit giving a 12% CPS increase over `veth` in production; Meta found netkit's softirq load on live traffic "indistinguishable from host." Both are netkit closing a gap *down to* host, not a container process outrunning one. My result is a different, larger claim: container beating bare host outright, by 4.8x, under load, and I don't have a fully satisfying resolution for the size of that gap beyond the likely candidate: my host-side baseline is a plain `systemd`-managed process with nothing skipped and none of Cilium's other host-side optimizations (eBPF kube-proxy replacement, host routing) in play, since it isn't running through Cilium at all, closer to the "veth-era" baseline these vendor numbers are measured *against* than to the tuned host baseline they're measured *to*.

That last point is a confound I haven't isolated yet, and I want to flag it plainly rather than let it hide inside the 4.8x number: this benchmark compares a plain host process to a container behind Cilium's full eBPF stack, not to a host running Cilium's eBPF host-side optimizations (kube-proxy replacement, host routing) with no `netkit` involved. So the 4.8x figure is really "container + netkit + Cilium's eBPF host optimizations" versus "unmodified host," and I can't yet tell how much of the gap is `netkit`'s dispatch-path saving specifically versus Cilium's other eBPF machinery doing work on the host's behalf that a plain `systemd` process never gets. Separating those two would mean running the host side with `cilium-agent`'s host-side eBPF programs attached (host-reachable services, host routing) but no container, no `netkit`, no pod netns, an eBPF-on-host-only baseline in between the two I actually tested. That's a follow-up worth doing before treating 4.8x as netkit's number specifically rather than Cilium's number generally.

A 3-node etcd Raft cluster (the real quorum shape, 2-of-3 majority) showed the same underlying mechanism: a quorum-committing `PUT` sends `AppendEntries` to two followers and waits for the faster one to ack: trivial per-request CPU work, cost dominated by dispatch.

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="etcd-raft-commit-latency-2-node-vs-3-node-quorum-ms-title etcd-raft-commit-latency-2-node-vs-3-node-quorum-ms-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="etcd-raft-commit-latency-2-node-vs-3-node-quorum-ms-title">etcd Raft commit latency (ms)</title>
  <desc id="etcd-raft-commit-latency-2-node-vs-3-node-quorum-ms-desc">etcd Raft commit latency (ms). grouped bar data: 3-node: Host 18.4, Container 12.44; 2-node (reference, degenerate unanimity): Host 16.8, Container 12.58.Source: Author benchmark, 100 sequential quorum-committing PUTs, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">etcd Raft commit latency (ms)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="159.0" y="92.0" width="16.0" height="185.0" fill="#f97316" />
<text x="167.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">18.40</text>
<rect x="177.0" y="151.9" width="16.0" height="125.1" fill="#38bdf8" />
<text x="185.0" y="147.9" text-anchor="middle" font-size="9" fill="currentColor">12.44</text>
<text x="177.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="177.0" dy="0">3-node</tspan></text>
<rect x="369.0" y="108.1" width="16.0" height="168.9" fill="#f97316" />
<text x="377.0" y="104.1" text-anchor="middle" font-size="9" fill="currentColor">16.80</text>
<rect x="387.0" y="150.5" width="16.0" height="126.5" fill="#38bdf8" />
<text x="395.0" y="146.5" text-anchor="middle" font-size="9" fill="currentColor">12.58</text>
<text x="387.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="387.0" dy="0">2-node (ref.)</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, 100 sequential quorum-committing PUTs, 2026</text>
</svg>
</figure>

Container: 12.44ms average. Host: 18.40ms. **1.48x**, same dispatch-cost mechanism as `wrk`. The 2-node number alongside it isn't a second valid data point, it's what the same test showed before it was actually testing a majority vote: at 2 nodes "quorum" means both nodes must ack, so the leader isn't relaxing anything by only needing 2-of-3. Worth noting the direction: the host got *slower* at 3 nodes (16.80ms to 18.40ms), not faster, because the leader now fans out `AppendEntries` to two followers instead of one, and that extra packet's dispatch cost lands harder on the host's more expensive per-packet path. The container barely moved.

The north-south data point is the odd one out, and worth stating plainly rather than forcing it into the concurrency story: 30 *serial* plain-HTTP requests from `control-plane-01`, no DNS, no TLS, one at a time, and the container path still won, 1.379ms average versus 1.894ms for the host, ~27% lower, with zero concurrency in play. That doesn't fit "needs concurrent load" the way the MPI result later in this post does, and I'm not claiming it does. The likely reason is architectural, not load-related: the container path here goes through a single cheap eBPF socket-layer redirect on the *client's own kernel*, not a full pod-to-pod round trip, a different mechanism from the "many requests queued on a netkit device" story above. Filed as a real result, not a proven mechanism.

## Where the bare host wins: work netkit can't touch

Redis, `-c 20 -n 100000`:

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="redis-throughput-host-vs-container-title redis-throughput-host-vs-container-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="redis-throughput-host-vs-container-title">Redis throughput, host vs container</title>
  <desc id="redis-throughput-host-vs-container-desc">Redis throughput, host vs container. grouped bar data: SET: Host 39017, Container 35174; GET: Host 38447, Container 32248.Source: Author benchmark, redis-benchmark -c 20 -n 100000, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">Redis throughput, host vs container</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="159.0" y="92.0" width="16.0" height="185.0" fill="#f97316" />
<text x="167.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">39,017</text>
<rect x="177.0" y="110.2" width="16.0" height="166.8" fill="#38bdf8" />
<text x="185.0" y="106.2" text-anchor="middle" font-size="9" fill="currentColor">35,174</text>
<text x="177.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="177.0" dy="0">SET</tspan></text>
<rect x="369.0" y="94.7" width="16.0" height="182.3" fill="#f97316" />
<text x="377.0" y="90.7" text-anchor="middle" font-size="9" fill="currentColor">38,447</text>
<rect x="387.0" y="124.1" width="16.0" height="152.9" fill="#38bdf8" />
<text x="395.0" y="120.1" text-anchor="middle" font-size="9" fill="currentColor">32,248</text>
<text x="387.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="387.0" dy="0">GET</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, redis-benchmark -c 20 -n 100000, 2026</text>
</svg>
</figure>

Host ahead by 11-19% (SET 39,017 vs 35,174, GET 38,447 vs 32,248). Redis's single-threaded command loop is the bottleneck here, not the network: one core processes `SET`/`GET` sequentially no matter which datapath delivered the packet, so netkit's CPU saving has nothing to attach to. Whatever small overhead the container path adds on top now shows through directly as lost throughput, instead of being hidden behind a bigger saving elsewhere.

PostgreSQL tells the same story, with a twist. Standalone `pgbench` (async commit, scale factor 2; TPC-B row-locks a single `pgbench_branches`/`pgbench_tellers` row per transaction, a server-side serialization bottleneck), 2-node cluster since standalone throughput isn't a quorum operation and cluster size doesn't apply to it:

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="postgresql-throughput-standalone-vs-sync-replication-tps-title postgresql-throughput-standalone-vs-sync-replication-tps-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="postgresql-throughput-standalone-vs-sync-replication-tps-title">PostgreSQL 2-node throughput (TPS)</title>
  <desc id="postgresql-throughput-standalone-vs-sync-replication-tps-desc">PostgreSQL 2-node throughput, standalone vs sync replication (TPS). grouped bar data: Standalone: Host 739.4, Container 572.3; Sync replication: Host 579.8, Container 568.6.Source: Author benchmark, pgbench -c 20 -T 30, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">PostgreSQL 2-node throughput (TPS)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="159.0" y="92.0" width="16.0" height="185.0" fill="#f97316" />
<text x="167.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">739.4</text>
<rect x="177.0" y="133.8" width="16.0" height="143.2" fill="#38bdf8" />
<text x="185.0" y="129.8" text-anchor="middle" font-size="9" fill="currentColor">572.3</text>
<text x="177.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="177.0" dy="0">Standalone</tspan></text>
<rect x="369.0" y="131.9" width="16.0" height="145.1" fill="#f97316" />
<text x="377.0" y="127.9" text-anchor="middle" font-size="9" fill="currentColor">579.8</text>
<rect x="387.0" y="134.7" width="16.0" height="142.3" fill="#38bdf8" />
<text x="395.0" y="130.7" text-anchor="middle" font-size="9" fill="currentColor">568.6</text>
<text x="387.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="387.0" dy="0">Sync</tspan><tspan x="387.0" dy="11">replication</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, pgbench -c 20 -T 30, 2026</text>
</svg>
</figure>

Standalone: host ahead 29% (739 vs 572 TPS), same lock-contention story as Redis. That part doesn't involve a quorum vote at all, so cluster size doesn't change it. Turning synchronous replication on does, and this is where cluster size matters: `synchronous_standby_names = 'ANY 1 (standby2, standby3)'` on a real 3-node cluster (primary plus two standbys) waits for whichever standby acks first, not the one-and-only standby a 2-node setup is stuck with. That's the shape a production quorum-commit deployment actually runs.

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="postgresql-marginal-cost-of-synchronous-replication-title postgresql-marginal-cost-of-synchronous-replication-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="postgresql-marginal-cost-of-synchronous-replication-title">PostgreSQL: cost of sync replication (%)</title>
  <desc id="postgresql-marginal-cost-of-synchronous-replication-desc">PostgreSQL: cost of sync replication (%). grouped bar data: 3-node: Host 0.5, Container 2.7; 2-node (reference, degenerate unanimity): Host 21.6, Container 0.6.Source: Author benchmark, pgbench -c 20 -T 30, standalone vs sync-commit throughput delta, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">PostgreSQL: cost of sync replication (%)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="159.0" y="272.7" width="16.0" height="4.3" fill="#f97316" />
<text x="167.0" y="268.7" text-anchor="middle" font-size="9" fill="currentColor">0.5%</text>
<rect x="177.0" y="253.9" width="16.0" height="23.1" fill="#38bdf8" />
<text x="185.0" y="249.9" text-anchor="middle" font-size="9" fill="currentColor">2.7%</text>
<text x="177.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="177.0" dy="0">3-node</tspan></text>
<rect x="369.0" y="92.0" width="16.0" height="185.0" fill="#f97316" />
<text x="377.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">21.6%</text>
<rect x="387.0" y="271.9" width="16.0" height="5.1" fill="#38bdf8" />
<text x="395.0" y="267.9" text-anchor="middle" font-size="9" fill="currentColor">0.6%</text>
<text x="387.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="387.0" dy="0">2-node (ref.)</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, pgbench -c 20 -T 30, standalone vs sync-commit throughput delta, 2026</text>
</svg>
</figure>

At a real quorum, turning synchronous replication on cost the host **0.5%** throughput. Essentially free. The 2-node reference bar (21.6%) isn't a smaller version of the same fact, it's a different regime entirely: at 2 nodes there's no majority to wait for, only unanimity, so every commit pays the full round-trip cost with no faster-of-two option. The container's marginal cost stayed near zero either way (0.6% at 2 nodes, 2.7% at 3), because its per-packet dispatch cost was already cheap enough that a full round trip barely registered in the first place. Sequential commit latency (confound-controlled, same Alpine/musl `psql` client both sides) tells the same story at smaller scale: container 14.24ms vs host 20.86ms at 3 nodes, a 1.46x gap, down from 1.61x at 2 nodes as the host caught up.

## Kafka: the exception that isn't

Same "wait for the other replica" shape as Raft and Postgres, on paper. Kafka answered differently:

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="kafka-produce-latency-leader-ack-vs-quorum-write-ms-title kafka-produce-latency-leader-ack-vs-quorum-write-ms-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="kafka-produce-latency-leader-ack-vs-quorum-write-ms-title">Kafka produce latency: leader-ack vs quorum write (ms)</title>
  <desc id="kafka-produce-latency-leader-ack-vs-quorum-write-ms-desc">Kafka produce latency: leader-ack vs quorum write (ms). grouped bar data: acks=1: Host 40.26, Container 25.55; acks=all: Host 259.9, Container 350.11.Source: Author benchmark, kafka-producer-perf-test, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">Kafka produce latency: leader-ack vs quorum write (ms)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="159.0" y="255.7" width="16.0" height="21.3" fill="#f97316" />
<text x="167.0" y="251.7" text-anchor="middle" font-size="9" fill="currentColor">40.3</text>
<rect x="177.0" y="263.5" width="16.0" height="13.5" fill="#38bdf8" />
<text x="185.0" y="259.5" text-anchor="middle" font-size="9" fill="currentColor">25.6</text>
<text x="177.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="177.0" dy="0">acks=1</tspan></text>
<rect x="369.0" y="139.7" width="16.0" height="137.3" fill="#f97316" />
<text x="377.0" y="135.7" text-anchor="middle" font-size="9" fill="currentColor">259.9</text>
<rect x="387.0" y="92.0" width="16.0" height="185.0" fill="#38bdf8" />
<text x="395.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">350.1</text>
<text x="387.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="387.0" dy="0">acks=all</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, kafka-producer-perf-test, 2026</text>
</svg>
</figure>

`acks=1` (leader-ack only, not a quorum vote, so cluster size shouldn't matter here) fits the pattern: container 1.6x faster at 2 nodes. `acks=all` + `min.insync.replicas=2` is the genuine quorum write, and that's the number worth getting from the real 3-node majority rather than the 2-node degenerate case:

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="kafka-acks-all-latency-2-node-vs-3-node-quorum-ms-title kafka-acks-all-latency-2-node-vs-3-node-quorum-ms-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="kafka-acks-all-latency-2-node-vs-3-node-quorum-ms-title">Kafka acks=all latency (ms)</title>
  <desc id="kafka-acks-all-latency-2-node-vs-3-node-quorum-ms-desc">Kafka acks=all latency (ms). grouped bar data: 3-node: Host 98.75, Container 374.39; 2-node (reference, degenerate unanimity): Host 259.9, Container 350.11.Source: Author benchmark, kafka-producer-perf-test acks=all, min.insync.replicas=2, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">Kafka acks=all latency (ms)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="159.0" y="228.2" width="16.0" height="48.8" fill="#f97316" />
<text x="167.0" y="224.2" text-anchor="middle" font-size="9" fill="currentColor">98.75</text>
<rect x="177.0" y="92.0" width="16.0" height="185.0" fill="#38bdf8" />
<text x="185.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">374.4</text>
<text x="177.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="177.0" dy="0">3-node</tspan></text>
<rect x="369.0" y="148.6" width="16.0" height="128.4" fill="#f97316" />
<text x="377.0" y="144.6" text-anchor="middle" font-size="9" fill="currentColor">259.9</text>
<rect x="387.0" y="104.0" width="16.0" height="173.0" fill="#38bdf8" />
<text x="395.0" y="100.0" text-anchor="middle" font-size="9" fill="currentColor">350.1</text>
<text x="387.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="387.0" dy="0">2-node (ref.)</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, kafka-producer-perf-test acks=all, min.insync.replicas=2, 2026</text>
</svg>
</figure>

At a real 3-node quorum, host `acks=all` latency dropped by more than half versus the 2-node reference (259.90ms to 98.75ms): the cleanest confirmation in this whole post of "race the faster of two independent followers instead of waiting on the only one." The likely reason it shows up here at all: Raft's `AppendEntries` and Postgres's WAL streaming are the leader *pushing* to a follower and waiting for one ack, a single dispatch round trip. Kafka's followers instead run a continuous *fetch* loop against the leader (`replica.fetch.wait.max.ms`, default 500ms); `acks=all` completion depends on that poll loop noticing the new record, not a dedicated round trip triggered by the write. At 2 nodes that poll loop has no alternative to race against; at 3 it does, and the win is large. Not independently confirmed with the same CPU-profiling rigor as the `wrk` result, so I'm stating it as the likely mechanism, not a proven one.

One number that doesn't fit any of this: `acks=1` on the container side got markedly *worse* at 3 nodes (25.55ms to 69.08ms avg, throughput roughly flat), despite `acks=1` not being a quorum operation at all. Flagged as an open anomaly rather than forced into a story I haven't verified.

## MPI: no concurrency, no advantage

A real 2-rank MPI ping-pong (`mpicc`-compiled, launched through a hand-rolled Slurm cluster, mirrored topology on host and Kubernetes), as close to a raw TCP round trip as this series gets. Same layout both sides, deliberately, and with the controller kept off the compute path so its numbers aren't diluted by controller overhead: one node runs `slurmctld` only, the other two run `slurmd` and actually carry the two MPI ranks. `sbatch` on the controller node wraps `mpirun` to launch the two ranks over `ssh`.

```
HOST (bare processes)
  worker-01: slurmctld only, sbatch submitted here
      |
      |  mpirun, ssh launcher, private network
      v
  worker-02: slurmd, MPI rank 0  <--->  worker-03: slurmd, MPI rank 1
```

```
CONTAINER (same topology, as pods)
  slurm-ctl pod (worker-01): slurmctld only, sbatch submitted here
      |
      |  mpirun, ssh launcher, Cilium netkit
      v
  slurm-worker pod (worker-02): slurmd, rank 0  <--->  slurm-worker-2 pod (worker-03): slurmd, rank 1
```

No `hostNetwork` on the container side here either. `munge` (shared auth key) and `sshd` are installed identically on both container pods purely so the launch mechanism matches the host's `ssh`-based `mpirun` launcher exactly, not to introduce a variable that isn't on the host side too.

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="mpi-ping-pong-round-trip-latency-s-title mpi-ping-pong-round-trip-latency-s-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="mpi-ping-pong-round-trip-latency-s-title">MPI ping-pong round-trip latency (µs)</title>
  <desc id="mpi-ping-pong-round-trip-latency-s-desc">MPI ping-pong round-trip latency (µs). grouped bar data: 0-byte round trip: Host 122.74, Container 131.37.Source: Author benchmark, 2-rank MPI ping-pong over Slurm, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">MPI ping-pong round-trip latency (µs)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="264.0" y="104.1" width="16.0" height="172.9" fill="#f97316" />
<text x="272.0" y="100.1" text-anchor="middle" font-size="9" fill="currentColor">122.7</text>
<rect x="282.0" y="92.0" width="16.0" height="185.0" fill="#38bdf8" />
<text x="290.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">131.4</text>
<text x="282.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="282.0" dy="0">0-byte round</tspan><tspan x="282.0" dy="11">trip</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, 2-rank MPI ping-pong over Slurm, 2026</text>
</svg>
</figure>

<figure class="blog-chart">
<svg viewBox="0 0 560 380" style="max-width: 100%; height: auto; font-family: 'Inter', system-ui, sans-serif; --chart-muted: #4b5563;" role="img" aria-labelledby="mpi-ping-pong-bandwidth-mb-s-title mpi-ping-pong-bandwidth-mb-s-desc">
  <style>
    @media (prefers-color-scheme: dark) { svg { --chart-muted: #d1d5db; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
  </style>
  <title id="mpi-ping-pong-bandwidth-mb-s-title">MPI ping-pong 64KB bandwidth (MB/s)</title>
  <desc id="mpi-ping-pong-bandwidth-mb-s-desc">MPI ping-pong 64KB bandwidth (MB/s). grouped bar data: 64KB round trip: Host 175.64, Container 142.27.Source: Author benchmark, 2-rank MPI ping-pong over Slurm, 2026 .</desc>
  <text x="280.0" y="29" text-anchor="middle" font-size="18" font-weight="800" fill="currentColor">MPI ping-pong 64KB bandwidth (MB/s)</text>

  <line x1="72" y1="277.0" x2="492" y2="277.0" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="230.8" x2="492" y2="230.8" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="184.5" x2="492" y2="184.5" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="138.2" x2="492" y2="138.2" stroke="currentColor" opacity="0.08" />
<line x1="72" y1="92.0" x2="492" y2="92.0" stroke="currentColor" opacity="0.08" />
<rect x="72" y="62" width="10" height="10" fill="#f97316" />
<text x="87" y="71" font-size="11" fill="currentColor">Host</text>
<rect x="177" y="62" width="10" height="10" fill="#38bdf8" />
<text x="192" y="71" font-size="11" fill="currentColor">Container</text>
<rect x="264.0" y="92.0" width="16.0" height="185.0" fill="#f97316" />
<text x="272.0" y="88.0" text-anchor="middle" font-size="9" fill="currentColor">175.6</text>
<rect x="282.0" y="127.2" width="16.0" height="149.8" fill="#38bdf8" />
<text x="290.0" y="123.2" text-anchor="middle" font-size="9" fill="currentColor">142.3</text>
<text x="282.0" y="295.0" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8"><tspan x="282.0" dy="0">64KB round</tspan><tspan x="282.0" dy="11">trip</tspan></text>
  <text x="280.0" y="366" text-anchor="middle" font-size="10" fill="var(--chart-muted, currentColor)">Source: Author benchmark, 2-rank MPI ping-pong over Slurm, 2026</text>
</svg>
</figure>

Host wins latency by ~7% and a 64KB synchronous-exchange throughput test by ~19% (176 vs 142 MB/s). This is the cleanest confirmation of what the `wrk` result actually proved: netkit's advantage is CPU time *saved per request under concurrent load*: many requests stacked up, cheaper dispatch paid off across all of them. A strict 1:1 ping-pong has exactly one message in flight at a time. There's no queue for a cheaper datapath to save CPU across, so the result looks like the earlier `ping`/`iperf3` tests: dominated by the raw path, not by dispatch efficiency.

> **A note on RDMA**, since MPI is what most people associate with it: this test measured plain TCP MPI, the only option on this instance type, which has no InfiniBand or RoCE capability. Real HPC and large-scale AI training clusters typically run MPI over RDMA, and RDMA changes the picture completely, not just the numbers. RDMA is kernel-bypass by design: userspace `libibverbs` talks straight to NIC hardware queue pairs, with no syscalls, no socket, no netfilter, and critically, no `netkit` and no host network namespace either. Neither side of this comparison, host or container, touches the kernel networking stack RDMA is built to avoid. None of this post's reasoning (dispatch cost, concurrency, `%soft` CPU time) applies to an RDMA path, because none of that machinery is in an RDMA path to begin with. Whether Cilium/netkit adds overhead to RDMA traffic is a different, harder question this test doesn't answer. It needs SR-IOV or RDMA-aware device plugins to even test, and this cluster's hardware doesn't have them.

## The rule, sharpened

> For east-west traffic: container/netkit wins when a request's cost is dominated by network dispatch **under concurrent load**. Host wins, or it's a wash, when the cost is dominated by CPU/lock serialization the datapath can't touch, or when there's no concurrency for a cheaper datapath to save CPU across in the first place.

Every east-west result here fits that once you check for the two things that break a naive read of it: whether the "container wins" is actually just a faster underlying network path (the cloud provider's private vs. public network, nothing to do with Cilium), and whether "dispatch-bound" work has enough concurrent volume behind it for the saving to compound. Kafka's pull-based replication and a synchronous MPI ping-pong both look dispatch-bound on paper and both behave differently: not because the rule is wrong, but because concurrency and protocol shape turned out to matter as much as "does this workload talk to the network a lot."

North-south doesn't fit that rule, and I'm not stretching it to; it's answering a different question (external client through a service redirect, not two matched endpoints on the same kind of path) with a different likely mechanism (a cheap client-side eBPF hook, not a busy datapath saving CPU under load). Keep the two traffic shapes separate when you're reasoning about your own workload; conflating them is exactly how a naive read goes wrong.

If you're deciding whether to worry about CNI overhead for a specific workload, the question that actually predicts the answer isn't "is this network-heavy"; it's "does the load stack up requests fast enough for a cheaper per-packet datapath to matter." A quorum-writing database under real concurrent traffic: probably fine, possibly faster. A single long-running batch job doing one synchronous exchange at a time: measure it, don't assume.

One more axis sits underneath all of this, orthogonal to it: quorum size. Dispatch cost decides *which side wins*. Quorum size decides *how much replication cost there is to begin with*, and a real majority (2-of-3) has less of it to pay than a degenerate 2-node "quorum" that's actually unanimity. Shrinking that cost helps whichever side was paying more of it, which for Postgres and Kafka was the host, dramatically. Not a universal law though: etcd's host path got outright slower at 3 nodes, not just less advantaged, because fanning `AppendEntries` out to a second follower is a real, separate tax that doesn't always net out in the host's favor. If you're running a 2-node database or coordination service in production for cost reasons, know that you're paying the unanimity tax, not a quorum tax, and that tax usually (not always) lands harder on bare-host deployments than on netkit ones.

**Key takeaways**

- For east-west traffic (pod-to-pod, most of this post), Cilium's `netkit` datapath beats a bare host socket specifically when a service handles many concurrent requests (proven via CPU profiling, not assumed).
- It loses, or ties, when the bottleneck is server-side: a single-threaded command loop (Redis), row-lock contention (PostgreSQL), or a strictly synchronous 1:1 exchange (MPI) with no concurrency for a cheaper datapath to save CPU across.
- Two east-west results looked like exceptions until traced to mechanism: Kafka's pull-based replication and a raw MPI ping-pong both behave differently from Raft- and WAL-style "wait for one ack" protocols.
- North-south (external client to service) is a different traffic shape entirely: container still wins there, but for a different reason, with zero concurrency involved.
- Quorum size is a separate axis from all of the above: a genuine 3-node majority (not 2-node unanimity) shrank the host's replication cost dramatically for Postgres and Kafka, narrowing or reversing the gap, because dispatch cost decides which side wins while quorum size decides how much cost there is to shrink. etcd is the exception, not the rule.

## References

- [Cilium netkit: The Final Frontier in Container Networking Performance](https://isovalent.com/blog/post/cilium-netkit-a-new-container-networking-paradigm-for-the-ai-era/). Nico Vibert, Isovalent, 2024. The primary source on netkit's design and the mechanism behind its performance (per-CPU backlog queue skip, L3-by-default), plus the ByteDance and Meta production data points cited above.
- [Introduction to Linux netkit interfaces with a grain of eBPF](https://blog.yadutaf.fr/2025/07/01/introduction-to-linux-netkit-interfaces-with-a-grain-of-ebpf/). A clear walkthrough of netkit's primary/peer device model and its `netkit/primary` and `netkit/peer` eBPF hooks.
- [Creating a Yogurt Phone with netkit eBPF](https://blog.yadutaf.fr/2025/09/16/creating-a-yogurt-phone-with-netkit-ebpf). A deeper look at cross-namespace routing with netkit, including the `netkit_xnet()` metadata-scrubbing step and why `bpf_redirect()` is needed in place of `veth`'s `bpf_redirect_peer()`.

[^netkit-isovalent]: Nico Vibert, ["Cilium netkit: The Final Frontier in Container Networking Performance"](https://isovalent.com/blog/post/cilium-netkit-a-new-container-networking-paradigm-for-the-ai-era/), Isovalent, July 2024 (updated July 2024).
