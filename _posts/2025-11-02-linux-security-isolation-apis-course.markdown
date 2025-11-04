---
layout: post
title:  "Linux Security and Isolation APIs: course experience"
date:   2025-11-02 10:00:00 +0000
categories: [Linux, Security, Kernel, Isolation, APIs]
---

*A reflection on the [man7.org Linux Security and Isolation APIs course](https://man7.org/training/secisol/index.html) and the fascinating journey from time-sharing systems to modern containerization, how sasauge are made*

---

## Introduction

Recently, I had the opportunity to attend the **Linux Security and Isolation APIs** course offered by [man7.org Training](https://man7.org/training/secisol/index.html), delivered by Michael Kerrisk, author of *The Linux Programming Interface*. This four-day deep dive into the low-level Linux features that power containers, virtualization, and sandboxing was an eye-opening experience that connected the dots between historical computing concepts and modern cloud infrastructure.

What became particularly fascinating as we explored namespaces, cgroups, seccomp, and SELinux is how all these capabilities trace their roots back to fundamental concepts in time-sharing systems and isolation in the context of security. Modern containerization and virtualization are, in many ways, side effects or biproducts of solving these deeper security and resource isolation challenges.

## The Historical Context: From Time-Sharing to Containers

Before diving into the technical details, it's worth understanding the evolutionary path that led us here. The course beautifully connected the dots between:

1. **Time-sharing systems** - The need to run multiple users' processes on the same hardware securely
2. **Security isolation** - Ensuring one user's processes couldn't interfere with another's
3. **Resource control** - Fairly allocating CPU, memory, and I/O across users
4. **Modern containers** - Today's application of these same principles

This historical perspective makes it clear that containerization didn't emerge in a vacuum. It's the natural evolution of isolation mechanisms that UNIX and Linux have been refining for decades.

## The Evolution of Namespaces: Starting with Mount

One of the most interesting historical tidbits from the course was learning that the **first namespace—the mount namespace—came into existence in the early 2000s**. This wasn't created for containers (which didn't exist yet), but to solve a very specific problem: **on systems like UNIX/Linux, when users log in, PAM modules couldn't mount certain drives and make them available to specific sets of users.**

## Deep Dive: System Calls and Practical Implementation

The course provided extensive hands-on experience with the fundamental system calls that make namespaces work. Getting deep into `clone()`, `unshare()`, and `setns()` (nsenter behind) gave me a much better understanding of how container runtimes actually operate. The differences between `clone()` (creating a new process in new namespaces) and `unshare()` (moving the current process into new namespaces) became clear through practical exercises. Understanding `setns()` and how it allows processes to join existing namespaces was particularly enlightening—this is exactly what tools like `docker exec` do under the hood.

The hands-on experience writing seccomp BPF programs from scratch was challenging but incredibly valuable. While higher-level libraries like libseccomp make this easier, understanding the raw BPF syntax and how filters are constructed gave me deep insight into how system call filtering works at the kernel level. The course also covered working directly with the cgroupv2 API, which has a much cleaner unified hierarchy compared to v1.

Perhaps my biggest personal realization from the course was understanding a correlation I had previously missed: **how Linux capabilities govern and interact with namespace and cgroup operations**. This three-way relationship—capabilities acting as gatekeepers for what namespace and cgroup operations are permitted—is fundamental to container security but often overlooked. Understanding that user namespaces provide "fake capabilities" that allow namespace creation within that namespace, while still being restricted by the user namespace mapping, was a crucial insight. This correlation explained so much about why certain container configurations work or fail, and why rootless containers have different capabilities and limitations compared to rootful ones.

## Capabilities: From Simple Security Model to Sophisticated Governance

The course traced the evolution of Linux security from the traditional set-UID and set-GID model—which was a simplified security approach where processes either had full root privileges or didn't—to the modern capabilities system. Fundamentally, set-UID and set-GID programs simplified the security model, but they were all-or-nothing: a process either had root privileges or it didn't. The capabilities system broke this down into fine-grained privileges, allowing processes to have only the specific permissions they need.

One of the most enlightening aspects was learning how `execve()` transforms process capabilities according to Linux's capability transformation rules. When a process executes a new program via `execve()`, the kernel applies a complex set of rules based on the process's current capabilities, the file's capabilities (if any), and the bounding set. Understanding these transformation rules—how permitted, effective, and inheritable capability sets interact during exec—was crucial for understanding why some privilege escalation attempts work while others don't.

The course also revealed how cgroup v2 integrates with capabilities at a deeper level than I had realized. Cgroup v2 doesn't just limit resources; it also derives and can restrict capabilities for processes within a cgroup. This integration between cgroups and capabilities adds another layer of capability governance beyond what individual processes might have.

Playing around with system calls like `getresuid()`, `getresgid()`, `setcap()`, and `getcap()` during the practical exercises was not only enlightening but genuinely fun. These calls provide insight into the real vs effective user IDs, how capabilities are set and retrieved, and how they interact with the process's UID/GID context. Understanding the fundamental evolution of the Linux system to govern process capabilities in conjunction with namespaces and cgroups revealed just how sophisticated and well-integrated these mechanisms have become. It's a far cry from the simple binary root/non-root model, and seeing how all these pieces fit together was one of the most satisfying parts of the course.

## Personal Reflection

At last, personally I really enjoyed getting hands-on with the cgroup v2 API, exploring the various controllers including CPU, CPUSETS, freezer, and memory manager. Getting quite an in-depth understanding of the freezer controller for migrating workloads was particularly fascinating—understanding how processes can be frozen, migrated between systems, and then resumed opens up powerful possibilities for live migration and workload mobility that are fundamental to modern container orchestration systems.

## Course Information

**Course**: Linux Security and Isolation APIs (M7D-SECISOL02)  
**Provider**: [man7.org Training](https://man7.org/training/secisol/index.html)  
**Trainer**: Michael Kerrisk (author of *The Linux Programming Interface*)  
**Duration**: 4 days  
**Format**: Live online with extensive lab sessions  
**Materials**: 500+ page course book

For upcoming course dates and registration, visit the [man7.org training page](https://man7.org/training/secisol/index.html).

![Linux Security and Isolation APIs Course Certificate](/assets/linix_security_isolation_api_cert.png)


---

*This post reflects my experience and learnings from the course. For the most current course information, schedule, and pricing, please refer to the official [man7.org training website](https://man7.org/training/secisol/index.html).*


