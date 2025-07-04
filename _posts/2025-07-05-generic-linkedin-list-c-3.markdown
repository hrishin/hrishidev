---
layout: post
title:  "How Linux Kernel implements generic linked list: part 3"
date:   2025-07-04 10:12:12 +0000
categories: linked list, Linux, Linux kernel
---

A generic circular doubly linked list is not just a theoretical construct in the Linux kernel—it's a practical workhorse used in many core subsystems. In this final part, we'll see how the routines we implemented earlier are used in real Linux kernel code, focusing on process scheduling and device drivers.
Most importanly OS kernel as serve many routines, processes and with finite resrouces.
Thus performance and efficieny is really a
key matter for throughtput by optimising the common data strucures and routines.

## Recap

- [Part 1](./2020-11-20-generic-linkedin-list-c.markdown): We explored the motivation and building blocks of the Linux kernel's generic linked list.
- [Part 2](./2025-07-04-generic-linkedin-list-c-2.markdown): We implemented the essential list routines in C.
- Part 3 (this post): We'll see how these routines are used in real Linux kernel subsystems.

---

## 1. Process Scheduling: The Runqueue

One of the most important uses of the generic linked list is in the Linux scheduler's **runqueue**. The runqueue is a list of all processes that are ready to run on a CPU.

### Example: The `task_struct` and `run_list`

In the Linux kernel, each process is represented by a `task_struct`. Here's a simplified version:

```c
struct task_struct {
    // ... other fields ...
    struct list_head run_list;
    // ... other fields ...
};
```

The runqueue itself is just a list head:

```c
struct list_head runqueue;
```

When a process becomes runnable, it's added to the runqueue:

```c
list_add_tail(&p->run_list, &runqueue);
```

When the scheduler picks a process to run, it removes it from the runqueue:

```c
list_del(&p->run_list);
```

To iterate over all runnable processes:

```c
struct list_head *pos;
list_for_each(pos, &runqueue) {
    struct task_struct *p = list_entry(pos, struct task_struct, run_list);
    // ... do something with p ...
}
```

> **Note:** The kernel uses a macro `list_entry()` which is functionally equivalent to our `CONTAINER_OF`.

---

## 2. Device Drivers: Managing Devices

Device drivers often need to keep track of multiple devices of the same type. The generic list makes this easy.

### Example: Block Devices

Suppose you have a driver that manages several block devices. Each device structure embeds a `list_head`:

```c
struct my_block_device {
    // ... device fields ...
    struct list_head list;
};
```

The driver keeps a global list of all devices:

```c
struct list_head device_list;
INIT_LIST_HEAD(&device_list);
```

When a new device is registered:

```c
struct my_block_device *dev = kmalloc(...);
list_add_tail(&dev->list, &device_list);
```

To remove a device:

```c
list_del(&dev->list);
kfree(dev);
```

To iterate over all devices:

```c
struct list_head *pos;
list_for_each(pos, &device_list) {
    struct my_block_device *dev = list_entry(pos, struct my_block_device, list);
    // ... operate on dev ...
}
```

---

## 3. Buffer Caches and More

The generic list is also used in:

- **Buffer caches** (e.g., managing free/used buffers)
- **Networking** (e.g., lists of sockets, packets)
- **Filesystem structures** (e.g., superblocks, inodes)
- **Timers and deferred work**

The pattern is always the same: embed a `list_head` in your structure, and use the list routines to manage collections of those structures.

---

## 4. Why This Matters

How Can a Task Be in Multiple Lists at Once?

A powerful feature of the Linux kernel's generic linked list is that a single structure (like `task_struct`) can participate in multiple lists at the same time by embedding multiple `list_head` fields. This is especially important in process scheduling and other subsystems.

### The `task_struct` Example

A process in Linux is represented by a `task_struct`. This struct can have **multiple `list_head` fields**, each used for a different purpose:

```c
struct task_struct {
    // ... other fields ...
    struct list_head run_list;   // For the runqueue (processes ready to run)
    struct list_head sibling;    // For the list of children of a parent process
    // ... other fields ...
};
```

#### Why Multiple List Heads?

- **`run_list`**: Used to link the process into the scheduler's runqueue (all runnable tasks).
- **`sibling`**: Used to link the process into its parent's list of children.
- There could be others, e.g., for memory management, signal handling, etc.

#### How Does This Work in Practice?

Each `list_head` field is independent. When you add a `task_struct` to the runqueue, you use its `run_list`:

```c
list_add_tail(&p->run_list, &runqueue);
```

When you add the same `task_struct` to its parent's children list, you use its `sibling`:

```c
list_add_tail(&p->sibling, &parent->children);
```

**These two lists are completely independent**—the only thing they share is that the same `task_struct` is a member of both, via different embedded `list_head` fields.

#### How Does the Kernel Know Which List It's Traversing?

When traversing a list, the kernel always uses the correct `list_head` field and the correct macro:

```c
// Traversing the runqueue
list_for_each(pos, &runqueue) {
    struct task_struct *p = list_entry(pos, struct task_struct, run_list);
    // ... p is a runnable task ...
}

// Traversing a parent's children
list_for_each(pos, &parent->children) {
    struct task_struct *p = list_entry(pos, struct task_struct, sibling);
    // ... p is a child of parent ...
}
```

The `list_entry` macro (or `CONTAINER_OF`) tells the kernel which field to use to get back to the parent structure.

#### Why Is This Useful?

- **Performance and Efficiency:** This avoids the need for dynamically allocating wrapper nodes. It's memory efficient, cache friendly, and allows fast constant-time insert/remove operations.vNo extra allocations or indirection—just extra fields in the struct. 
Moving state of the process/task for involves manipulating the `head_list` pointers of the task, thats it really.

- **Type safety:** The list routines are type-agnostic but safe, thanks to macros like `CONTAINER_OF`.

- **Flexibility:** The same object can be part of many different lists, each for a different purpose.

- **Reusability:** The same routines work for any structure, anywhere in the kernel(fun fact, Linux implement most of these routines in macros).


#### Real Kernel Example

From [include/linux/sched.h](https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h):

```c
struct task_struct {
    // ...
    struct list_head tasks;         // global task list
    struct list_head children;      // list of children
    struct list_head thread_group;  // Threads in the same thread group
    struct list_head run_list;      // Run queue entry
    struct list_head cg_list;       // Cgroup list
    // ...
};
```

A process can be:
- In the global task list (`tasks`)
- In its parent's children list (`sibling`)
- In the runqueue (`run_list`)
- And more!


```ascii
[struct list_head: tasks] ----                  [struct task_struct: python]         [struct task_struct: app]
                              |                 +--------------------------+         +----------------------+
                              |                 | name: "python"           |         | name: "app"          |
                              |---------------->| tasks.next  ------------>|-------->| tasks.next           |
                              |<----------------| tasks.prev  <------------|<--------| tasks.prev           |
[struct list_head: run_list]------------------->| run_list.next ---------->|         |                      |
                           |<-------------------| run_list.prev <----------|         |                      |
+--------------------------+         +----------------------+       +----------------------+   +------------+
```
---

**Summary:**  
By embedding multiple `list_head` fields in a struct, the Linux kernel allows a single object (like a process) to participate in multiple, independent linked lists at the same time. Each list is managed separately, and the correct field is always used for each list operation.

---

## 5. Real Kernel References

- [Linux list.h (main header)](https://elixir.bootlin.com/linux/latest/source/include/linux/list.h)
- [Example: task_struct in the scheduler](https://elixir.bootlin.com/linux/latest/source/include/linux/sched.h)
- [Example: device list in block subsystem](https://elixir.bootlin.com/linux/latest/source/include/linux/genhd.h)

---

## Conclusion

The Linux kernel's generic linked list is a beautiful example of C's power: a simple, type-agnostic, and efficient data structure that underpins much of the kernel's flexibility and performance. By understanding and using these patterns, you can write more reusable and robust C code—kernel or user space.

**Thanks for reading this series! **
--- 