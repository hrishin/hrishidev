---
layout: post
title:  "How Linux Kernel implements generic linked list: part 2"
date:   2025-07-04 10:12:12 +0000
categories: linked list, Linux, Linux kernel
---

In [Part 1](https://hrishi.dev/linked/list,/linux,/linux/kernel/list/2020/11/20/generic-linkedin-list-c.html), we explored the 
motivation and building blocks behind the Linux kernel’s generic circular doubly linked list. 
Now, let’s implement the core routines that make this data structure so powerful and reusable.

### The Core List Routines

The Linux kernel’s list API provides a set of macros and functions to manipulate lists generically. 
The most important routines are:

- **INIT_LIST_HEAD**: Initialize a list head.
- **list_add**: Add a new entry after the specified head.
- **list_add_tail**: Add a new entry before the specified head (at the tail).
- **list_del**: Remove an entry from the list.
- **list_empty**: Check if the list is empty.
- **list_for_each**: Iterate over the list.

Let’s implement these in C, using the `list_head` structure from Part 1.

---

### 1. List Head Initialization

```c
struct list_head {
    struct list_head *next, *prev;
};

#define INIT_LIST_HEAD(ptr) do { \
    (ptr)->next = (ptr); (ptr)->prev = (ptr); \
} while (0)
```

This macro sets up a list head so that both `next` and `prev` point to itself, representing an empty list.

---

### 2. Adding Entries

#### Add After Head (like `push_front`)

```c
static inline void list_add(struct list_head *new, struct list_head *head) {
    new->next = head->next;
    new->prev = head;
    head->next->prev = new;
    head->next = new;
}
```

#### Add Before Head (like `push_back`)

```c
static inline void list_add_tail(struct list_head *new, struct list_head *head) {
    new->next = head;
    new->prev = head->prev;
    head->prev->next = new;
    head->prev = new;
}
```

---

### 3. Deleting Entries

```c
static inline void list_del(struct list_head *entry) {
    entry->prev->next = entry->next;
    entry->next->prev = entry->prev;
    entry->next = entry->prev = NULL; // Optional: helps catch bugs
}
```

---

### 4. Checking if List is Empty

```c
static inline int list_empty(const struct list_head *head) {
    return head->next == head;
}
```

---

### 5. Iterating Over the List

```c
#define list_for_each(pos, head) \
    for (pos = (head)->next; pos != (head); pos = pos->next)
```

To get the parent structure from a `list_head *`, use the `CONTAINER_OF` macro from Part 1.

---

### 6. Example: Using the List

Let’s put it all together with a simple example:

```c
#include <stdio.h>
#include <stdlib.h>

struct list_head {
    struct list_head *next, *prev;
};

#define INIT_LIST_HEAD(ptr) do { \
    (ptr)->next = (ptr); (ptr)->prev = (ptr); \
} while (0)

static inline void list_add(struct list_head *new, struct list_head *head) {
    new->next = head->next;
    new->prev = head;
    head->next->prev = new;
    head->next = new;
}

static inline void list_add_tail(struct list_head *new, struct list_head *head) {
    new->next = head;
    new->prev = head->prev;
    head->prev->next = new;
    head->prev = new;
}

static inline void list_del(struct list_head *entry) {
    entry->prev->next = entry->next;
    entry->next->prev = entry->prev;
    entry->next = entry->prev = NULL;
}

static inline int list_empty(const struct list_head *head) {
    return head->next == head;
}

#define OFFSET_OF(type, member) ((size_t) &(((type *)0)->member))
#define CONTAINER_OF(ptr, type, member) \
    ((type *)((char *)(ptr) - OFFSET_OF(type, member)))

#define list_for_each(pos, head) \
    for (pos = (head)->next; pos != (head); pos = pos->next)

struct task_t {
    int     pid;
    struct  list_head tasks;
};

int main() {
    struct list_head task_list;
    INIT_LIST_HEAD(&task_list);

    // Create and add tasks
    struct task_t *t1 = malloc(sizeof(struct task_t));
    t1->pid = 1;
    list_add(&t1->tasks, &task_list);

    struct task_t *t2 = malloc(sizeof(struct task_t));
    t2->pid = 2;
    list_add_tail(&t2->tasks, &task_list);

    // Iterate and print
    struct list_head *pos;
    printf("Task list:\n");
    list_for_each(pos, &task_list) {
        struct task_t *task = CONTAINER_OF(pos, struct task_t, tasks);
        printf("PID: %d\n", task->pid);
    }

    // Delete all tasks
    while (!list_empty(&task_list)) {
        struct task_t *task = CONTAINER_OF(task_list.next, struct task_t, tasks);
        list_del(&task->tasks);
        free(task);
    }

    return EXIT_SUCCESS;
}
```

---

### Summary

- The Linux kernel’s generic linked list is built on a simple, type-agnostic `list_head` structure.
- Macros like `CONTAINER_OF` and `OFFSET_OF` allow you to recover the parent structure from a list node.
- The core routines (`INIT_LIST_HEAD`, `list_add`, `list_add_tail`, `list_del`, `list_empty`, and `list_for_each`) make it easy to build, traverse, and manipulate lists of any type.

In Part 3, we’ll see how these routines are used in real Linux kernel subsystems, such as process scheduling and device drivers.