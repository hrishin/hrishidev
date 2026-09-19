---
layout: post
title:  "From Power Button to Shell Prompt: The Linux Boot Process, From Firmware to PXE and Metal³"
date:   2026-05-05 06:00:00 +0000
categories: [Linux, OS, Systems]
description: "A stage-by-stage walkthrough of the Linux boot process on x86-64: the reset vector, BIOS/UEFI, POST, GRUB2 and Secure Boot, kernel init, initramfs, systemd, and PAM-based login. Then the same stages on a diskless server: PXE, iPXE, DHCP relay, and bare-metal provisioning with Metal3, Ironic, and Cluster API."
image: /assets/os-boot-process-stages.png
redirect_from:
  - /linux,/os,/systems/2026/05/05/os-boot-process.html
---

*A stage-by-stage look at what happens from the moment you press the power button to the moment your shell is ready for input, then the same stages on a blank server that has to fetch its bootloader, kernel, and operating system over the network*

---

## Introduction

Pressing the power button on a modern computer triggers a carefully orchestrated sequence of events spanning
firmware, bootloaders, the kernel, and init systems. Each layer hands off to the next with increasing
sophistication.

This post walks through every stage of that journey on a modern x86-64 Linux system, from the first CPU instruction
executed out of reset to the login prompt waiting for your credentials. Part 2 then takes the same stages onto a
server with no operating system on its disk, where Stage 3 becomes a network conversation (PXE, DHCP, iPXE) and the
first Linux to boot is a RAM-only agent that installs the real one, using Metal³ and Ironic as the worked example.

![A seven-stage diagram of the Linux boot sequence: Firmware, POST, Bootloader, Kernel Init, initramfs, systemd, and Login/Shell](/assets/os-boot-process-stages.png)

## Table of Contents

1. [Stage 1: Reset Vector and Firmware (BIOS/UEFI)](#stage-1-reset-vector-and-firmware-biosuefi)
2. [Stage 2: POST (Power-On Self-Test)](#stage-2-post-power-on-self-test)
3. [Stage 3: Bootloader](#stage-3-bootloader)
4. [Stage 4: Kernel Initialization](#stage-4-kernel-initialization)
5. [Stage 5: initramfs (Early Userspace)](#stage-5-initramfs-early-userspace)
6. [Stage 6: Init System (systemd)](#stage-6-init-system-systemd)
7. [Stage 7: Login and Shell](#stage-7-login-and-shell)
8. [Putting It All Together](#putting-it-all-together)
9. [Part 2: Booting Without a Disk](#part-2-booting-without-a-disk)
10. [Stage 3, Over the Network: PXE, DHCP, TFTP, and iPXE](#stage-3-over-the-network-pxe-dhcp-tftp-and-ipxe)
11. [Crossing VLANs: The DHCP Relay](#crossing-vlans-the-dhcp-relay)
12. [The Metal3 Provisioning Sequence](#the-metal3-provisioning-sequence)
13. [The Metal3 Object Model](#the-metal3-object-model)
14. [Troubleshooting a Network Boot](#troubleshooting-a-network-boot)

---

## Stage 1: Reset Vector and Firmware (BIOS/UEFI)

### The First Instruction

When power is applied, the CPU does not start executing from RAM: it's empty at power-on. Instead, every
processor has a hardwired **reset vector** mapped by the chipset to a ROM chip on the motherboard containing
the firmware. The address is architecture-defined:

| Architecture | Reset vector |
|---|---|
| x86-64 | `0xFFFFFFF0`, 16 bytes below the top of 32-bit address space, entered in 16-bit real mode |
| ARM64 (AArch64) | Configured via the `RVBAR_EL3` register, entered at Exception Level 3 (EL3) |

On x86-64, the reset vector holds a `JMP` that transfers into the full firmware image. On ARM64, the SoC's
trusted firmware (e.g., ARM Trusted Firmware-A) runs first at EL3 before handing off to UEFI at EL2/EL1.

### BIOS vs. UEFI

**BIOS (Basic Input/Output System)** is the legacy firmware standard from the late 1970s. It operates in 16-bit
real mode and relies on a 512-byte **Master Boot Record (MBR)** at the start of the boot disk. The MBR contains
a first-stage bootloader and a partition table, all crammed into 512 bytes.

**UEFI (Unified Extensible Firmware Interface)**, formalized by the [UEFI Specification](https://uefi.org/specifications),
replaced BIOS and brings several critical improvements:

| Feature | BIOS | UEFI |
|---|---|---|
| Mode at startup | 16-bit real mode | 32/64-bit protected mode |
| Boot partition | MBR (512 bytes) | EFI System Partition (FAT32, megabytes) |
| Bootloader size | ~446 bytes | Full PE/COFF executables |
| Secure Boot | No | Yes |
| Network boot | Vendor extensions | Built-in PXE and HTTP boot |

UEFI firmware reads the **EFI System Partition (ESP)**, a FAT32 partition that contains bootloader executables
(`*.efi` files). The firmware itself understands filesystems, which is a significant leap over BIOS.

---

## Stage 2: POST (Power-On Self-Test)

Before handing off to a bootloader, the firmware runs **POST**, a series of hardware diagnostics:

1. **CPU test**: verify the processor is functioning correctly
2. **Memory initialization**: train and test DRAM, set up memory channels and timings
3. **Chipset initialization**: configure the PCH (Platform Controller Hub), PCIe lanes, clocks
4. **Device enumeration**: discover PCI/PCIe devices, assign I/O ports and memory-mapped I/O ranges
5. **Video initialization**: bring up a display so error messages can be shown
6. **Peripheral detection**: USB, SATA controllers, NVMe drives

The beep codes you may have heard from old machines are POST error signals. One long beep, for example, usually
means a memory failure. Modern UEFI systems display graphical error screens instead.

After POST, the firmware has a complete picture of the hardware and constructs the
**[ACPI tables](https://uefi.org/acpi)**, data structures that describe the hardware topology to the OS.

---

## Stage 3: Bootloader

### UEFI Path: The EFI Application

On a UEFI system, the firmware consults its NVRAM boot entries (managed with `efibootmgr`) to find an EFI
binary to execute. The ESP is a FAT32 partition; on a running Linux system it is mounted at `/boot/efi`, so
firmware-internal paths like `/EFI/ubuntu/shimaa64.efi` appear on disk as `/boot/efi/EFI/ubuntu/shimaa64.efi`.

<!-- [PERSONAL EXPERIENCE] -->
A typical Ubuntu ARM64 ESP, from a machine I booted and inspected while writing this post, looks like this:

```
/boot/efi/EFI/BOOT/BOOTAA64.EFI    ← removable-media fallback (copy of shim)
/boot/efi/EFI/ubuntu/
    shimaa64.efi                    ← NVRAM entry points here
    grubaa64.efi
    mmaa64.efi
    grub.cfg
    BOOTAA64.CSV
```

If NVRAM entries are wiped (firmware update, hardware reset), the firmware falls back to the well-known path
`/EFI/BOOT/BOOTAA64.EFI` (i.e. `/boot/efi/EFI/BOOT/BOOTAA64.EFI` on disk). On this machine that file is a
copy of `shimaa64.efi`.

`BOOTAA64.CSV` is a small text file that pairs a human-readable label with the path to the real bootloader:

```
shimaa64.efi,Ubuntu,,This is the boot entry for Ubuntu
```

The EFI `fallback` application reads this CSV to **re-register** the NVRAM boot entry pointing at
`shimaa64.efi` if it was lost. It's a self-healing mechanism so the system can boot again after a firmware
flash clears NVRAM.

<!-- [PERSONAL EXPERIENCE] -->
`efibootmgr` shows the boot configuration. This is the actual output from the same machine:

```bash
# efibootmgr -v        (device-path hex dumps trimmed)

BootCurrent: 0003
Timeout: 5 seconds
BootOrder: 0003,0000,0002
Boot0000* UiApp	FvVol(64074afe-340a-4be6-94ba-91b5b4d0f71e)/FvFile(462caa21-7614-4503-836e-8ab6f4662331)
Boot0002* UEFI VBOX HARDDISK 	PciRoot(0x0)/Pci(0x3,0x0)/SCSI(0,0){auto_created_boot_option}
Boot0003* Ubuntu	HD(1,GPT,1549550d-11b7-41cc-a243-e4ea041f7dd1,0x800,0x165800)/\EFI\ubuntu\shimaa64.efi

```

### Secure Boot and the Shim

On Secure Boot-enabled systems (the default on most Ubuntu installs) the firmware won't execute an arbitrary
EFI binary. It has to be signed by a trusted key. The firmware ships with Microsoft's key in its database, and
Microsoft co-signs a small EFI binary called the **shim**. The actual boot chain becomes:

```
Firmware → shimaa64.efi (signed by Microsoft)
              ↓ verifies against distro key in MOK database
           grubaa64.efi (signed by Canonical)
              ↓ verifies kernel signature
           vmlinuz (signed by Canonical)
```

`mmaa64.efi` (MokManager) is a helper that runs when you need to enroll or manage **Machine Owner Keys (MOK)**,
for example when you install a custom kernel module that needs signing.

### GRUB2

**[GRUB (Grand Unified Bootloader)](https://www.gnu.org/software/grub/manual/grub/grub.html)** is the most
common bootloader on Linux systems. After shim hands off, GRUB:

1. Reads its configuration from `/boot/grub/grub.cfg`
2. Presents a menu of kernel choices (with a timeout)
3. Loads the selected kernel image (`vmlinuz`) and initial RAM disk (`initrd`) into memory
4. Passes a **kernel command line**: a string of parameters like `root=/dev/sda1 ro quiet splash`
5. Transfers control to the kernel entry point

```
shim → grubaa64.efi → reads grub.cfg → loads vmlinuz + initrd → jumps to kernel
```

The kernel image (`vmlinuz`) is a compressed executable. GRUB places it at a specific memory address and calls
the kernel's decompression stub, which unpacks the real kernel and jumps into it.

---

## Stage 4: Kernel Initialization

### Early Boot: Decompression and Setup

The kernel entry point (in `arch/x86/boot/header.S`) runs in a special mode. Its first job is:

1. **Decompress itself**: `vmlinuz` is a `zImage` or `bzImage`, gzip/lz4/zstd-compressed. The decompressor
   unpacks the kernel to a safe memory location.
2. **Switch to 64-bit long mode**: the CPU starts in real or protected mode; the kernel sets up page tables and
   transitions to 64-bit mode.
3. **Establish initial page tables**: a minimal identity mapping to get execution running.

### `start_kernel()`

After decompression and mode switches, execution reaches [`start_kernel()`](https://github.com/torvalds/linux/blob/9207d47f966be9f4d52e7e0119ac2b7a7e366f3e/init/main.c#L1016)
in `init/main.c`, the real starting point of the kernel's C code. This function calls hundreds of initialization
routines in sequence:

```c
asmlinkage __visible void __init __no_sanitize_address start_kernel(void)
{
    ...
    setup_arch(&command_line);   // arch-specific: ACPI, NUMA, memory map
    mm_init();                   // memory management subsystem
    sched_init();                // scheduler
    rcu_init();                  // RCU synchronization
    init_IRQ();                  // interrupt controller
    time_init();                 // timers and clocks
    ...
    rest_init();                 // spawn PID 1
}
```

Key subsystems initialized here:

- **Memory management**: the buddy allocator, slab allocator, vmalloc
- **Scheduler**: CFS (Completely Fair Scheduler) data structures
- **Interrupt subsystem**: IDT (Interrupt Descriptor Table), APIC
- **VFS (Virtual Filesystem Switch)**: the abstraction layer over all filesystems
- **Driver model**: the `kobject`/`sysfs` infrastructure

### Device Detection and Driver Binding

The kernel reads the ACPI tables and walks the PCI bus, building a device tree. For each discovered device,
it matches against registered drivers using the bus's `match()` function. When a match is found, the driver's
`probe()` function runs: it allocates resources, maps registers, and registers the device with higher-level
subsystems (block layer, network stack, etc.).

### Mounting the Root Filesystem

The kernel needs a root filesystem (`/`) to find the rest of the OS. But the real root might live on:
- an encrypted LVM volume
- a software RAID array
- an NVMe device requiring a driver not compiled into the kernel

This chicken-and-egg problem is solved by **initramfs**.

---

## Stage 5: initramfs (Early Userspace)

### What is initramfs?

**[initramfs](https://www.kernel.org/doc/html/latest/admin-guide/initrd.html)** (initial RAM filesystem) is a
compressed `cpio` archive embedded alongside the kernel or passed as a separate file by the bootloader. The
kernel extracts it into a `tmpfs` filesystem in memory and mounts it as the initial `/`.

```
initramfs contains:
  /bin/sh
  /sbin/init  (or systemd)
  /lib/modules/<kver>/kernel/drivers/...  (essential drivers)
  /usr/lib/systemd/system/
  /etc/crypttab
  /etc/mdadm.conf
  ...
```

### The initramfs Job

The init binary inside initramfs (often `systemd` or a script like `busybox init`) performs early setup:

1. **Load kernel modules**: storage drivers (NVMe, AHCI), filesystem drivers (ext4, btrfs), crypto drivers
2. **Assemble storage**: activate RAID arrays (`mdadm`), open LUKS volumes (`cryptsetup`), activate LVM
3. **Find and mount the real root**: using the `root=` kernel parameter
4. **`pivot_root` or `switch_root`**: replace the initramfs `/` with the real root filesystem
5. **Execute the real init**: hand off to `/sbin/init` on the real root

The `switch_root` call is irreversible: the initramfs is freed from memory and the process continues in the real root.

---

## Stage 6: Init System (systemd)

Modern Linux distributions use **systemd** as PID 1, the first real userspace process and parent of all others.

### systemd's Startup Phases

systemd organizes startup into **[targets](https://www.freedesktop.org/software/systemd/man/systemd.special.html)**
(analogous to runlevels in SysV init). The default target for a desktop is `graphical.target`; for a server,
`multi-user.target`. These are dependency graphs of **units**.

```
sysinit.target
    ↓
basic.target
    ↓
multi-user.target ──── sshd.service, NetworkManager.service, ...
    ↓
graphical.target ───── display-manager.service
```

systemd processes units in parallel wherever dependencies allow, making boot dramatically faster than sequential
SysV scripts.

### Key Unit Types

| Unit type | Purpose | Example |
|---|---|---|
| `.service` | A daemon or one-shot process | `sshd.service` |
| `.mount` | A filesystem mount point | `home.mount` |
| `.socket` | Socket-activated service | `systemd-journald.socket` |
| `.target` | Synchronization point / group | `network.target` |
| `.timer` | cron-like scheduled activation | `fstrim.timer` |

### Socket Activation

One of systemd's powerful features is **socket activation**: systemd creates the socket *before* starting the
service, queuing connections. The service starts on first use. This means services can declare dependencies on
each other via sockets without strict ordering. They all start in parallel, and connections simply block until
the service is ready.

### Journal and Logging

systemd replaces syslog with **journald**, a structured binary log. All stdout/stderr of services is captured
automatically. Query with `journalctl`:

```bash
journalctl -b          # logs since last boot
journalctl -u sshd     # logs for a specific service
journalctl --since "10 min ago"
```

---

## Stage 7: Login and Shell

### Getty and Login

For a text console, systemd starts **getty** on each virtual terminal (e.g., `agetty` on `/dev/tty1`). Getty:

1. Opens the TTY device
2. Prints the login prompt
3. Reads the username
4. Calls `/bin/login`, which reads the password and authenticates via
   **[PAM (Pluggable Authentication Modules)](https://man7.org/linux/man-pages/man8/PAM.8.html)**
5. On success, drops privileges to the user's UID/GID and execs the user's shell

### PAM

PAM separates authentication policy from the applications that need it. The `/etc/pam.d/login` configuration
chains modules:

```
auth    required   pam_unix.so      ← check /etc/shadow
auth    optional   pam_google_authenticator.so  ← TOTP 2FA
session required   pam_limits.so    ← apply ulimits from /etc/security/limits.conf
session required   pam_systemd.so   ← register session with logind
```

PAM sits on top of the same kernel-level security primitives (namespaces, capabilities, LSMs) covered in my
recap of the [man7.org Linux Security and Isolation APIs course](https://hrishi.dev/linux/security/kernel/isolation/apis/2025/11/02/linux-security-isolation-apis-course.html),
if you want to go a layer deeper than PAM itself.

### Shell Startup

Once login succeeds, the shell (e.g., `bash`) is execed. Bash reads startup files in order:

```
/etc/profile          ← system-wide environment
~/.bash_profile       ← user login setup (sources ~/.bashrc)
~/.bashrc             ← interactive shell config (aliases, prompt, PATH)
```

At this point, a shell prompt appears and the OS is fully booted.

---

## Putting It All Together

The timeline below is an **illustrative example**, not a benchmark from a specific machine, since actual timings
vary significantly with storage type (NVMe vs. spinning disk), firmware implementation, and how many services a
distribution starts by default. Use `systemd-analyze` and `systemd-analyze blame` on your own machine to see
real, measured numbers for this breakdown.

```
[0 ms]      CPU reset → firmware ROM at 0xFFFFFFF0
[10 ms]     POST: memory training, device enumeration, ACPI table construction
[500 ms]    UEFI boot manager loads grubx64.efi from ESP
[600 ms]    GRUB displays menu, loads vmlinuz + initrd into RAM
[700 ms]    Kernel decompresses, switches to 64-bit mode
[800 ms]    start_kernel(): mm_init, sched_init, IRQ init, driver probing
[900 ms]    initramfs: load storage drivers, assemble volumes, mount real root
[1.0 s]     switch_root → PID 1 = systemd on the real root
[1.5 s]     systemd activates sysinit.target → basic.target
[2.0 s]     Network, storage, logging services start in parallel
[3.0 s]     multi-user.target reached: system is operational
[3.5 s]     graphical.target: display manager starts
[4.0 s]     Login prompt appears
```

The general pattern holds even if your own numbers differ: NVMe storage and UEFI cut boot time dramatically
compared to the old BIOS-plus-spinning-disk path, where POST alone could take several seconds and a full boot
commonly ran 30-60 seconds.

---

## Part 2: Booting Without a Disk

Everything above assumes a disk with an operating system on it. A server fresh from the rack has none. The
firmware still starts at the reset vector and still runs POST, but Stage 3 has nothing to read. Instead, the boot
NIC asks the network for a bootloader, the bootloader fetches a kernel and initramfs over HTTP, and the first Linux
to run is a RAM-only agent whose only job is to write an OS image to disk and reboot. Then Stages 1-7 run again,
this time from disk.

The worked example is **[Metal³](https://metal3.io/)**, a Kubernetes-native bare-metal provisioner whose Bare
Metal Operator drives **[OpenStack Ironic](https://docs.openstack.org/ironic/latest/)** to boot, image, and reboot
physical hosts. The example builds a Kubernetes cluster because that is what the Cluster API layer on top is for,
but the Kubernetes part is optional: a `BareMetalHost` with `spec.image` and `spec.userData` set by hand gets the
same network boot, image write, and cloud-init run, so the same machinery provisions a database host, a GPU box, or
a plain fleet of Linux servers. The PXE and DHCP mechanics are identical under Foreman, MAAS, or Tinkerbell.

| Component | Role |
|---|---|
| **Cluster API (CAPI)** | `Cluster`, `Machine`, `KubeadmControlPlane`. Renders the cloud-init that runs `kubeadm init` or `join`. |
| **CAPM3** | Metal³'s CAPI provider. Maps a `Machine` to a `BareMetalHost` and renders its metadata and network data. |
| **BMO** | Bare Metal Operator. Owns the `BareMetalHost` state machine; the only client of Ironic. |
| **Ironic pod** | `ironic` (API, port 6385), `dnsmasq` (DHCP, TFTP), `httpd` (iPXE scripts and images, port 6180). |
| **DHCP relay** | The router interface on each host VLAN with `ip helper-address <Ironic IP>`. |
| **BMC** | IPMI or Redfish on the host. Power and boot-device control only. |
| **IPA** | Ironic Python Agent: a RAM-only Linux that writes the image and the config drive. |

---

## Stage 3, Over the Network: PXE, DHCP, TFTP, and iPXE

With no bootable disk, the firmware falls through to network boot and the NIC's **PXE (Preboot eXecution
Environment)** ROM takes over. PXE only speaks DHCP and TFTP, so its whole job is to fetch a smarter bootloader,
**[iPXE](https://ipxe.org/)**, which adds HTTP and a second DHCP round of its own.

```
 host firmware / iPXE              DHCP relay (router)         Ironic: dnsmasq + httpd
 ─────────────────────             ───────────────────         ───────────────────────
 PXE ROM
   │ DHCPDISCOVER (broadcast)
   │ opt 60 "PXEClient", opt 93 arch ──► giaddr = 10.20.0.1 ──►
   ◄── OFFER: ip, router, next-server, filename = snponly.efi ◄──
   │ TFTP GET snponly.efi ─────── routed unicast, no relay ───►
   ◄── iPXE binary
   ▼
 iPXE
   │ DHCPDISCOVER again, opt 175 marks it as iPXE ──► relay ──►
   ◄── filename = http://172.22.0.2:6180/boot.ipxe
   │ HTTP GET boot.ipxe, then pxelinux.cfg/<mac> ────────────►   per-host script Ironic wrote
   │ HTTP GET IPA kernel + initramfs ─────────────────────────►
   ▼
 kernel + initramfs (IPA) boot: Stages 4-6 run entirely out of RAM
```

- **Round one.** The ROM's `DHCPDISCOVER` carries option 60 (`PXEClient`) and option 93, the client architecture
  from [RFC 4578](https://www.rfc-editor.org/rfc/rfc4578). dnsmasq matches the architecture and answers with an
  address, a router, and a `filename`: `snponly.efi` for UEFI, `undionly.kpxe` for BIOS. The ROM fetches it over
  TFTP. That file is iPXE, and it is the last thing TFTP is used for.
- **Round two.** iPXE sends its own `DHCPDISCOVER`, tagged with option 175. dnsmasq recognizes the tag and hands
  back a different `filename`: an HTTP URL for `boot.ipxe`. Without the tag the ROM would loop, re-downloading iPXE
  forever.
- **HTTP.** `boot.ipxe` chains to `pxelinux.cfg/<mac>`, the per-host script Ironic wrote when it started the
  deploy. It names the IPA kernel, initramfs, and command line (`ipa-api-url=http://172.22.0.2:6385`, an agent
  token, `BOOTIF=<mac>`). iPXE loads both files and jumps to the kernel exactly as GRUB does.
- **Stages 4-6, in RAM.** Decompression, `start_kernel()`, and the initramfs proceed unchanged, with one
  difference: IPA's initramfs never does `switch_root`. The ramdisk *is* the root filesystem, and systemd inside it
  starts `ironic-python-agent.service` as the workload.

One caveat: Ironic's iPXE binaries are not signed by Microsoft, so a host with Secure Boot enforced refuses
`snponly.efi`. That is the main reason fleets move to **Redfish virtual media**, where Ironic asks the BMC to mount
a signed IPA ISO as a virtual CD-ROM and the whole chain above collapses into one BMC call.

---

## Crossing VLANs: The DHCP Relay

`DHCPDISCOVER` is a broadcast from a client with no address yet. Broadcasts do not cross routers, so on a fabric
where each rack is its own VLAN, a host on VLAN 20 cannot reach dnsmasq on VLAN 10 without help. The help is the
**DHCP relay** from [RFC 2131](https://www.rfc-editor.org/rfc/rfc2131#section-4.3.1), configured on Cisco-style
gear as `ip helper-address` on the host VLAN's routed interface.

```
   VLAN 20 · 10.20.0.0/24              router / L3 switch               VLAN 10 · 172.22.0.0/24
   ┌────────────────┐          ┌──────────────────────────┐          ┌──────────────────────┐
   │ host NIC       │ DISCOVER │ SVI20  10.20.0.1         │ unicast  │ Ironic pod           │
   │ 0.0.0.0 → bcast│ ───────► │ helper → 172.22.0.2      │ ───────► │ VIP 172.22.0.2       │
   │                │          │ SVI10  172.22.0.1        │ giaddr=  │ dnsmasq picks the    │
   │                │ ◄─────── │                          │ ◄─────── │ range from giaddr    │
   └────────────────┘  OFFER   └──────────────────────────┘  OFFER   └──────────────────────┘
           │  delivered on VLAN 20                          to giaddr           ▲
           │                                                                    │
           └──── TFTP 69 · HTTP 6180 · API 6385: routed unicast (needs option 3) ┘
                 dnsmasq hands out option 3 = 10.20.0.1 so the host can leave its subnet
```

The relay turns the broadcast into a unicast packet to the helper target and sets `giaddr` to its own address on
the host VLAN. dnsmasq picks the address range from `giaddr`, never from the interface the packet came in on, and
sends the `OFFER` back to the relay, which delivers it on the host VLAN.

Only the four DHCP messages pass through the relay, and they do so **three times per provisioning**: for the PXE
ROM, for iPXE, and once more when the Linux kernel inside IPA brings up its NIC. That third round is the one people
forget. If it returns no router option, IPA has an address but cannot reach Ironic, and the host sits in
`provisioning` until the deploy callback times out. Everything after DHCP (TFTP, HTTP, the agent API) is ordinary
routed unicast.

```
! switch side, one block per host VLAN
interface Vlan20
  ip address 10.20.0.1 255.255.255.0
  ip helper-address 172.22.0.2     ! Ironic VIP, not a node IP

# dnsmasq side, one block per relayed subnet
dhcp-range=set:vlan20,10.20.0.50,10.20.0.200,255.255.255.0,12h
dhcp-option=tag:vlan20,option:router,10.20.0.1
# the stock ironic-image config only knows one DHCP_RANGE; extra ranges need a custom config
```

Skip all of this on a single flat VLAN, and skip it entirely with Redfish virtual media plus DHCP-less IPA, where a
pre-provisioning network-data Secret gives the agent a static address.

---

## The Metal3 Provisioning Sequence

Here is every call from `kubectl apply` of a `Cluster` to a physical server showing up as a Ready `Node`. The
host sits on VLAN 20 and Ironic on VLAN 10, so the DHCP hops cross the relay from the previous section.

<div style="overflow-x:auto;border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:8px;margin:1em 0;">
<svg id="metal3-seq" viewBox="0 0 1278 1764" role="img" style="display:block;min-width:1100px;width:100%;height:auto;" aria-label="Sequence of calls from applying a Cluster manifest to a bare-metal node joining the cluster, across kubectl, Cluster API, CAPM3, BMO, Ironic, dnsmasq and httpd, the DHCP relay, the BMC, the host firmware, and the Ironic Python Agent" xmlns="http://www.w3.org/2000/svg">
<style>
#metal3-seq{--bg:#fdfdfd;--lane:#ffffff;--band:#e4eaf4;--note:#fff1cc;--net:#c8420a;--mono:ui-monospace,SFMono-Regular,Menlo,monospace;color:#151414;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme: dark){#metal3-seq{--bg:#181a1b;--lane:#24282a;--band:#2b3441;--note:#3b2f12;--net:#ff8e55;color:#e8e6e3}}
</style>
<defs><marker id="m3s-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker><marker id="m3s-ahn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--net)"/></marker></defs>
<line x1="72" y1="78" x2="72" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="13" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="72" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">kubectl</text>
<text x="72" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">operator / GitOps</text>
<line x1="198" y1="78" x2="198" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="139" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="198" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">CAPI</text>
<text x="198" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">core + kubeadm bootstrap</text>
<line x1="324" y1="78" x2="324" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="265" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="324" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">CAPM3</text>
<text x="324" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">Metal3Machine ctrl</text>
<line x1="450" y1="78" x2="450" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="391" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="450" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">BMO</text>
<text x="450" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">BareMetalHost ctrl</text>
<line x1="576" y1="78" x2="576" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="517" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="576" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">Ironic</text>
<text x="576" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">API + conductor :6385</text>
<line x1="702" y1="78" x2="702" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="643" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="702" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">BMC</text>
<text x="702" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">IPMI / Redfish</text>
<line x1="828" y1="78" x2="828" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="769" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="828" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">dnsmasq / httpd</text>
<text x="828" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">DHCP · TFTP · HTTP 6180</text>
<line x1="954" y1="78" x2="954" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="895" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="954" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">DHCP relay</text>
<text x="954" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">router SVI, helper-address</text>
<line x1="1080" y1="78" x2="1080" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="1021" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="1080" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">Host firmware</text>
<text x="1080" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">PXE ROM → iPXE</text>
<line x1="1206" y1="78" x2="1206" y2="1744" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="1147" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="1206" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">IPA → OS</text>
<text x="1206" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">agent in RAM, then disk</text>
<rect x="8" y="84" width="1262" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="102" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">0 · Inventory: happens once per host, before any cluster exists</text>
<line x1="72" y1="134" x2="450" y2="134" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="261.0" y="128" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">apply BareMetalHost + BMC Secret</text>
<line x1="450" y1="170" x2="576" y2="170" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="513.0" y="164" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">create node: driver, BMC addr, bootMAC</text>
<line x1="576" y1="206" x2="702" y2="206" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="639.0" y="200" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">boot=pxe (one-time), power on → inspection</text>
<line x1="1080" y1="242" x2="1206" y2="242" stroke="currentColor" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#m3s-ah)"/>
<text x="1134.9" y="236" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">same relayed PXE → iPXE → IPA chain as phase 3</text>
<line x1="1206" y1="278" x2="576" y2="278" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="891.0" y="272" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST /v1/continue_inspection (inventory)</text>
<line x1="576" y1="314" x2="450" y2="314" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="513.0" y="308" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">inspected → BMO fills status.hardware</text>
<rect x="442" y="340" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="464" y="354" font-size="11" font-style="italic" text-anchor="start" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">BMH: Available · powered off</text>
<rect x="8" y="372" width="1262" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="390" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">1 · Declare the cluster</text>
<line x1="72" y1="422" x2="198" y2="422" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="297.0" y="416" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">apply Cluster, Metal3Cluster, KubeadmControlPlane, Metal3MachineTemplate, Metal3DataTemplate, IPPool</text>
<rect x="190" y="448" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="212" y="462" font-size="11" font-style="italic" text-anchor="start" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">Machine + KubeadmConfig → user-data Secret (cloud-init: kubeadm init)</text>
<line x1="198" y1="494" x2="324" y2="494" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="261.0" y="488" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">Metal3Machine created (infrastructureRef)</text>
<rect x="8" y="516" width="1262" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="534" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">2 · Claim a host</text>
<line x1="324" y1="566" x2="450" y2="566" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="387.0" y="560" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">pick BMH matching hostSelector, set consumerRef</text>
<rect x="316" y="592" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="338" y="606" font-size="11" font-style="italic" text-anchor="start" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">Metal3Data: metaData + networkData Secrets (IPClaim → IPPool)</text>
<line x1="324" y1="638" x2="450" y2="638" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="387.0" y="632" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">patch BMH.spec: image, userData, networkData, metaData, online=true</text>
<rect x="8" y="660" width="1262" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="678" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">3 · Ironic boots the agent: host on VLAN 20, Ironic on VLAN 10, DHCP crosses the router via the helper</text>
<line x1="450" y1="710" x2="576" y2="710" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="513.0" y="704" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">instance_info = image/checksum/configdrive; provision → active</text>
<line x1="576" y1="746" x2="828" y2="746" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="702.0" y="740" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">write pxelinux.cfg/&lt;mac&gt; (IPA boot script)</text>
<line x1="576" y1="782" x2="702" y2="782" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="639.0" y="776" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">set boot device PXE (one-time), power on</text>
<line x1="702" y1="818" x2="1080" y2="818" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="891.0" y="812" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">power on, POST (5–10 min on big boxes)</text>
<line x1="1080" y1="854" x2="954" y2="854" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="1017.0" y="848" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">DHCPDISCOVER broadcast  opt60 PXEClient, opt93 arch</text>
<line x1="954" y1="890" x2="828" y2="890" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="891.0" y="884" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">unicast to helper target: giaddr=10.20.0.1, hops=1</text>
<line x1="828" y1="926" x2="954" y2="926" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="891.0" y="920" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">OFFER: range picked by giaddr, ip, router, next-server, snponly.efi</text>
<line x1="954" y1="962" x2="1080" y2="962" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="1017.0" y="956" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">OFFER re-broadcast on VLAN 20 (REQUEST/ACK repeat this path)</text>
<line x1="1080" y1="998" x2="828" y2="998" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="954.0" y="992" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">TFTP GET iPXE binary: routed unicast, no relay involved</text>
<line x1="1080" y1="1034" x2="954" y2="1034" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="1017.0" y="1028" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">iPXE DHCPs again (opt175) → relayed the same way</text>
<line x1="828" y1="1070" x2="1080" y2="1070" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="954.0" y="1064" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">filename = http://ironic:6180/boot.ipxe  (via relay)</text>
<line x1="1080" y1="1106" x2="828" y2="1106" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="954.0" y="1100" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">HTTP GET boot.ipxe → pxelinux.cfg/&lt;mac&gt; → IPA kernel + initramfs</text>
<line x1="1080" y1="1142" x2="1206" y2="1142" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="1117.8" y="1136" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">kernel boots IPA  (ipa-api-url, agent token, BOOTIF)</text>
<rect x="8" y="1164" width="1262" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="1182" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">4 · The agent writes the image</text>
<line x1="1206" y1="1214" x2="954" y2="1214" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3s-ahn)"/>
<text x="1046.55" y="1208" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">Linux DHCP (relayed again): needs option 3 router to reach Ironic off-subnet</text>
<line x1="1206" y1="1250" x2="576" y2="1250" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="891.0" y="1244" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST /v1/lookup, then heartbeat every ~10 s (callback_url :9999)</text>
<line x1="576" y1="1286" x2="1206" y2="1286" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="891.0" y="1280" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST :9999/v1/commands  prepare_image {url, checksum, configdrive}</text>
<line x1="1206" y1="1322" x2="828" y2="1322" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="1017.0" y="1316" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">GET image.raw  (httpd cache or your image server)</text>
<rect x="1198" y="1348" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="1192" y="1362" font-size="11" font-style="italic" text-anchor="end" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">dd to rootDeviceHints device · write config-2 partition (user_data, meta_data, network_data)</text>
<line x1="1206" y1="1394" x2="576" y2="1394" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="891.0" y="1388" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">heartbeat: deploy steps done</text>
<line x1="576" y1="1430" x2="702" y2="1430" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="639.0" y="1424" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">boot device = disk (persistent), reboot</text>
<line x1="576" y1="1466" x2="450" y2="1466" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="513.0" y="1460" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">node active → BMH Provisioned</text>
<rect x="8" y="1488" width="1262" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="1506" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">5 · The OS boots from disk and joins</text>
<line x1="702" y1="1538" x2="1080" y2="1538" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="891.0" y="1532" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST, boot from disk: IPA is gone</text>
<rect x="1198" y="1564" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="1192" y="1578" font-size="11" font-style="italic" text-anchor="end" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">cloud-init ConfigDrive: network_data → static IP on the baremetal net · user_data → kubeadm init / join</text>
<line x1="1206" y1="1610" x2="198" y2="1610" stroke="currentColor" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#m3s-ah)"/>
<text x="702.0" y="1604" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">kubelet registers Node (label metal3.io/uuid=&lt;bmh uid&gt;) with the workload API</text>
<line x1="324" y1="1646" x2="1206" y2="1646" stroke="currentColor" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#m3s-ah)"/>
<text x="765.0" y="1640" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">find Node by label, set Node.spec.providerID = metal3://ns/bmh/m3m</text>
<line x1="324" y1="1682" x2="198" y2="1682" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3s-ah)"/>
<text x="261.0" y="1676" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">Metal3Machine ready=true → Machine Running</text>
<rect x="190" y="1708" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="212" y="1722" font-size="11" font-style="italic" text-anchor="start" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">KCP initialized → MachineDeployment workers repeat phases 2–5, one host each</text>
</svg>
</div>

<p style="font-size:0.9em;opacity:.85;margin-top:-0.4em;">Orange arrows are network boot (DHCP, TFTP, HTTP); black arrows are Kubernetes API, Ironic API, and BMC calls; dashed arrows go via the workload cluster API; yellow boxes are local state changes. Phase 0 runs once per host; phases 1-5 run for every Machine. Scroll sideways on narrow screens.</p>

| Phase | What happens | Takes |
|---|---|---|
| **0 · Inventory** | You apply a `BareMetalHost` and BMC Secret. BMO registers it in Ironic, which PXE-boots IPA once to inspect the hardware. Host ends `available`, powered off. | 10-20 min, once per host |
| **1 · Declare** | You apply `Cluster`, `KubeadmControlPlane`, and the Metal³ templates. CAPI creates a `Machine` and a user-data Secret (cloud-init with `kubeadm init`). | seconds |
| **2 · Claim** | CAPM3 picks an `available` host matching `hostSelector`, renders its metadata and `network_data.json` (static IPs from an `IPPool`, NIC mapping from inspected MACs), and patches the BMH: `image`, `userData`, `networkData`, `online: true`. | seconds |
| **3 · Network boot** | BMO tells Ironic to deploy. Ironic writes `pxelinux.cfg/<mac>`, asks the BMC for a one-time PXE boot, and powers the host on. POST, then the PXE → iPXE → IPA chain. | 3-10 min, mostly POST |
| **4 · Deploy** | IPA DHCPs (third relayed round), looks itself up, and heartbeats to Ironic. Ironic calls back on port 9999 with `prepare_image`; IPA streams the image to the `rootDeviceHints` device and writes the `config-2` partition. Ironic sets boot=disk on the BMC and reboots. BMH turns `provisioned`. | 2-6 min, image-size bound |
| **5 · Boot and join** | Stages 1-7 from disk. cloud-init's ConfigDrive datasource applies the network data and runs `kubeadm`. kubelet registers with label `metal3.io/uuid`; CAPM3 finds the Node, sets `providerID`, and the `Machine` goes `Running`. | 3-10 min |

For a non-Kubernetes host, phases 1 and 2 collapse into you writing `spec.image` and `spec.userData` on the
`BareMetalHost` yourself, and phase 5 is just cloud-init running whatever that user-data says.

### Mapping the Phases Back to the Seven Stages

| Phase | In Part 1 terms |
|---|---|
| 3 | Stages 1-2 as before. Stage 3 is PXE ROM → iPXE instead of shim → GRUB. Stages 4-6 boot IPA's kernel, initramfs, and systemd. No Stage 7: nobody logs in. |
| 4 | Still inside the ramdisk, on a `tmpfs` root. The agent streams an image to a block device it found during Stage 4 driver probing. |
| 5 | The full Stage 1-7 sequence from the freshly written disk. cloud-init runs as ordinary systemd units in Stage 6 and reads the `config-2` partition IPA wrote. |

### Two State Machines Side by Side

The Cluster API `Machine` sees three transitions. All the hardware work happens inside the `BareMetalHost` row.

```
BareMetalHost.status.provisioning.state  (BMO)

 registering → inspecting* → preparing* → available → provisioning* → provisioned
                                             ▲                             │
                                             └──── deprovisioning* ◄───────┘   Machine deleted
 * = IPA is booted over the network in this state

Machine.status.phase  (Cluster API)

 Pending ─────────────► Provisioning ────────────────────────────► Provisioned → Running
 bootstrap Secret       waiting for the BMH to reach provisioned     Node.spec.providerID
 ready, no host yet     and for the Node to appear                   matches
```

A host stuck in `provisioning` past Ironic's 30-minute deploy-callback timeout means the BMC powered the box on but
IPA never called home: one of the orange arrows is broken.

### What Crosses the Wire

| Direction | Port | Carries | If blocked |
|---|---|---|---|
| Ironic → BMC | 623/udp IPMI, 443 Redfish | power, boot device, virtual media | BMH stuck `registering` |
| host → relay → dnsmasq | 67/68 udp | DHCP with `giaddr` set, three rounds | `PXE-E51 No DHCP offers` |
| host → dnsmasq | 69/udp | iPXE binary via TFTP | `PXE-E32 TFTP open timeout` |
| host → httpd | 6180/tcp | `boot.ipxe`, per-MAC script, IPA kernel and initramfs | iPXE `Could not chain` |
| IPA → Ironic | 6385/tcp | lookup, heartbeat, inspection callback | BMH sits in `provisioning` until timeout |
| Ironic → IPA | 9999/tcp | agent commands (`prepare_image`) | heartbeats fine, nothing happens |
| OS → control plane | 6443/tcp | `kubeadm join` | BMH `provisioned`, Machine stuck `Provisioning` |

The arrow most firewall policies get wrong is Ironic → IPA on 9999: Ironic has to open a connection *to* the host,
across the router. A policy that only allows host → Ironic passes every check up to the heartbeat and then stalls.

---

## The Metal3 Object Model

The same objects, viewed as a composition instead of in time order. One provisioned host is a spine of six
objects, each owning or referencing a few helpers on its row. Read it top-down: that is the order they come into
existence. Everything from the `BareMetalHost` row down works without the rows above it.

```
Cluster ─────────────────── infrastructureRef ──► Metal3Cluster (controlPlaneEndpoint = your VIP)
  │ controlPlaneRef
  ▼
KubeadmControlPlane / MachineDeployment ── infrastructureRef ──► Metal3MachineTemplate (image, hostSelector, dataTemplate)
  │ one Machine per replica
  ▼
Machine ─────────────────── bootstrap.configRef ──► KubeadmConfig ──► Secret <machine> (cloud-init: kubeadm init|join)
  │ infrastructureRef
  ▼
Metal3Machine ───────────── spec.dataTemplate ──► Metal3DataTemplate (metaData, networkData templates)
  │                         owns Metal3Data ──► Secret <m3m>-metadata, Secret <m3m>-networkdata
  │                                └── owns IPClaim ──► IPAddress ◄── from IPPool
  │ claims a host (consumerRef)
  ▼
BareMetalHost ───────────── bmc.credentialsName ──► Secret (BMC username/password)
  │                         owns HardwareData (inspection), HostFirmwareSettings, HostFirmwareComponents
  │                         spec.image / userData / metaData / networkData  ◄── patched in by CAPM3 (or by you)
  │ BMO mirrors it into
  ▼
Ironic node (not a Kubernetes object) ── BMC + PXE / virtual media ──► physical host ──► Node in the workload cluster
```

The two objects that carry the Metal³-specific decisions are the `Metal3MachineTemplate` (which image, which
hosts) and the `Metal3DataTemplate`, which turns inspected hardware into per-host cloud-init data:

```yaml
apiVersion: infrastructure.cluster.x-k8s.io/v1beta1
kind: Metal3DataTemplate
metadata: { name: test1-controlplane-template, namespace: metal3 }
spec:
  clusterName: test1
  metaData:
    objectNames: [{ key: name, object: machine }]         # → ds.meta_data.name
    fromHostInterfaces: [{ key: provisioningMAC, interface: eth0 }]
  networkData:
    links:
      ethernets:
        - { type: phy, id: enp1s0, macAddress: { fromHostInterface: eth1 } }   # from status.hardware.nics
    networks:
      ipv4:
        - id: baremetal
          link: enp1s0
          ipAddressFromIPPool: baremetal-r01
          routes: [{ network: 0.0.0.0, prefix: 0, gateway: { fromIPPool: baremetal-r01 } }]
```

Rules that follow from the composition:

- **Inventory survives.** Deleting a `Machine` releases the host: BMO runs a cleaning boot and the BMH returns to
  `available`. `BareMetalHost`, its BMC Secret, `IPPool`, and the templates are never deleted by Cluster API.
- **Templates are immutable.** Changing an image means a new `Metal3MachineTemplate` name and a rolling
  replacement from the pool.
- **The label matters.** The kubelet must register with `node-labels: "metal3.io/uuid={{ ds.meta_data.uuid }}"`;
  that is how CAPM3 finds the `Node` in phase 5. Without it the OS boots and joins, and the `Machine` never leaves
  `Provisioning`.
- **Names you can rely on.** `HardwareData` and the firmware objects share the BMH's name; the providerID is
  `metal3://<ns>/<bmh>/<metal3machine>` on both the `Metal3Machine` and the workload `Node`.

---

## Troubleshooting a Network Boot

Work top to bottom: each row assumes the ones above it are healthy. The `PXE-Exx` code on the console tells you
the exact hop.

| Symptom | Where to look | Usual cause |
|---|---|---|
| `PXE-E51 No DHCP or proxyDHCP offers` | `tcpdump -ni <prov-if> udp port 67` on the Ironic host. Relayed packets show `Gateway-IP` | No helper on the SVI, helper pointing at a node IP instead of the VIP, DHCP snooping, PXE on the wrong NIC |
| Packets arrive, no OFFER: dnsmasq logs `no address range available for DHCP request via 10.20.0.1` | dnsmasq container log | No `dhcp-range` for the relay's subnet |
| `PXE-E32 TFTP open timeout` or `PXE-E53 No boot filename received` | `tcpdump udp port 69`; dnsmasq log for "sent snponly.efi" | ACL blocks 69/udp; UEFI vs BIOS mismatch with `bootMode` |
| iPXE: `Could not chain http://172.22.0.2:6180/boot.ipxe` | `curl` that URL from the host VLAN; httpd log | No router option, 6180 blocked, or iPXE's second DHCP answered by another server |
| iPXE loads `inspector.ipxe` or 404s on `pxelinux.cfg/<mac>` | `ls /shared/html/pxelinux.cfg/` in the ironic container | `bootMACAddress` on the BMH is not the NIC that PXE-boots |
| IPA is up on the console, but after 30 min `timeout reached while waiting for callback` | ironic log for `lookup` / `heartbeat`; `journalctl -u ironic-python-agent` on the console | Third DHCP round returned no router, or 6385 blocked |
| Heartbeats logged; conductor says `Failed to connect to the agent ... :9999` | `curl http://<host-ip>:9999/v1/status` from the Ironic host | Firewall only allows host → Ironic |
| Deploy succeeds, host reboots into IPA again | BMC boot settings; Ironic log for "set boot device to disk" | BMC ignored the persistent boot-device change; wrong `rootDeviceHints` |
| BMH `provisioned`, OS up, Machine stuck `Provisioning` | `cloud-init status --long`; is `/dev/disk/by-label/config-2` there? Node labels | Config drive not read, wrong link/MAC in `networkData`, or the `metal3.io/uuid` label missing |

Five-minute triage, in order:

```bash
# 1. Where is it stuck?
kubectl get bmh -A -o custom-columns=NAME:.metadata.name,STATE:.status.provisioning.state,ERR:.status.errorMessage
# 2. Is DHCP arriving? Relayed packets carry Gateway-IP.
tcpdump -ni <prov-if> -v 'udp port 67' | grep -E 'Gateway-IP|Client-Ethernet|Server-ID'
# 3. What did dnsmasq and httpd do with it?
kubectl -n baremetal-operator-system logs deploy/ironic -c dnsmasq | tail -50
kubectl -n baremetal-operator-system logs deploy/ironic -c ironic-httpd | grep -E 'boot.ipxe|pxelinux.cfg'
# 4. Is the agent talking, and can Ironic talk back?
kubectl -n baremetal-operator-system logs deploy/ironic -c ironic | grep -E 'heartbeat|lookup|Failed to connect'
curl -s http://<host-prov-ip>:9999/v1/status
# 5. Console (BMC KVM / SOL): the PXE-Exx code names the hop
```

---

## Closing Thoughts

Understanding this stack, from the reset vector through PAM, is invaluable when debugging boot failures,
hardening systems, building embedded Linux images, or simply satisfying the curiosity of knowing what your
machine is actually doing before that shell prompt appears. It pays off twice on bare metal: the network-boot
path in Part 2 is the same seven stages with Stage 3 swapped out, run once for the agent and once for the real
OS, and every `PXE-Exx` code or stuck `provisioning` state maps back to one specific hop in that sequence.

> **Key Takeaways**
> - The CPU always starts at a hardwired **reset vector**, `0xFFFFFFF0` on x86-64, mapped to firmware ROM rather than RAM.
> - **UEFI replaced BIOS** with 64-bit execution, a FAT32 EFI System Partition, Secure Boot, and a full driver model.
> - The bootloader's job is narrow: find the kernel, load it, pass parameters, and jump. GRUB2 does this via `shim → grubaa64.efi → vmlinuz`.
> - **initramfs** solves the chicken-and-egg problem of needing drivers to mount the filesystem that contains the drivers.
> - **systemd** parallelizes init by activating units based on a dependency graph rather than a static script order.
> - **PAM** decouples authentication policy from the applications that need it, enabling pluggable 2FA, LDAP, and biometrics.
> - On a diskless host, Stage 3 becomes **PXE ROM → DHCP → TFTP → iPXE → DHCP again → HTTP → kernel + initramfs**, and the first Linux to boot is a RAM-only agent that installs the real one.
> - Only the four DHCP messages need a **relay** to cross VLANs, three rounds per provisioning; everything after is routed unicast that needs a router option to work.
> - In Metal³, the `BareMetalHost` state machine does all the hardware work and needs no Kubernetes on the host it provisions; a BMH stuck in `provisioning` past the 30-minute callback timeout means one network-boot hop is broken, and the console's `PXE-Exx` code names it.
