---
layout: post
title:  "From Power Button to Shell Prompt, Part 2: PXE Network Boot with Metal³ Bare Metal Operator and Ironic"
date:   2026-09-18 06:00:00 +0000
categories: [Linux, OS, Systems]
description: "How a diskless server boots Linux over the network: PXE, DHCP, TFTP, and iPXE, the DHCP relay that carries it across VLANs, and how Metal3's Bare Metal Operator and Ironic use that chain to boot the Ironic Python Agent, write an OS image, and reboot from disk. Ends with a hop-by-hop troubleshooting table for PXE-Exx errors and stuck BareMetalHosts."
image: /assets/os-boot-process-stages.png
---

*Part 2 of a 3-part series on the Linux boot process. [Part 1](https://hrishi.dev/linux/os/systems/2026/05/05/os-boot-process-1.html) covered the seven stages of booting from a
disk. This part takes those stages onto a blank server that has to fetch its bootloader, kernel, and operating
system over the network.*

---

## Introduction

**[Part 1](https://hrishi.dev/linux/os/systems/2026/05/05/os-boot-process-1.html)** walked through a Linux boot from a disk: firmware, POST, bootloader, kernel init, initramfs,
systemd, and login. A server fresh from the rack has no disk to boot from. The firmware still starts at the reset
vector and still runs POST, but Stage 3 has nothing to read. Instead, the boot NIC asks the network for a
bootloader, the bootloader fetches a kernel and initramfs over HTTP, and the first Linux to run is a RAM-only agent
whose only job is to write an OS image to disk and reboot. Then Stages 1-7 run again, this time from disk.

The worked example is **[Metal³](https://metal3.io/)**, a Kubernetes-native bare-metal provisioner whose Bare
Metal Operator (BMO) drives **[OpenStack Ironic](https://docs.openstack.org/ironic/latest/)** to boot, image, and
reboot physical hosts. This part stays at the level of one host: a `BareMetalHost` with `spec.image` and
`spec.userData` set by hand gets a network boot, an image write, and a cloud-init run, which is all it takes to
provision a database host, a GPU box, or a plain fleet of Linux servers. **[Part 3](https://hrishi.dev/linux/os/systems/2026/09/19/os-boot-process-3.html)** puts Cluster API on top
of the same machinery to build Kubernetes clusters. The PXE and DHCP mechanics are identical under Foreman, MAAS,
or Tinkerbell.

| Component | Role |
|---|---|
| **BMO** | Bare Metal Operator. Owns the `BareMetalHost` state machine; the only client of Ironic. |
| **Ironic pod** | `ironic` (API, port 6385), `dnsmasq` (DHCP, TFTP), `httpd` (iPXE scripts and images, port 6180). |
| **DHCP relay** | The router interface on each host VLAN with `ip helper-address <Ironic IP>`. |
| **BMC** | IPMI or Redfish on the host. Power and boot-device control only. |
| **IPA** | Ironic Python Agent: a RAM-only Linux that writes the image and the config drive. |

## Table of Contents

1. [Stage 3, Over the Network: PXE, DHCP, TFTP, and iPXE](#stage-3-over-the-network-pxe-dhcp-tftp-and-ipxe)
2. [Crossing VLANs: The DHCP Relay](#crossing-vlans-the-dhcp-relay)
3. [From BareMetalHost to Provisioned Host](#from-baremetalhost-to-provisioned-host)
4. [Mapping the Phases Back to the Seven Stages](#mapping-the-phases-back-to-the-seven-stages)
5. [The BareMetalHost State Machine](#the-baremetalhost-state-machine)
6. [What Crosses the Wire](#what-crosses-the-wire)
7. [Troubleshooting a Network Boot](#troubleshooting-a-network-boot)

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

## From BareMetalHost to Provisioned Host

Here is every call BMO and Ironic make with that boot chain, from `kubectl apply` of a `BareMetalHost` to a
physical server running the OS you asked for. Nothing in this section involves Cluster API: you write the object
by hand, and BMO does the rest. The host sits on VLAN 20 and Ironic on VLAN 10, so the DHCP hops cross the relay
from the previous section.

```yaml
apiVersion: metal3.io/v1alpha1
kind: BareMetalHost
metadata: { name: r01-s07, namespace: metal3 }
spec:
  online: true
  bootMACAddress: 52:54:00:9a:1b:2c          # the NIC that PXE-boots, on VLAN 20
  bootMode: UEFI
  bmc:
    address: redfish://10.0.0.7/redfish/v1/Systems/1   # or ipmi://10.0.0.7
    credentialsName: r01-s07-bmc               # Secret with username / password
  rootDeviceHints: { deviceName: /dev/nvme0n1 }
  image:
    url: http://images.example.internal/ubuntu-24.04.raw
    checksum: http://images.example.internal/ubuntu-24.04.raw.sha256
    checksumType: sha256
    format: raw
  userData: { name: r01-s07-user-data, namespace: metal3 }        # cloud-init
  networkData: { name: r01-s07-network-data, namespace: metal3 }  # optional static addressing
```

<div style="overflow-x:auto;border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:8px;margin:1em 0;">
<svg id="metal3-bmh-seq" viewBox="0 0 1026 1486" role="img" style="display:block;min-width:900px;width:100%;height:auto;" aria-label="Sequence of calls from applying a BareMetalHost by hand to the host booting its freshly written OS from disk, across kubectl, BMO, Ironic, the BMC, dnsmasq and httpd, the DHCP relay, the host firmware, and the Ironic Python Agent" xmlns="http://www.w3.org/2000/svg">
<style>
#metal3-bmh-seq{--bg:#fdfdfd;--lane:#ffffff;--band:#e4eaf4;--note:#fff1cc;--net:#c8420a;--mono:ui-monospace,SFMono-Regular,Menlo,monospace;color:#151414;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme: dark){#metal3-bmh-seq{--bg:#181a1b;--lane:#24282a;--band:#2b3441;--note:#3b2f12;--net:#ff8e55;color:#e8e6e3}}
</style>
<defs><marker id="m3b-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker><marker id="m3b-ahn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--net)"/></marker></defs>
<line x1="72" y1="78" x2="72" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="13" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="72" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">kubectl</text>
<text x="72" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">operator / GitOps</text>
<line x1="198" y1="78" x2="198" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="139" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="198" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">BMO</text>
<text x="198" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">BareMetalHost ctrl</text>
<line x1="324" y1="78" x2="324" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="265" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="324" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">Ironic</text>
<text x="324" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">API + conductor :6385</text>
<line x1="450" y1="78" x2="450" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="391" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="450" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">BMC</text>
<text x="450" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">IPMI / Redfish</text>
<line x1="576" y1="78" x2="576" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="517" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="576" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">dnsmasq / httpd</text>
<text x="576" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">DHCP · TFTP · HTTP 6180</text>
<line x1="702" y1="78" x2="702" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="643" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="702" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">DHCP relay</text>
<text x="702" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">router SVI, helper-address</text>
<line x1="828" y1="78" x2="828" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="769" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="828" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">Host firmware</text>
<text x="828" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">PXE ROM → iPXE</text>
<line x1="954" y1="78" x2="954" y2="1466" stroke="currentColor" stroke-opacity="0.22" stroke-width="1"/>
<rect x="895" y="14" width="118" height="46" rx="4" fill="var(--lane)" stroke="currentColor" stroke-opacity="0.35"/>
<text x="954" y="34" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">IPA → OS</text>
<text x="954" y="50" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity="0.7">agent in RAM, then disk</text>
<rect x="8" y="84" width="1010" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="102" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">0 · Inventory: happens once per host, before any image is chosen</text>
<line x1="72" y1="134" x2="198" y2="134" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="135" y="128" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">apply BareMetalHost + BMC Secret</text>
<line x1="198" y1="170" x2="324" y2="170" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="261" y="164" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">create node: driver, BMC addr, bootMAC</text>
<line x1="324" y1="206" x2="450" y2="206" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="387" y="200" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">boot=pxe (one-time), power on → inspection</text>
<line x1="828" y1="242" x2="954" y2="242" stroke="currentColor" stroke-width="1.3" stroke-dasharray="5 4" marker-end="url(#m3b-ah)"/>
<text x="882.76" y="236" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">same relayed PXE → iPXE → IPA chain as phase 3</text>
<line x1="954" y1="278" x2="324" y2="278" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="639" y="272" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST /v1/continue_inspection (inventory)</text>
<line x1="324" y1="314" x2="198" y2="314" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="261" y="308" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">inspected → BMO fills status.hardware</text>
<rect x="190" y="340" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="212" y="354" font-size="11" font-style="italic" text-anchor="start" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">BMH: Available · powered off</text>
<rect x="8" y="372" width="1010" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="390" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">1-2 · You set the spec by hand: in Part 3, Cluster API declares the cluster and CAPM3 claims the host here</text>
<line x1="72" y1="422" x2="198" y2="422" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="175.58" y="416" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">patch BMH.spec: image, userData, networkData, online=true</text>
<rect x="190" y="448" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="212" y="462" font-size="11" font-style="italic" text-anchor="start" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">userData Secret = your cloud-init · networkData Secret = optional static addressing</text>
<rect x="8" y="480" width="1010" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="498" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">3 · Ironic boots the agent: host on VLAN 20, Ironic on VLAN 10, DHCP crosses the router via the helper</text>
<line x1="198" y1="530" x2="324" y2="530" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="261" y="524" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">instance_info = image/checksum/configdrive; provision → active</text>
<line x1="324" y1="566" x2="576" y2="566" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="450" y="560" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">write pxelinux.cfg/&lt;mac&gt; (IPA boot script)</text>
<line x1="324" y1="602" x2="450" y2="602" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="387" y="596" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">set boot device PXE (one-time), power on</text>
<line x1="450" y1="638" x2="828" y2="638" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="639" y="632" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">power on, POST (5–10 min on big boxes)</text>
<line x1="828" y1="674" x2="702" y2="674" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="765" y="668" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">DHCPDISCOVER broadcast  opt60 PXEClient, opt93 arch</text>
<line x1="702" y1="710" x2="576" y2="710" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="639" y="704" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">unicast to helper target: giaddr=10.20.0.1, hops=1</text>
<line x1="576" y1="746" x2="702" y2="746" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="639" y="740" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">OFFER: range picked by giaddr, ip, router, next-server, snponly.efi</text>
<line x1="702" y1="782" x2="828" y2="782" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="765" y="776" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">OFFER re-broadcast on VLAN 20 (REQUEST/ACK repeat this path)</text>
<line x1="828" y1="818" x2="576" y2="818" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="702" y="812" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">TFTP GET iPXE binary: routed unicast, no relay involved</text>
<line x1="828" y1="854" x2="702" y2="854" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="765" y="848" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">iPXE DHCPs again (opt175) → relayed the same way</text>
<line x1="576" y1="890" x2="828" y2="890" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="702" y="884" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">filename = http://ironic:6180/boot.ipxe  (via relay)</text>
<line x1="828" y1="926" x2="576" y2="926" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="702" y="920" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">HTTP GET boot.ipxe → pxelinux.cfg/&lt;mac&gt; → IPA kernel + initramfs</text>
<line x1="828" y1="962" x2="954" y2="962" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="865.12" y="956" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">kernel boots IPA  (ipa-api-url, agent token, BOOTIF)</text>
<rect x="8" y="984" width="1010" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="1002" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">4 · The agent writes the image</text>
<line x1="954" y1="1034" x2="702" y2="1034" stroke="var(--net)" stroke-width="1.6" marker-end="url(#m3b-ahn)"/>
<text x="794.56" y="1028" font-size="11" text-anchor="middle" fill="var(--net)" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">Linux DHCP (relayed again): needs option 3 router to reach Ironic off-subnet</text>
<line x1="954" y1="1070" x2="324" y2="1070" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="639" y="1064" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST /v1/lookup, then heartbeat every ~10 s (callback_url :9999)</text>
<line x1="324" y1="1106" x2="954" y2="1106" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="639" y="1100" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST :9999/v1/commands  prepare_image {url, checksum, configdrive}</text>
<line x1="954" y1="1142" x2="576" y2="1142" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="765" y="1136" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">GET image.raw  (httpd cache or your image server)</text>
<rect x="946" y="1168" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="940" y="1182" font-size="11" font-style="italic" text-anchor="end" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">dd to rootDeviceHints device · write config-2 partition (user_data, meta_data, network_data)</text>
<line x1="954" y1="1214" x2="324" y2="1214" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="639" y="1208" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">heartbeat: deploy steps done</text>
<line x1="324" y1="1250" x2="450" y2="1250" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="387" y="1244" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">boot device = disk (persistent), reboot</text>
<line x1="324" y1="1286" x2="198" y2="1286" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="261" y="1280" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">node active → BMH Provisioned</text>
<rect x="8" y="1308" width="1010" height="28" rx="3" fill="var(--band)"/>
<text x="18" y="1326" font-size="12" font-weight="700" fill="currentColor" font-family="var(--mono)">5 · The OS boots from disk</text>
<line x1="450" y1="1358" x2="828" y2="1358" stroke="currentColor" stroke-width="1.3" marker-end="url(#m3b-ah)"/>
<text x="639" y="1352" font-size="11" text-anchor="middle" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">POST, boot from disk: IPA is gone</text>
<rect x="946" y="1384" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="940" y="1398" font-size="11" font-style="italic" text-anchor="end" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">cloud-init ConfigDrive: network_data → static IP on the baremetal net · user_data → whatever you asked for</text>
<rect x="190" y="1430" width="16" height="20" rx="3" fill="var(--note)" stroke="currentColor" stroke-opacity="0.5"/>
<text x="212" y="1444" font-size="11" font-style="italic" text-anchor="start" fill="currentColor" stroke="var(--bg)" stroke-width="4" paint-order="stroke" stroke-linejoin="round">later: remove spec.image → deprovisioning (IPA cleaning boot) → Available again</text>
</svg>
</div>

<p style="font-size:0.9em;opacity:.85;margin-top:-0.4em;">Orange arrows are network boot (DHCP, TFTP, HTTP); black arrows are Kubernetes API, Ironic API, and BMC calls; the dashed arrow stands in for the full network-boot chain; yellow boxes are local state changes. Phase 0 runs once per host; phases 1-5 run for every deploy. Scroll sideways on narrow screens.</p>

| Phase | What happens | Takes |
|---|---|---|
| **0 · Inventory** | You apply the `BareMetalHost` and its BMC Secret. BMO creates an Ironic node with the driver, BMC address, and boot MAC, and Ironic asks the BMC for a one-time PXE boot and powers the host on. The PXE → iPXE → IPA chain above runs once for inspection: IPA posts the hardware inventory to `/v1/continue_inspection`, BMO fills `status.hardware`, and the host ends `available`, powered off. | 10-20 min, once per host |
| **1-2 · Set the spec** | You patch the BMH with `image`, `userData`, `networkData`, and `online: true`. In [Part 3](https://hrishi.dev/linux/os/systems/2026/09/19/os-boot-process-3.html) this is two phases, Cluster API declaring a cluster and CAPM3 claiming a host and rendering the same fields into the BMH; the numbering here is kept so the two diagrams line up. | seconds |
| **3 · Network boot** | BMO copies image URL, checksum, and config drive into the Ironic node's `instance_info` and asks for provision state `active`. Ironic writes `pxelinux.cfg/<mac>`, asks the BMC for a one-time PXE boot, and powers the host on. POST, then the PXE → iPXE → IPA chain. | 3-10 min, mostly POST |
| **4 · Deploy** | IPA DHCPs (third relayed round), calls `POST /v1/lookup`, and heartbeats to Ironic every ~10 s. Ironic calls back on port 9999 with `prepare_image`; IPA streams the image to the `rootDeviceHints` device and writes the `config-2` partition holding `user_data`, `meta_data`, and `network_data`. Ironic sets boot=disk on the BMC, persistently this time, and reboots. The BMH turns `provisioned`. | 2-6 min, image-size bound |
| **5 · Boot from disk** | Stages 1-7 from the freshly written disk. cloud-init's ConfigDrive datasource applies `network_data` and runs whatever `user_data` says. | 3-10 min |

To release the host, remove `spec.image` (in Part 3, deleting the `Machine` does this for you): BMO runs a
cleaning boot of IPA to wipe the disks and the BMH returns to `available`.

### Mapping the Phases Back to the Seven Stages

| Phase | In [Part 1](https://hrishi.dev/linux/os/systems/2026/05/05/os-boot-process-1.html) terms |
|---|---|
| 3 | Stages 1-2 as before. Stage 3 is PXE ROM → iPXE instead of shim → GRUB. Stages 4-6 boot IPA's kernel, initramfs, and systemd. No Stage 7: nobody logs in. |
| 4 | Still inside the ramdisk, on a `tmpfs` root. The agent streams an image to a block device it found during Stage 4 driver probing. |
| 5 | The full Stage 1-7 sequence from the freshly written disk. cloud-init runs as ordinary systemd units in Stage 6 and reads the `config-2` partition IPA wrote. |

### The BareMetalHost State Machine

```
BareMetalHost.status.provisioning.state  (BMO)

 registering → inspecting* → preparing* → available → provisioning* → provisioned
                                             ▲                             │
                                             └──── deprovisioning* ◄───────┘   spec.image removed
 * = IPA is booted over the network in this state
```

A host stuck in `provisioning` past Ironic's 30-minute deploy-callback timeout means the BMC powered the box on but
IPA never called home: one of the orange arrows in the sequence above is broken.

### What Crosses the Wire

| Direction | Port | Carries | If blocked |
|---|---|---|---|
| Ironic → BMC | 623/udp IPMI, 443 Redfish | power, boot device, virtual media | BMH stuck `registering` |
| host → relay → dnsmasq | 67/68 udp | DHCP with `giaddr` set, three rounds | `PXE-E51 No DHCP offers` |
| host → dnsmasq | 69/udp | iPXE binary via TFTP | `PXE-E32 TFTP open timeout` |
| host → httpd | 6180/tcp | `boot.ipxe`, per-MAC script, IPA kernel and initramfs | iPXE `Could not chain` |
| IPA → Ironic | 6385/tcp | lookup, heartbeat, inspection callback | BMH sits in `provisioning` until timeout |
| Ironic → IPA | 9999/tcp | agent commands (`prepare_image`) | heartbeats fine, nothing happens |

The arrow most firewall policies get wrong is Ironic → IPA on 9999: Ironic has to open a connection *to* the host,
across the router. A policy that only allows host → Ironic passes every check up to the heartbeat and then stalls.

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

The network-boot path is the same seven stages from Part 1 with Stage 3 swapped out, run once for the agent and
once for the real OS. Every `PXE-Exx` code on the console and every stuck `provisioning` state maps back to one
specific hop in that sequence, which is what makes it debuggable: work down the table above and the first row
that fails names the hop.

> **Key Takeaways**
> - On a diskless host, Stage 3 becomes **PXE ROM → DHCP → TFTP → iPXE → DHCP again → HTTP → kernel + initramfs**, and the first Linux to boot is a RAM-only agent that installs the real one.
> - Only the four DHCP messages need a **relay** to cross VLANs, three rounds per provisioning; everything after is routed unicast that needs a router option to work.
> - Ironic's iPXE binaries are unsigned, so **Secure Boot** hosts need Redfish virtual media instead, which collapses the whole chain into one BMC call.
> - The `BareMetalHost` state machine does all the hardware work and needs no Kubernetes on the host it provisions: inspect once, then `provisioning → provisioned` per deploy.
> - The firewall rule most policies miss is **Ironic → IPA on 9999**: heartbeats look healthy and nothing happens.
> - A BMH stuck in `provisioning` past the 30-minute callback timeout means one network-boot hop is broken, and the console's `PXE-Exx` code names it.

## Up Next

One host is provisioned. **[Part 3: Automated Host Provisioning with Metal³, CAPM3, and Cluster API](https://hrishi.dev/linux/os/systems/2026/09/19/os-boot-process-3.html)** puts
Cluster API on top of the same `BareMetalHost` machinery, so a single `Cluster` manifest claims hosts from the
inventory, renders their cloud-init and network data, images them, and joins them as Kubernetes nodes.
