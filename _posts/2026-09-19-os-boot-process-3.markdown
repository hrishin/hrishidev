---
layout: post
title:  "From Power Button to Shell Prompt, Part 3: Automated Host Provisioning with Metal³, CAPM3, and Cluster API"
date:   2026-09-19 06:00:00 +0000
categories: [Linux, OS, Systems]
description: "How Cluster API, CAPM3, and the Bare Metal Operator turn one Cluster manifest into a Kubernetes cluster on bare metal: the full call sequence from kubectl apply to a Ready Node, the Metal3 object model and Metal3DataTemplate, the BareMetalHost and Machine state machines side by side, and where the Cluster API layer gets stuck."
image: /assets/os-boot-process-stages.png
---

*Part 3 of a 3-part series on the Linux boot process. [Part 1](https://hrishi.dev/linux/os/systems/2026/05/05/os-boot-process-1.html) covered the seven stages of a disk boot;
[Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html) took them over the network with PXE, iPXE, and Metal³'s Bare Metal Operator and Ironic. This final
part puts Cluster API on top so a single manifest provisions a whole cluster.*

---

## Introduction

**[Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html)** ended with one host: you wrote a `BareMetalHost` with `spec.image` and `spec.userData` by hand,
BMO and Ironic network-booted the Ironic Python Agent, and the agent wrote the OS. That is fine for a handful of
machines. For a Kubernetes cluster you want to declare "three control-plane nodes and ten workers with this image"
and have the right hosts claimed from the inventory, addressed, imaged, joined, and released back to the pool when
they are deleted. That is what **Cluster API (CAPI)** and its Metal³ provider **CAPM3** add. They never touch a
BMC or a DHCP server themselves: everything they do ends in a patch to a `BareMetalHost`, and the machinery from
Part 2 takes it from there.

| Component | Role |
|---|---|
| **Cluster API (CAPI)** | `Cluster`, `Machine`, `KubeadmControlPlane`. Renders the cloud-init that runs `kubeadm init` or `join`. |
| **CAPM3** | Metal³'s CAPI provider. Maps a `Machine` to a `BareMetalHost` and renders its metadata and network data. |
| **BMO** | Bare Metal Operator. Owns the `BareMetalHost` state machine; the only client of Ironic. |
| **Ironic, dnsmasq, httpd, DHCP relay, BMC, IPA** | The network-boot and image-write path from [Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html). Unchanged here. |

## Table of Contents

1. [The Metal3 Provisioning Sequence](#the-metal3-provisioning-sequence)
2. [Two State Machines Side by Side](#two-state-machines-side-by-side)
3. [The Metal3 Object Model](#the-metal3-object-model)
4. [Troubleshooting the Cluster API Layer](#troubleshooting-the-cluster-api-layer)

---

## The Metal3 Provisioning Sequence

Here is every call from `kubectl apply` of a `Cluster` to a physical server showing up as a Ready `Node`. The
host sits on VLAN 20 and Ironic on VLAN 10, so the DHCP hops cross the relay covered in [Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html#crossing-vlans-the-dhcp-relay).

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
| **0 · Inventory** ([Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html#from-baremetalhost-to-provisioned-host)) | You apply a `BareMetalHost` and BMC Secret. BMO registers it in Ironic, which PXE-boots IPA once to inspect the hardware. Host ends `available`, powered off. | 10-20 min, once per host |
| **1 · Declare** | You apply `Cluster`, `KubeadmControlPlane`, and the Metal³ templates. CAPI creates a `Machine` and a user-data Secret (cloud-init with `kubeadm init`). | seconds |
| **2 · Claim** | CAPM3 picks an `available` host matching `hostSelector`, renders its metadata and `network_data.json` (static IPs from an `IPPool`, NIC mapping from inspected MACs), and patches the BMH: `image`, `userData`, `networkData`, `online: true`. | seconds |
| **3 · Network boot** ([Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html#stage-3-over-the-network-pxe-dhcp-tftp-and-ipxe)) | BMO tells Ironic to deploy. Ironic writes `pxelinux.cfg/<mac>`, asks the BMC for a one-time PXE boot, and powers the host on. POST, then the PXE → iPXE → IPA chain. | 3-10 min, mostly POST |
| **4 · Deploy** ([Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html#from-baremetalhost-to-provisioned-host)) | IPA DHCPs (third relayed round), looks itself up, and heartbeats to Ironic. Ironic calls back on port 9999 with `prepare_image`; IPA streams the image to the `rootDeviceHints` device and writes the `config-2` partition. Ironic sets boot=disk on the BMC and reboots. BMH turns `provisioned`. | 2-6 min, image-size bound |
| **5 · Boot and join** | Stages 1-7 from disk ([Part 1](https://hrishi.dev/linux/os/systems/2026/05/05/os-boot-process-1.html)). cloud-init's ConfigDrive datasource applies the network data and runs `kubeadm`. kubelet registers with label `metal3.io/uuid`; CAPM3 finds the Node, sets `providerID`, and the `Machine` goes `Running`. | 3-10 min |

Phases 0, 3, and 4 are exactly the inventory, network boot, and deploy from Part 2, where you wrote `spec.image`
and `spec.userData` on the `BareMetalHost` yourself. The new work is phases 1 and 2, where Cluster API and CAPM3
decide *what* to write into the `BareMetalHost`, and the second half of phase 5, where the booted node is matched
back to its `Machine`.

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

### What Crosses the Wire, Beyond Part 2

The [Part 2 table](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html#what-crosses-the-wire) covers every port on the network-boot path. Cluster API adds one
more, and it is the one that tells you the boot succeeded but the join did not:

| Direction | Port | Carries | If blocked |
|---|---|---|---|
| OS → control plane | 6443/tcp | `kubeadm join` | BMH `provisioned`, Machine stuck `Provisioning` |

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
- **The label matters.** The kubelet must register with `node-labels: "metal3.io/uuid={% raw %}{{ ds.meta_data.uuid }}{% endraw %}"`;
  that is how CAPM3 finds the `Node` in phase 5. Without it the OS boots and joins, and the `Machine` never leaves
  `Provisioning`.
- **Names you can rely on.** `HardwareData` and the firmware objects share the BMH's name; the providerID is
  `metal3://<ns>/<bmh>/<metal3machine>` on both the `Metal3Machine` and the workload `Node`.

---

## Troubleshooting the Cluster API Layer

Start with the [network-boot table in Part 2](https://hrishi.dev/linux/os/systems/2026/09/18/os-boot-process-2.html#troubleshooting-a-network-boot) whenever the `BareMetalHost`
is anywhere other than `provisioned`. The rows below are for the case where BMO and Ironic did their job and the
`Machine` still does not reach `Running`.

| Symptom | Where to look | Usual cause |
|---|---|---|
| `Machine` stuck `Pending`, no `Metal3Machine` activity | `kubectl get kubeadmconfig -o yaml`, the `DataSecretAvailable` condition | The bootstrap Secret has not been rendered yet; for a worker, the control plane is not initialized |
| `Metal3Machine` never gets a host; CAPM3 logs `No available host found. Requeuing.` | `kubectl get bmh -A` state column; labels vs `hostSelector` in the `Metal3MachineTemplate` | No BMH in `available` matches `hostSelector`, or every matching host is already consumed by another cluster |
| Changed the image, existing nodes keep the old one, or the `Metal3MachineTemplate` edit is rejected | `spec.image` on each `Metal3Machine` vs the template; the `infrastructureRef` on the `KubeadmControlPlane` or `MachineDeployment` | Templates are immutable by design; a new template name referenced from the control plane or deployment triggers a rolling replacement from the pool |
| BMH `provisioned`, OS up, Machine stuck `Provisioning` | `cloud-init status --long`; is `/dev/disk/by-label/config-2` there? Node labels | Config drive not read, wrong link/MAC in `networkData`, or the `metal3.io/uuid` label missing |
| Node is `Ready` in the workload cluster, `Machine` stuck `Provisioning` | `kubectl get node -o jsonpath='{.spec.providerID}'` on the workload cluster; CAPM3 logs | `metal3.io/uuid` label missing from the kubelet, so CAPM3 cannot find the Node to set `providerID` |

```bash
# 1. Which layer is stuck?
kubectl get cluster,kubeadmcontrolplane,machinedeployment,machine,metal3machine -n metal3
kubectl get bmh -n metal3 -o custom-columns=NAME:.metadata.name,STATE:.status.provisioning.state,CONSUMER:.spec.consumerRef.name,ERR:.status.errorMessage
# 2. Did CAPM3 find a host, and what did it write?
kubectl -n capm3-system logs deploy/capm3-controller-manager | grep -iE 'available host|consumerRef|providerID'
kubectl get bmh -n metal3 <host> -o jsonpath='{.spec.image.url}{"\n"}{.spec.userData.name}{"\n"}'
# 3. Did the node come up with the right identity?
KUBECONFIG=workload.kubeconfig kubectl get nodes -L metal3.io/uuid -o custom-columns=NAME:.metadata.name,UUID:.metadata.labels.metal3\.io/uuid,PROVIDER:.spec.providerID
```

---

## Closing Thoughts

Three posts, one boot sequence. Part 1 followed it from the reset vector to a shell prompt on a machine with a disk.
Part 2 swapped Stage 3 for a network conversation and ran the sequence twice, once for a RAM-only agent and once for
the OS it wrote. This part added the layer that decides which hosts get which image and which cloud-init, and
matches the booted node back to the object that asked for it. Every layer above still ends in the same seven
stages, and when something goes wrong, the fastest question to ask is which stage, on which boot, on which hop.

> **Key Takeaways**
> - Cluster API and CAPM3 never touch hardware: every decision ends in a **patch to a `BareMetalHost`**, and BMO and Ironic (Part 2) do the rest.
> - One provisioned host is a spine of six objects, `Cluster → KubeadmControlPlane → Machine → Metal3Machine → BareMetalHost → Ironic node`, and everything from the `BareMetalHost` down works without the rows above it.
> - The `Metal3DataTemplate` is where inspected hardware becomes per-host cloud-init: static IPs from an `IPPool`, NIC mapping from inspected MACs.
> - The **`metal3.io/uuid` node label** is how CAPM3 finds the Node in phase 5; without it the OS boots and joins, and the `Machine` never leaves `Provisioning`.
> - **Inventory survives.** Deleting a `Machine` cleans the host and returns it to `available`; the `BareMetalHost`, its BMC Secret, the `IPPool`, and the templates stay.
> - Templates are immutable: a new image means a new `Metal3MachineTemplate` name and a rolling replacement from the pool.
