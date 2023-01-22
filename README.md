# Bisection Bandwidth Tester
This repository involves a unit test for measuring the bandwidth of communication of a group of processors (mainly GPUs). It is based on MPI and tests various capabilites such as CPU-Only MPI, GPU-Aware MPI, CPU-Staged MPI, NCCL, and IPC.

Porting the capabilities are controlled by preprocessor directives. With no specification, it targets CPU by default. To port on Nvidia GPUs, one needs to ```#define SCI_CUDA```. To port on AMD GPUs, one needs to ```#define SCI_HIP```. Please refer to the table at the bottom to enable desired capabilities.

There are two parameters to describe the logical group topology. The first one is the number of processors (p) and the second one is the group size (g). The benchmarking tool splits the global communicator ```MPI_COMM_WORLD``` into subcommunicators with ```MPI_Comm_split```. Eeach group talks to all other groups with a mapping between GPU as shown in the figure below.

![Group Examples](https://github.com/merthidayetoglu/OLCF_BW_test/blob/main/results/group_examples.png)

These partitioning scenarios can be applied to test communication bandwidth accross nodes, among GPUs within nodes, and between pairs of GPUs.

Considering a hierarchical communication network, MPI ranks are assumed to be assigned in as an SMP style. For example, if there are six GPUs and three nodes, GPU 0 and GPU 1 are in the same node and so GPU 2 and GPU3. In this test, each MPI rank runs a single GPU and user is responsible to place the ranks correctly. The first GPU of a group talks to the first GPU on each group, the second GPU of a group talks to the second GPU on each group, and so on. This test excludes the self communication.

This tool runs like
```cpp
mpirun -np #numproc Alltoall #count #numiter #groupsize
```
where count is the number of 32-byte words between two GPUs. The number of iterations is for averaging the bandwidth over many times. The program performs one warmup round and measures time over the number of iteration where each iteration are bulk-synchronized individually.

The figure below summarizes the Summit results. Note that all involved GPUs both sends and receives data and the measurement of the aggregate bidirectional bandwidth of a group is reported in GB/s.

![Summit Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/summit_bandwidth.png)

We use the default MPI implementation in the system. You can find more details in the dropdown menu on the bottom of the page. The table below summarizes the testing capabilities (where available).

| Porting Options   | Capability | Include |
| :---:               | ---: | :--- |
|Default is on CPU  | MPI | `#define MPI` |
|`#define SCI_CUDA` | CUDA-Aware MPI <br> CPU-Staged MPI <br> NCCL <br> CUDA IPC | `#define MPI` <br> `#define MPI_staged` <br> `#define NCCL` <br> `#define IPC` |
|`#define SCI_HIP`  | GPU-Aware MPI <br> CPU-Staged MPI <br> (AMD port) NCCL <br> HIP IPC | `#define MPI` <br> `#define MPI_staged` <br> `#define NCCL` <br> `#define IPC` |

Running on each system is like driving a different sports car, which has different handling and steering behaviour. This benchmarking tool helps understanding of the system characteristics. Our evaluation of various systems can be found below.

<details><summary>Summit Results (Cont'd)</summary>
<p>

Summit has [IBM Spectrum MPI](https://www.ibm.com/docs/en/SSZTET_EOS/eos/guide_101.pdf), which uses a lower-level transport layer called parallel active message interfece (PAMI). By default, PAMI variables are configured to have a lower latency [as reported here](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#spectrum-mpi-tunings-needed-for-maximum-bandwidth). Thanks [Chris Zimmer](https://www.olcf.ornl.gov/directory/staff-member/christopher-zimmer/) for pointing it out! To obtain full theoretical bandwidth, we set up the PAMI variables as:
```bash
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"
```

Results with default configuration is shown below (not to be confused with the full-bandwidth configuration that is shown above). We include the equation for calculating the theoretical bandwidth of the CPU-Staged mode.

![Summit Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/summit_latency.png)

NCCL performs irrespective of the PAMI configuration, because it uses UCX API across nodes. CUDA-Aware MPI breaks down with large message sizes due to a known problem.
  
[Summit User Guide](https://docs.olcf.ornl.gov/systems/summit_user_guide.html)
  
</p>
</details>


<details><summary>Crusher Results</summary>
<p>

Crusher is a testbed for Frontier&mdash;the first official exascale system. They have the same node architecture and software toolchain. It has Cray MPICH MPI implementation by default.
  
![Crusher Across Nodes](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/crusher_across_nodes.png)

![Crusher Within Nodes](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/crusher_within_nodes.png)

[Crusher User Guide](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html)
  
</p>
</details>


<details><summary>Perlmutter Results</summary>
<p>


![Perlmutter Bandwidth](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/perlmutter_bandwidth.png)

</p>
</details>

<details><summary>Delta Results</summary>
<p>

Delta is an NCSA system that is composed of multi-GPU nodes with four Nvidia A100 GPUs each. It has Slingshot 10 and runs OpenMPI+UCX by default.
  
![Delta Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/delta_measurement.png)

[Delta User Guide](https://wiki.ncsa.illinois.edu/display/DSC/Delta+User+Guide)
  
</p>
</details>


<details><summary>Spock Results</summary>
<p>
  
Spock is an experimental system at OLC that is composed of multi-GPU nodes with four AMD MI100 GPUs each. It has Slingshot 10 and runs Cray MPICH+OFI by default. We also tried Cray MPICH+UCX by loading modules `craype-network-ucx` and `cray-mpich-ucx`.

![Spock Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/spock_measurement.png)

The results below are taken within one node with the default MPI because Cray MPICH+UCX crahes with buffer size larger than 16 KB when GPUs are involved.

![Spock Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/spock_within_nodes.png)

[Spock User Guide](https://docs.olcf.ornl.gov/systems/spock_quick_start_guide.html)

<p>

</p>
</details>

<details><summary>ThetaGPU Results</summary>
<p>

ThetaGPU is an Nvidia DGX-A100 System with eight GPUs per node. The GPUs each GPU is connected to six NVSwitches via NVLinks, where each link has 100 GB/s bidirectional bandwidth. Considering the physical communication architecture, we can model the bisection bandwidth within a fully-connected topology, where each GPUs has a peak bandwidth of 600 GB/s. As a result, the bisection bandwidth of a group can be written as:
  
```math
\beta_{\textrm{group}}^{-1} = g\times600\textrm{ GB/s}
```
where g is the number of GPUs in each group. The figure below shows the bandwidth measurements with various configuration within the DGX-A100 node.

![ThetaGPU Measurement](https://github.com/merthidayetoglu/Bisection_Bandwidth_Tester/blob/main/results/thetaGPU_within_nodes.png)


[ThetaGPU User Guide](https://maps.app.goo.gl/GLmdk82YJF3EWeiJ9)

</p>
</details>

For reproducibility, we provide the preworked Makefiles and run scripts for various systems, including Summit, Crusher, Spock, Delta, and ThetaGPU in the scripts folder.

<table id="tg-htl9E">
<thead>
  <tr>
    <th colspan="6">Peak Bandwidth Across Nodes in GB/s</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td colspan="3">MPI</td>
    <td rowspan="2">NCCL</td>
    <td rowspan="2">Ideal</td>
  </tr>
  <tr>
    <td></td>
    <td>CPU-Only</td>
    <td>GPU-Aware</td>
    <td>CPU-Staged</td>
  </tr>
  <tr>
    <td>Spock (OFI)</td>
    <td>24.34</td>
    <td>18.11</td>
    <td>20.06</td>
    <td>13.60</td>
    <td>25</td>
  </tr>
  <tr>
    <td>Spock (UCX)</td>
    <td>21.22</td>
    <td>20.48</td>
    <td>19.75</td>
    <td>13.19</td>
    <td>25</td>
  </tr>
  <tr>
    <td>Delta (UCX)</td>
    <td>24.11</td>
    <td>20.53</td>
    <td></td>
    <td>23.73</td>
    <td>25</td>
  </tr>
  <tr>
    <td>Summit (PAMI)</td>
    <td>44.50</td>
    <td>38.91</td>
    <td>39.87</td>
    <td>30.92</td>
    <td>50</td>
  </tr>
  <tr>
    <td>Perlmutter (OFI)</td>
    <td>84.52</td>
    <td>66.22</td>
    <td>52.45</td>
    <td></td>
    <td>200</td>
  </tr>
  <tr>
    <td>Crusher (OFI)</td>
    <td>59.49</td>
    <td>115.89</td>
    <td>58.75</td>
    <td>4.96</td>
    <td>200</td>
  </tr>
  <tr>
    <td>ThetaGPU (UCX)</td>
    <td>32.94</td>
    <td>11.91</td>
    <td>26.28</td>
    <td>277.45</td>
    <td>400</td>
  </tr>
</tbody>
</table>
<script charset="utf-8">var TGSort=window.TGSort||function(n){"use strict";function r(n){return n?n.length:0}function t(n,t,e,o=0){for(e=r(n);o<e;++o)t(n[o],o)}function e(n){return n.split("").reverse().join("")}function o(n){var e=n[0];return t(n,function(n){for(;!n.startsWith(e);)e=e.substring(0,r(e)-1)}),r(e)}function u(n,r,e=[]){return t(n,function(n){r(n)&&e.push(n)}),e}var a=parseFloat;function i(n,r){return function(t){var e="";return t.replace(n,function(n,t,o){return e=t.replace(r,"")+"."+(o||"").substring(1)}),a(e)}}var s=i(/^(?:\s*)([+-]?(?:\d+)(?:,\d{3})*)(\.\d*)?$/g,/,/g),c=i(/^(?:\s*)([+-]?(?:\d+)(?:\.\d{3})*)(,\d*)?$/g,/\./g);function f(n){var t=a(n);return!isNaN(t)&&r(""+t)+1>=r(n)?t:NaN}function d(n){var e=[],o=n;return t([f,s,c],function(u){var a=[],i=[];t(n,function(n,r){r=u(n),a.push(r),r||i.push(n)}),r(i)<r(o)&&(o=i,e=a)}),r(u(o,function(n){return n==o[0]}))==r(o)?e:[]}function v(n){if("TABLE"==n.nodeName){for(var a=function(r){var e,o,u=[],a=[];return function n(r,e){e(r),t(r.childNodes,function(r){n(r,e)})}(n,function(n){"TR"==(o=n.nodeName)?(e=[],u.push(e),a.push(n)):"TD"!=o&&"TH"!=o||e.push(n)}),[u,a]}(),i=a[0],s=a[1],c=r(i),f=c>1&&r(i[0])<r(i[1])?1:0,v=f+1,p=i[f],h=r(p),l=[],g=[],N=[],m=v;m<c;++m){for(var T=0;T<h;++T){r(g)<h&&g.push([]);var C=i[m][T],L=C.textContent||C.innerText||"";g[T].push(L.trim())}N.push(m-v)}t(p,function(n,t){l[t]=0;var a=n.classList;a.add("tg-sort-header"),n.addEventListener("click",function(){var n=l[t];!function(){for(var n=0;n<h;++n){var r=p[n].classList;r.remove("tg-sort-asc"),r.remove("tg-sort-desc"),l[n]=0}}(),(n=1==n?-1:+!n)&&a.add(n>0?"tg-sort-asc":"tg-sort-desc"),l[t]=n;var i,f=g[t],m=function(r,t){return n*f[r].localeCompare(f[t])||n*(r-t)},T=function(n){var t=d(n);if(!r(t)){var u=o(n),a=o(n.map(e));t=d(n.map(function(n){return n.substring(u,r(n)-a)}))}return t}(f);(r(T)||r(T=r(u(i=f.map(Date.parse),isNaN))?[]:i))&&(m=function(r,t){var e=T[r],o=T[t],u=isNaN(e),a=isNaN(o);return u&&a?0:u?-n:a?n:e>o?n:e<o?-n:n*(r-t)});var C,L=N.slice();L.sort(m);for(var E=v;E<c;++E)(C=s[E].parentNode).removeChild(s[E]);for(E=v;E<c;++E)C.appendChild(s[v+L[E-v]])})})}}n.addEventListener("DOMContentLoaded",function(){for(var t=n.getElementsByClassName("tg"),e=0;e<r(t);++e)try{v(t[e])}catch(n){}})}(document)</script>

Please send and email to [merth@stanford.edu](merth@stanford.edu) for any questions or contributions. Especially, extension of this tool for benchmarking GASNet would be great!
