#!/usr/bin/env python3
"""Typography-only renderer for KAIS Figures 2--6.

Reads frozen summaries and layer files. It contains no clustering, training,
parameter search, model fitting, or metric implementation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kais-figure-r8/mpl-cache")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.optimize import linear_sum_assignment

REPO = Path("/home/emarinho/workspace/programs/rtknnc")
ARCH = REPO / "paper_figures_archive/article_20260826"
OLD = REPO / "paper/KAIS"
OUT = Path("/tmp/kais-figure-r8/candidates")
PREV = Path("/tmp/kais-figure-r8/previews")
OUT.mkdir(parents=True, exist_ok=True)
PREV.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.0,
    "axes.labelsize": 9.0, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.transparent": False,
})

def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def save(fig, stem):
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(
        pdf, format="pdf", facecolor="white",
        metadata={"CreationDate": None, "ModDate": None,
                  "Creator": "KAIS-R8 frozen-data typography renderer"},
    )
    fig.savefig(PREV / f"{stem}.png", dpi=180, facecolor="white")
    plt.close(fig)
    return pdf

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_layer(path):
    cluster, point_id, x0, x1 = [], [], [], []
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            left, right = line.split(":", 1)
            fields = right.split()
            # Native layer identifiers are one-based; the archived renderer
            # converted them to zero-based labels before colour alignment.
            cluster.append(int(left) - 1)
            point_id.append(int(fields[0]))
            x0.append(float(fields[1])); x1.append(float(fields[2]))
    order = np.argsort(point_id)
    return tuple(np.asarray(values)[order] for values in (cluster, x0, x1))

def frozen_reference(dataset):
    path = REPO / "csv/synthetic" / f"{dataset}.csv"
    table = np.genfromtxt(path, delimiter=",", names=True)
    names = table.dtype.names or ()
    coords = np.column_stack([table[n] for n in names if n.startswith("x")])
    return coords[:, :2], np.asarray(table["label"], dtype=int)

def fig2():
    src = ARCH / "experimental/multiscale"
    summary = rows(src / "blobs_var_density_multiscale_summary.csv")
    overlaps = rows(src / "blobs_var_density_multiscale_overlaps.csv")
    kvals = [int(r["K"]) for r in summary]
    counts = [int(r["n_clusters"]) for r in summary]
    ari = [float(r["ARI"]) for r in summary]
    nmi = [float(r["NMI"]) for r in summary]
    saturation = int(summary[0]["tested_range_saturation_K"])

    # Keep the configured 8--10 pt typography, but use the inter-panel whitespace
    # more efficiently so the figure and its manuscript caption fit one page.
    fig = plt.figure(figsize=(4.8, 6.15))
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 1.38), hspace=0.76)
    ax1, ax2, ax3 = (fig.add_subplot(grid[i, 0]) for i in range(3))
    stable = "#e6e6e6"
    ax1.axvspan(saturation-.25, 16.25, color=stable, zorder=0)
    ax1.plot(kvals, counts, "o-", color="#1f77b4", lw=1.6, ms=4)
    ax1.axhline(5, color="#555555", ls="--", lw=1, label=r"reference count $C^*=5$")
    ax1.set_yscale("log"); ax1.set_ylim(3.2, 900); ax1.set_xlim(1.7, 16.3)
    ax1.set_ylabel(r"Clusters $m_K$ (log scale)")
    ax1.set_title("(a) Cluster-count evolution", loc="left", fontweight="semibold")
    ax1.grid(True, which="both", alpha=.22); ax1.legend(frameon=False, loc="upper right")
    ax1.set_xticks(range(2, 17, 2))
    for k, value in zip(kvals[:7], counts[:7]):
        if k == 2 or value != counts[k-3]:
            offset = (5, -1) if k == 2 else (0, 5)
            ax1.annotate(str(value), (k, value), xytext=offset,
                         textcoords="offset points", fontsize=8,
                         ha="left" if k == 2 else "center")
    ax1.text(12, 7.2, "partition unchanged\n" + r"$K=8,\ldots,16$", fontsize=8, ha="center")

    ax2.axvspan(saturation-.25, 16.25, color=stable, zorder=0)
    ax2.plot(kvals, ari, "o-", color="#d95f02", lw=1.5, ms=4, label="ARI")
    ax2.plot(kvals, nmi, "s-", color="#1b9e77", lw=1.5, ms=3.5, label="NMI")
    ax2.set(xlim=(1.7,16.3), ylim=(0,1.02), xlabel=r"Neighbourhood scale $K$", ylabel="External score")
    ax2.set_xticks(range(2,17,2)); ax2.grid(True, alpha=.22)
    ax2.set_title("(b) Agreement with reference labels", loc="left", fontweight="semibold")
    ax2.legend(frameon=False, loc="lower right")
    ax2.annotate(r"$m_7=C^*=5$, but ARI=0.7805", xy=(7,ari[5]), xytext=(8.2,.55),
                 arrowprops={"arrowstyle":"->","color":"#555","lw":.8}, fontsize=8)

    ks = (5,6,7,8); xpos = {k:i for i,k in enumerate(ks)}
    sizes = {k:{} for k in ks}
    links = []
    for r in overlaps:
        sk,dk,sc,dc = map(int,(r["src_k"],r["dst_k"],r["src_cluster"],r["dst_cluster"]))
        sizes[sk][sc]=int(r["src_size"]); sizes[dk][dc]=int(r["dst_size"])
        links.append((sk,dk,sc,dc,int(r["overlap"])))
    # Terminal lineage follows the frozen merge links; this only controls colour.
    terminal = {}
    for k in reversed(ks):
        for c in sizes[k]:
            if k == 8: terminal[k,c]=c
            else:
                dst = next(dc for sk,dk,sc,dc,_ in links if sk==k and sc==c)
                terminal[k,c]=terminal[k+1,dst]
    # Stable ordering groups source nodes by terminal lineage and cluster id.
    order = {k:sorted(sizes[k], key=lambda c:(terminal[k,c],c)) for k in ks}
    gap=.006; scale=.80/2000.0; intervals={}
    for k in ks:
        used=sum(sizes[k].values())*scale+gap*(len(order[k])-1); cursor=(1-used)/2
        for c in order[k]:
            height=sizes[k][c]*scale; intervals[k,c]=(cursor,cursor+height); cursor+=height+gap
    incoming=defaultdict(list)
    for link in links: incoming[link[1],link[3]].append(link)
    slices={}
    for key, group in incoming.items():
        cursor=intervals[key][0]
        for sk,dk,sc,dc,overlap in sorted(group,key=lambda x:intervals[x[0],x[2]][0]):
            height=overlap*scale; slices[sk,sc,dk]=(cursor,cursor+height); cursor+=height
    palette=plt.get_cmap("tab10"); colours={c:palette(i) for i,c in enumerate(order[8])}
    for sk,dk,sc,dc,_ in links:
        s0,s1=intervals[sk,sc]; d0,d1=slices[sk,sc,dk]; x0=xpos[sk]+.055; x1=xpos[dk]-.055
        cl=x0+.42*(x1-x0); cr=x0+.58*(x1-x0)
        verts=[(x0,s0),(cl,s0),(cr,d0),(x1,d0),(x1,d1),(cr,d1),(cl,s1),(x0,s1),(x0,s0)]
        codes=[MplPath.MOVETO]+[MplPath.CURVE4]*3+[MplPath.LINETO]+[MplPath.CURVE4]*3+[MplPath.CLOSEPOLY]
        ax3.add_patch(PathPatch(MplPath(verts,codes),facecolor=colours[terminal[sk,sc]],edgecolor="none",alpha=.35))
    for k in ks:
        for c in order[k]:
            lo,hi=intervals[k,c]
            ax3.add_patch(Rectangle((xpos[k]-.055,lo),.11,hi-lo,facecolor=colours[terminal[k,c]],edgecolor="white",lw=.5,zorder=3))
            if k==8: ax3.text(xpos[k]+.09,(lo+hi)/2,str(sizes[k][c]),fontsize=8,va="center")
        ax3.text(xpos[k],1.025,f"$K={k}$\n{len(order[k])} clusters",fontsize=8,ha="center",va="bottom")
    ax3.set(xlim=(-.25,3.45),ylim=(-.02,1.12),xlabel=r"Neighbourhood scale $K$")
    ax3.set_xticks(range(4),labels=[str(k) for k in ks]); ax3.set_yticks([])
    ax3.spines[["left","right","top"]].set_visible(False)
    ax3.set_title("(c) Nested K-layer transition forest",loc="left",fontweight="semibold",pad=18)
    fig.subplots_adjust(top=.96,bottom=.045,left=.15,right=.94)
    return save(fig,"Fig2_multiscale_transition_typography")

DATASETS=("anisotropic","blobs","blobs_var_density","bridge","circles","moons","swiss_roll","uniform")
DLABEL={"anisotropic":"Anisotropic blobs","blobs":"Equal-density blobs","blobs_var_density":"Variable-density blobs","bridge":"Sparse bridge","circles":"Concentric circles","moons":"Two moons","swiss_roll":"Swiss roll","uniform":"Uniform cloud"}
MODELS=("uniform","inv_r","inv_r2","sigmoid")
STYLES={"uniform":("uniform $h(u)=1$","#0072B2","o","-",-.15),"inv_r":("inverse $h(u)=1/u$","#E69F00","s","--",-.05),"inv_r2":("inverse-square $h(u)=1/u^2$","#D55E00","D","-.",.05),"sigmoid":(r"sigmoid $S_\alpha(u)$","#009E73","^",":",.15)}
REF={"anisotropic":3,"blobs":4,"blobs_var_density":5,"bridge":3,"circles":2,"moons":2,"swiss_roll":6,"uniform":None}
YT=(1,2,3,5,10,30,100,300,1000)

def fig3():
    data=rows(ARCH/"data/vote_model_benchmark_long.csv")
    counts={(r["dataset"],r["vote_model"],int(r["K"])):int(r["n_clusters"]) for r in data if r["dataset"] in DATASETS and r["vote_model"] in MODELS}
    assert len(counts)==8*4*15
    fig,axes=plt.subplots(4,2,figsize=(4.8,6.45),sharex=True,sharey=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for model in MODELS:
            label,color,marker,ls,offset=STYLES[model]; vals=[counts[dataset,model,k] for k in range(2,17)]
            ax.plot(range(2,17),vals,color=color,ls=ls,lw=1.1,alpha=.86)
            ax.scatter([k+offset for k in range(2,17)],vals,color=color,marker=marker,s=13,lw=.35,edgecolors="white",zorder=3)
        ref=REF[dataset]
        if ref:
            ax.axhline(ref,color="#333",ls=(0,(4,3)),lw=.9)
            ax.text(16.25,ref*1.09,rf"$C^*={ref}$",fontsize=8,ha="right",va="bottom")
        else: ax.text(.97,.08,"no reference partition",transform=ax.transAxes,fontsize=8,ha="right",color="#555")
        ax.set_title(DLABEL[dataset],fontsize=9,fontweight="semibold",pad=2)
        ax.set_yscale("log"); ax.set_xlim(1.5,16.5); ax.set_ylim(.82,1150); ax.set_xticks((2,4,6,8,10,12,14,16))
        ax.yaxis.set_major_locator(FixedLocator(YT)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:str(int(v)) if v in YT else "")); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major",color="#c7c7c7",lw=.45,alpha=.65); ax.grid(which="minor",axis="y",color="#ddd",lw=.3,alpha=.4)
    handles=[Line2D([0],[0],color=STYLES[m][1],ls=STYLES[m][3],marker=STYLES[m][2],markersize=4,label=STYLES[m][0]) for m in MODELS]
    handles.append(Line2D([0],[0],color="#333",ls=(0,(4,3)),lw=1,label=r"reference count $C^*$"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.955),ncol=2,frameon=False,fontsize=8)
    fig.suptitle("Unpruned RTKNNC cluster count across neighbourhood scales",fontsize=9.2,fontweight="semibold",y=.995)
    fig.supxlabel(r"Neighbourhood size $K$",fontsize=9,y=.018); fig.supylabel("Number of clusters (log scale)",fontsize=9,x=.02)
    fig.tight_layout(rect=(.055,.045,1,.885),h_pad=.75,w_pad=.5)
    return save(fig,"Fig3_all_cluster_counts_typography")

CB=("#9A9A9A","#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#F0E442","#332288","#88CCEE","#44AA99","#117733","#AA4499")

def fig4():
    layerdir=ARCH/"experimental/layers"; bench=rows(ARCH/"data/vote_model_benchmark_long.csv")
    selected={r["vote_model"]:r for r in bench if r["dataset"]=="moons" and int(r["K"])==10}; assert set(selected)==set(MODELS)
    layers={m:next(layerdir.glob(f"moons_{m}_K10Cs*.layer")) for m in MODELS}
    # Preserve the GNUplot source's 2-row by 3-column panel correspondence.
    fig,axes=plt.subplots(2,3,figsize=(4.8,5.5)); axes=axes.flat
    titles=[
        "reference labels",
        "budget-matched\nuniform\n$h(u)=1$\n$C=2$; ARI=1; NMI=1",
        "budget-matched\ninverse\n$h(u)=1/u$\n$C=2$; ARI=1; NMI=1",
        "reference\ninverse-square\n$g(r)=1/r^2$\n$C=2$; ARI=1; NMI=1",
        "budget-matched\nsigmoid\n" + r"$h(u)=S_\alpha(u)$, $\alpha=8$" + "\n$C=2$; ARI=1; NMI=1",
    ]
    models=("uniform","uniform","inv_r","inv_r2","sigmoid")
    reference_xy, reference = frozen_reference("moons")
    for i,(ax,model,title) in enumerate(zip(axes,models,titles)):
        cl,x,y=read_layer(layers[model])
        assert np.allclose(reference_xy[:,0],x) and np.allclose(reference_xy[:,1],y)
        ids=reference if i==0 else cl
        colours=[CB[1+int(v)%12] for v in ids]
        ax.scatter(x,y,c=colours,s=4.5,edgecolors="none"); ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel(r"$x_0$",fontsize=8); ax.set_ylabel(r"$x_1$",fontsize=8)
        ax.set_title(title,fontsize=8,pad=3); ax.grid(False)
        ax.tick_params(labelsize=8)
    note=axes[5]; note.axis("off")
    note.text(-.10,.97,"Four budget-\nmatched shapes;\ninverse-square\nis the reference.\nColours identify\nclusters independently\nin each panel.\nGrey: unclustered\npoint.\nSame KNN graph and\nsource vote budget.",ha="left",va="top",fontsize=8,linespacing=1.04,color="#30343B")
    fig.suptitle("Two moons: vote-model comparison at $K=10$",fontsize=9.5,fontweight="semibold",y=.985)
    fig.subplots_adjust(left=.10,right=.975,bottom=.07,top=.76,hspace=.85,wspace=.48)
    return save(fig,"Fig4_moons_vote_models_typography")

def align_colour_ids(labels,truth):
    p=np.unique(labels); t=np.unique(truth); matrix=np.zeros((len(p),len(t)),dtype=int)
    for i,a in enumerate(p):
        for j,b in enumerate(t): matrix[i,j]=np.count_nonzero((labels==a)&(truth==b))
    ri,ci=linear_sum_assignment(-matrix); mapping={int(p[i]):int(t[j]) for i,j in zip(ri,ci)}
    nextid=int(t.max())+1
    for a in p:
        if int(a) not in mapping: mapping[int(a)]=nextid; nextid+=1
    return np.asarray([mapping[int(a)] for a in labels])

def structural(dataset,k,stem,suptitle,metric_lines,edge_lines,height):
    layerdir=ARCH/"experimental/structural_pruning/layers"; kernels=("inv_r","inv_r2","sigmoid")
    titles=(r"budget-matched inverse $h(u)=1/u$",r"reference inverse-square $g(r)=1/r^2$",r"budget-matched sigmoid $h(u)=S_\alpha(u)$, $\alpha=8$")
    names=(f"{dataset}_K{k}_inv_r_pruned.layer.csv",f"{dataset}_K{k}_pruned.layer.csv",f"{dataset}_K{k}_sigmoid_pruned.layer.csv")
    if dataset == "blobs_var_density":
        fig=plt.figure(figsize=(4.8,height))
        grid=fig.add_gridspec(2,4,left=.10,right=.97,bottom=.07,top=.80,hspace=.76,wspace=.60)
        axes=(fig.add_subplot(grid[0,0:2]),fig.add_subplot(grid[0,2:4]),fig.add_subplot(grid[1,1:3]))
    else:
        fig,axes=plt.subplots(3,1,figsize=(4.8,height))
    reference_xy, reference = frozen_reference(dataset)
    for ax,kernel,title,name,metric,edge in zip(axes,kernels,titles,names,metric_lines,edge_lines):
        labels,x,y=read_layer(layerdir/name); aligned=align_colour_ids(labels,reference)
        assert np.allclose(reference_xy[:,0],x) and np.allclose(reference_xy[:,1],y)
        colours=[plt.get_cmap("tab10")(int(v)%10) for v in aligned]
        ax.scatter(x,y,c=colours,s=5.2,alpha=.82,edgecolors="none")
        ax.set_aspect("equal",adjustable="box"); ax.set(xlabel=r"$x_0$",ylabel=r"$x_1$")
        display_title = title.replace(" $", "\n$", 1) if dataset == "blobs_var_density" else title
        ax.set_title(display_title+"\n"+metric+"\n"+edge,fontsize=8.2,pad=3); ax.tick_params(labelsize=8)
    fig.suptitle(suptitle,fontsize=9.5,fontweight="semibold",y=.985,linespacing=1.1)
    if dataset != "blobs_var_density":
        fig.subplots_adjust(left=.12,right=.97,bottom=.045,top=.80,hspace=.95)
    return save(fig,stem)

def main():
    products=[fig2(),fig3(),fig4()]
    products.append(structural("bridge",6,"Fig5_bridge_pruning_typography","Sparse bridge at $K=6$:\nnative pruning under three vote-allocation shapes",
        ("$C=3$, ARI=0.98532, NMI=0.96244","$C=3$, ARI=0.98535, NMI=0.96212","$C=3$, ARI=0.98535, NMI=0.96212"),
        (r"baseline $|E^{vis}|=6{,}940$; cuts=6",r"baseline $|E^{vis}|=6{,}928$; cuts=10",r"baseline $|E^{vis}|=6{,}972$; cuts=11"),6.35))
    products.append(structural("blobs_var_density",8,"Fig6_blobs_pruning_typography","Variable-density blobs at $K=8$:\nnative pruning under three vote-allocation shapes",
        ("$C=5$, ARI=0.96265, NMI=0.96141",)*3,
        (r"baseline $|E^{vis}|=9{,}402$; cuts=12",r"baseline $|E^{vis}|=9{,}365$; cuts=12",r"baseline $|E^{vis}|=9{,}470$; cuts=12"),6.4))
    print(json.dumps({p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in products},indent=2))

if __name__=="__main__": main()
