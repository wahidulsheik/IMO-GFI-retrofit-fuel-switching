"""
Code_06_Figures.py
Draws Figures 1-5 of the manuscript in a Times typeface (TeX Gyre Termes, a Times clone; if Times New Roman
is installed it is used instead). Run Code_02 and Code_03 first: Figures 3, 4 and 5 read their results from ./results/.
"""
import os, json, logging, matplotlib
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from Code_01_Model import HERE, targets, PATH, adopts, T
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
FONT = 'Times New Roman'
if not any(f.name == FONT for f in fm.fontManager.ttflist):
    for f in ['regular', 'bold', 'italic', 'bolditalic']:
        p = f'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{f}.otf'
        if os.path.exists(p): fm.fontManager.addfont(p)
    FONT = 'TeX Gyre Termes'
plt.rcParams.update({'font.family':FONT,'font.size':13,'mathtext.fontset':'custom','mathtext.rm':FONT,'mathtext.it':FONT+':italic','mathtext.bf':FONT+':bold'})
GREY='#8a8a85'; BLUE='#2b78d6'; ORANGE='#f26b38'; GREEN='#1aab7a'
def clean(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)

# ---------- Figure 1: EET screening flow ----------
fig,ax=plt.subplots(figsize=(10,10.2),dpi=150); ax.set_xlim(0,100); ax.set_ylim(0,102); ax.axis('off')
def box(x,y,w,h,txt,fc,ec,bold=False,rad=0.6,fs=13,rot=0,tc='#111'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rad}',fc=fc,ec=ec,lw=1.6))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',rotation=rot,color=tc,linespacing=1.25)
SIDE='#4aa8dc'; GBOX='#d9d9d9'; EDGE='#1f2a7a'
box(2,80,5,18,'Identification','#4aa8dc','#4aa8dc',True,rot=90,fs=14,tc='#1f2a7a')
box(2,30,5,44,'Screening using exclusion criteria','#4aa8dc','#4aa8dc',True,rot=90,fs=14,tc='#1f2a7a')
box(2,1,5,18,'Selection','#4aa8dc','#4aa8dc',True,rot=90,fs=14,tc='#1f2a7a')
box(15,82,28,16,'Identification of EETs\nfrom DNV (n)=37',GBOX,EDGE,True,rad=3)
rows=[(67,'Items removed (n)=3\nRemaining items (n)=34','Exclusion criteria: Asset Compatibility\n(Filter 1)'),
      (55,'Items removed (n)=2\nRemaining items (n)=32','Exclusion criteria: Retrofit Feasibility\n(Filter 2)'),
      (42.5,'Items removed (n)=11\nRemaining items (n)=21','Exclusion criteria: Commercial Viability\n(Filter 3)'),
      (30,'Items removed (n)=9\nRemaining items (n)=12','Exclusion criteria: Typological De-\nduplication and Portfolio Stratification\n(Filter 4)')]
for y,l,r in rows:
    box(12,y,35,7,l,GBOX,'#222')
    box(56,y-0.5,42,8,r,GBOX,EDGE,True,rad=1.2,fs=12.5)
    ax.annotate('',xy=(47.3,y+3.5),xytext=(56,y+3.5),arrowprops=dict(arrowstyle='-|>',color='#1a3fc4',lw=1.8))
for a,b in [(82,74),(67,62),(55,49.5),(42.5,37)]:
    ax.annotate('',xy=(29.5,b),xytext=(29.5,a),arrowprops=dict(arrowstyle='-|>',color='#1a3fc4',lw=1.8))
ax.annotate('',xy=(29.5,19),xytext=(29.5,30),arrowprops=dict(arrowstyle='-|>',color='#1a3fc4',lw=1.8))
box(15,1,28,18,'Items used in\noptimisation model\n(n)=12',GBOX,'#222',True,rad=0.3)
fig.savefig(os.path.join(FIG,'Figure_1.png'),bbox_inches='tight'); plt.close(fig)

# ---------- Figure 2: GFI trajectories (S2) ----------
BB={1:4,2:6,3:8,4:12.4,5:16.8,6:21.2,7:25.6,8:30,9:37,10:44}
yrs=list(range(2028,2038)); G=93.3
base=[G*(1-BB[t]/100) for t in range(1,11)]; dct=[G*(1-(BB[t]+13)/100) for t in range(1,11)]
fig,ax=plt.subplots(figsize=(13.3,8.2),dpi=150)
ax.fill_between(yrs,base,112,color='#fbe9e9',zorder=0); ax.fill_between(yrs,dct,base,color='#fdf1de',zorder=0); ax.fill_between(yrs,0,dct,color='#e6f5ef',zorder=0)
ax.plot(yrs,base,color='#111',lw=2.6); ax.plot(yrs,dct,color='#111',lw=2.6,ls=(0,(5,3)))
fuels=[('Ammonia (fossil)',103,GREY,(0,(8,3,1,3)),'#555',0),('Methanol (fossil)',95,GREY,':','#555',2.5),('Heavy fuel oil',94,BLUE,'-',BLUE,-2.4),
       ('LNG',85,ORANGE,'-',ORANGE,0),('Methanol (renewable)',8,GREEN,':',GREEN,2),('Ammonia (renewable)',5,GREEN,(0,(8,3,1,3)),GREEN,-2.2)]
for n,g,c,ls,tc,dy in fuels:
    ax.plot(yrs,[g]*10,color=c,ls=ls,lw=3 if n=='Heavy fuel oil' else 2.2)
    ax.text(2037.15,g+dy,f'{n}  {g}',va='center',color=tc,fontsize=14)
ax.text(2037.15,base[-1],'Base Target',style='italic',va='center',fontsize=14)
ax.text(2037.15,dct[-1],'Direct Compliance Target',style='italic',va='center',fontsize=14)
ax.text(2032.5,81,'Tier 2',color='#a33',weight='bold',style='italic',fontsize=17)
ax.text(2032.5,65.5,'Tier 1',color='#8a6d00',weight='bold',style='italic',fontsize=17)
ax.text(2032,24.5,'Compliant',color='#0e6b45',weight='bold',style='italic',fontsize=17)
ax.set_xlim(2028,2039.9); ax.set_ylim(0,114); ax.set_xticks(yrs)
ax.set_xlabel('Calendar year',fontsize=15); ax.set_ylabel(r'Well-to-wake GHG fuel intensity (gCO$_2$eq/MJ)',fontsize=15)
ax.grid(color='#d4d4d4',lw=0.8,zorder=0.5); clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'Figure_2.png')); plt.close(fig)

# ---------- Figure 3 ----------
labs=['Static\nminimax cost','Adaptive\nminimax cost','Static\nminimax regret','Adaptive\nminimax regret']
t7=pd.read_csv(os.path.join(RES,'Table7_regret.csv')).set_index('formulation')['worst_case']
vals=[t7['Static, minimax cost'],t7['Adaptive, minimax cost'],t7['Static, minimax regret'],t7['Adaptive, minimax regret']]; cols=[GREY,GREY,BLUE,BLUE]
pct=lambda a,b: round(100*(1-b/a))
fig,ax=plt.subplots(figsize=(10,4.9),dpi=150); y=list(range(4))[::-1]
ax.barh(y,vals,color=cols,height=0.58)
for yi,v in zip(y,vals): ax.text(v+5,yi,f'{v:,.1f}',va='center',fontsize=13)
ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlim(0,340); ax.set_xlabel('Worst-case regret (thousand USD)')
ax.grid(axis='x',color='#e3e3e3'); ax.set_axisbelow(True); clean(ax)
for yy,t in [(y[1],f'recourse under cost\ncriterion: {pct(vals[0],vals[1])}% lower'),(y[2],f'criterion: {pct(vals[0],vals[2])}% lower\nthan static minimax cost'),(y[3],f'recourse: a further\n{pct(vals[2],vals[3])}% reduction')]:
    ax.text(200,yy,t,style='italic',color='#555',va='center',fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'Figure_3.png')); plt.close(fig)

# ---------- Figure 4 ----------
t8=pd.read_csv(os.path.join(RES,'Table8A_thresholds.csv')).set_index('pathway')
short={'LNG':'LNG','Ammonia (fossil)':'Ammonia (fossil)','Methanol (fossil)':'Methanol (fossil)','Ammonia (renewable)':'Ammonia (renew.)','Methanol (renewable)':'Methanol (renew.)'}
rows=[(short[n],t8.loc[n,'price_at_present'],t8.loc[n,'threshold']) for n in ['LNG','Ammonia (fossil)','Methanol (fossil)','Ammonia (renewable)','Methanol (renewable)']]
rows=sorted(rows,key=lambda r: r[2]/r[1]-1,reverse=True)
fig,ax=plt.subplots(figsize=(10,5.2),dpi=150)
for k,(n,p,t) in enumerate(rows):
    yy=len(rows)-1-k
    ax.plot([t,p],[yy,yy],color=GREY,lw=2.2,zorder=1)
    ax.scatter([p],[yy],color=ORANGE,s=130,zorder=2,label='Price at present' if k==0 else None)
    ax.scatter([t],[yy],color=GREEN,s=130,zorder=2,label='Adoption threshold' if k==0 else None)
    ax.text(p+0.18,yy,f'{round((t/p-1)*100):d}%'.replace('-','−'),va='center',fontsize=13)
ax.axvline(1,ls=':',color='#222',lw=1.3); ax.text(1.07,len(rows)-0.55,'HFO parity',style='italic',color='#555',fontsize=12)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(0.2,7.6); ax.set_ylim(-0.6,len(rows)-0.3); ax.set_xlabel('Delivered energy-equivalent price (× heavy fuel oil)')
ax.grid(axis='x',color='#e3e3e3'); ax.set_axisbelow(True); clean(ax); ax.legend(loc='upper right',frameon=False,fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'Figure_4.png')); plt.close(fig)

# ---------- Figure 5 ----------
b1=pd.read_csv(os.path.join(RES,'TableB1_ammonia_sweep.csv')); b4=pd.read_csv(os.path.join(RES,'TableB4_tightening.csv'))
D={'B1':[(m_,None,r_) for m_,r_ in zip(b1['multiple'],b1['worst_case_regret'])],'pace':[(p_,c_) for p_,c_ in zip(b4['pace'],b4['TCO'])],
   'cap':[(c,) for c in [1000,2000,3000,4000,6000,8000,10000,12000,14000]]}
for c,_ in [(c,None) for c in [1000,2000,3000,4000,6000,8000,10000,12000,14000]]:
    assert not any(adopts(r,'NH3_r',2.33,fuelcap={'NH3':c}) for r in ['S2','S3']) and all(adopts(r,'NH3_r',1.26,fuelcap={'NH3':c}) for r in ['S2','S3'])
fig,axs=plt.subplots(1,3,figsize=(15.5,5),dpi=150)
thA=float(pd.read_csv(os.path.join(RES,'Table8A_thresholds.csv')).set_index('pathway').loc['Ammonia (renewable)','th_S2'])
a=axs[0]; xs=[r[0] for r in D['B1']]; ys=[max(r[2],0) for r in D['B1']]
a.axvspan(thA,0.95,color='#e2f4ec'); a.plot(xs,ys,'-o',color=BLUE,lw=2.2,ms=7); a.invert_xaxis()
a.axvline(thA,color='#111',ls='--',lw=1.4); a.set_xlim(2.4,0.95); a.set_ylim(-3,53)
a.text(2.0,45,'declined',style='italic',color='#555'); a.text(1.22,12,'adopted',style='italic',color='#0e7a50')
a.text(1.24,30,'threshold\n1.26×',fontsize=12)
a.set_xlabel('Renewable ammonia price (× HFO energy-equivalent)'); a.set_ylabel('Worst-case regret (thousand USD)')
a.set_title('(a) Renewable ammonia price',weight='bold',loc='left',fontsize=14)
a=axs[1]; caps=[r[0]/1000 for r in D['cap']]
a.scatter(caps,[1]*len(caps),s=110,facecolors='white',edgecolors=GREY,lw=2); a.scatter(caps,[0]*len(caps),s=110,color=GREEN)
a.set_yticks([0,1]); a.set_yticklabels(['Adopted','Declined']); a.set_ylim(-0.6,1.6)
a.text(7.5,1.3,'at prevailing price (2.33×)',ha='center',color='#555',fontsize=12); a.text(7.5,-0.3,'at threshold price (1.26×)',ha='center',color='#0e7a50',fontsize=12)
a.set_xlabel('Conversion capital (million USD)'); a.set_title('(b) Conversion capital',weight='bold',loc='left',fontsize=14)
a=axs[2]; px=[r[0] for r in D['pace']]; py=[r[1]/1000 for r in D['pace']]
a.plot(px,py,'-s',color=ORANGE,lw=2.2,ms=7); a.text(1.05,70.55,'Fuel conversion declined\nat every pace tested',style='italic',color='#555',fontsize=12)
a.set_xlabel('Post-2035 tightening pace (× S2 trajectory)'); a.set_ylabel('Total cost of ownership (million USD)')
a.set_title('(c) Regulatory stringency',weight='bold',loc='left',fontsize=14)
for a in axs: a.grid(color='#e3e3e3'); a.set_axisbelow(True); clean(a)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'Figure_5.png')); plt.close(fig)
print('Figures written to', FIG)
