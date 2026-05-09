"""
Energy Management System — GUI  v8
====================================
Tkinter multi-page UI wrapping the v2 engine.

Pages:  Startup → Inputs+Thresholds → Summary → Optimize
Recovery:
  v1 CLI → energy_management_v1_backup.py
  v2 CLI → energy_management_v2_backup.py

Run:  python energy_management_gui8.py
Deps: same as v5  (tkinter is bundled with Python)
"""
from __future__ import annotations
import os, sys, time, warnings, pickle, threading, queue, platform
import importlib.util
from pathlib import Path
from typing import Any
warnings.filterwarnings("ignore")

# ── dependency check ──────────────────────────────────────────────────────────
REQUIRED = {"pandas":"pandas","numpy":"numpy","sklearn":"scikit-learn",
            "joblib":"joblib","gymnasium":"gymnasium",
            "stable_baselines3":"stable-baselines3[extra]","torch":"torch"}
missing = [pkg for mod,pkg in REQUIRED.items() if not importlib.util.find_spec(mod)]
if missing:
    print("Missing deps:\n  pip install", " ".join(missing)); sys.exit(1)

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn
from torch.utils.data import DataLoader as TorchDL, TensorDataset
import gymnasium as gym
from gymnasium import spaces
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "1945dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MLR_PATH        = MODELS_DIR / "mlr_model.joblib"
MLP_PATH        = MODELS_DIR / "mlp_model.joblib"
RNN_PATH        = MODELS_DIR / "rnn_model.pt"
RNN_SCALER_PATH = MODELS_DIR / "rnn_scaler.joblib"   # ← scaler saved separately
SCALER_PATH     = MODELS_DIR / "scaler.joblib"
RL_PATH         = MODELS_DIR / "rl_agent.zip"
META_PATH       = MODELS_DIR / "meta.pkl"
RL_TIMESTEPS    = 1_000_000

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#1a1d27"
CARD    = "#22263a"
BORDER  = "#2e3250"
ACCENT  = "#4f8ef7"
ACCENT2 = "#7c5cbf"
GREEN   = "#3ecf8e"
RED     = "#f05252"
YELLOW  = "#f5a623"
TEXT    = "#e8eaf0"
SUBTEXT = "#8b90a8"
WHITE   = "#ffffff"

# ── column heuristics ─────────────────────────────────────────────────────────
CO2_KW   = ["co2","carbon","emission","co₂"]
COST_KW  = ["cost","bill","energy_cost","electricity_cost","price"]
EFF_KW   = ["efficiency","eff","energy_eff","power_eff"]
FAULT_KW = ["fault","failure","error","anomaly","defect","alarm"]
EXCL_KW  = ["id","timestamp","date","time","index","unnamed"]
RO_KW    = ["date","time","day","month","year","hour","week",
            "temperature","temp","ambient","weather","humidity",
            "outdoor","season","solar_irradiance","irradiance",
            "efficiency","eff","cost","bill","price",
            "co2","carbon","emission","fault","failure","error",
            "demand","load_demand","grid_frequency","frequency"]

def _match(col:str, kws:list[str]) -> bool:
    """Match keyword against column name as a whole word (underscore/space bounded)."""
    import re
    c = col.lower().replace(" ","_")
    for k in kws:
        # exact match, or keyword is a complete token within the column name
        if re.search(r'(^|_)' + re.escape(k) + r'(_|$)', c):
            return True
    return False

def _best_match(df: pd.DataFrame, cols: list, kws: list[str]):
    """Return the column matching kws with the largest value range (avoids picking normalised proxies)."""
    candidates = [c for c in cols if _match(c, kws)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # prefer the one with the largest std — most likely the real target column
    return max(candidates, key=lambda c: df[c].dropna().std() if pd.api.types.is_numeric_dtype(df[c]) else 0)

def identify_columns(df:pd.DataFrame) -> dict[str,Any]:
    cols = list(df.columns)
    co2   = _best_match(df, cols, CO2_KW)
    cost  = _best_match(df, cols, COST_KW)
    eff   = _best_match(df, cols, EFF_KW)
    fault = _best_match(df, cols, FAULT_KW)
    targets = {co2,cost,eff,fault}-{None}
    feats = [c for c in cols if c not in targets
             and not _match(c,EXCL_KW) and pd.api.types.is_numeric_dtype(df[c])]
    ctrl  = [c for c in feats if not _match(c,RO_KW)]
    return {"co2_col":co2,"cost_col":cost,"eff_col":eff,"fault_col":fault,
            "feature_cols":feats,"controllable_cols":ctrl}

# ═══════════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════════
# --- PLACEHOLDER_ENGINE ---
class DataLoader:
    def __init__(self, path:Path):
        self.path=path; self.df_raw=None; self.col_map={}
        self.scaler=StandardScaler()
        self.X_train=self.X_test=self.X_test_raw=None
        self.y_train={}; self.y_test={}
        self.feature_cols=[]; self.controllable_cols=[]; self.controllable_indices=[]

    def load(self):
        self.df_raw = pd.read_csv(self.path)
        self.col_map = identify_columns(self.df_raw)
        cm = self.col_map
        # diagnostic — prints to terminal so you can verify column detection
        print("\n[DataLoader] Column detection:")
        for key in ["co2_col","cost_col","eff_col","fault_col"]:
            col = cm[key]
            if col and col in self.df_raw.columns:
                s = self.df_raw[col].dropna()
                print(f"  {key:12s} -> '{col}'  min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}")
            else:
                print(f"  {key:12s} -> NOT DETECTED")
        print()
        miss = [k for k,v in cm.items()
                if k not in("feature_cols","controllable_cols") and v is None]
        if miss: self._fallback()
        return self

    def _fallback(self):
        nc = [c for c in self.df_raw.columns if pd.api.types.is_numeric_dtype(self.df_raw[c])]
        cm = self.col_map; tail = nc[-4:] if len(nc)>=4 else nc
        for i,k in enumerate(["co2_col","cost_col","eff_col","fault_col"]):
            if cm[k] is None and i<len(tail): cm[k]=tail[i]
        ts={cm["co2_col"],cm["cost_col"],cm["eff_col"],cm["fault_col"]}-{None}
        cm["feature_cols"]=[c for c in nc if c not in ts and not _match(c,EXCL_KW)]
        cm["controllable_cols"]=[c for c in cm["feature_cols"] if not _match(c,RO_KW)]

    def preprocess(self):
        df=self.df_raw.copy(); cm=self.col_map
        tc=[v for k,v in cm.items() if k not in("feature_cols","controllable_cols") and v]
        df=df.dropna(subset=tc)
        self.feature_cols=cm["feature_cols"]
        self.controllable_cols=cm["controllable_cols"] or cm["feature_cols"]
        for c in self.feature_cols:
            if c in df.columns: df[c]=df[c].fillna(df[c].median())
        if not self.controllable_cols: self.controllable_cols=self.feature_cols
        self.controllable_indices=[self.feature_cols.index(c)
                                   for c in self.controllable_cols if c in self.feature_cols]
        X=df[self.feature_cols].values.astype(np.float32)
        fc=cm["fault_col"]
        if fc and fc in df.columns:
            fv=df[fc].values
            yf=fv.astype(int) if set(np.unique(fv)).issubset({0,1,0.,1.}) \
               else (fv>np.median(fv)).astype(int)
        else: yf=np.zeros(len(df),int)
        idx=np.arange(len(X))
        Xtr,Xte,itr,ite=train_test_split(X,idx,test_size=0.2,random_state=42)
        self.X_train=self.scaler.fit_transform(Xtr)
        self.X_test =self.scaler.transform(Xte)
        self.X_test_raw=Xte
        def _g(col): return df[col].values.astype(np.float32) if col and col in df.columns \
                             else np.zeros(len(df),np.float32)
        c2,co,ef=_g(cm["co2_col"]),_g(cm["cost_col"]),_g(cm["eff_col"])
        self.y_train={"co2":c2[itr],"cost":co[itr],"eff":ef[itr],"fault":yf[itr]}
        self.y_test ={"co2":c2[ite],"cost":co[ite],"eff":ef[ite],"fault":yf[ite]}
        return self

# --- PLACEHOLDER_ENGINE2 ---

class EnergyLSTM(nn.Module):
    def __init__(self,n:int,h:int=64,l:int=2):
        super().__init__()
        self.lstm=nn.LSTM(1,h,l,batch_first=True,dropout=0.2)
        self.fc=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,x):
        o,_=self.lstm(x.unsqueeze(-1)); return self.fc(o[:,-1,:])


class RNNEngine:
    def __init__(self,data:DataLoader):
        self.data=data; self.model=None
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_scaler=StandardScaler()

    def train(self,epochs=30,batch_size=64,progress_cb=None):
        d=self.data
        ytr=np.column_stack([d.y_train["co2"],d.y_train["cost"],d.y_train["eff"]])
        yte=np.column_stack([d.y_test["co2"], d.y_test["cost"], d.y_test["eff"]])
        ytrs=self.target_scaler.fit_transform(ytr)
        Xtt=torch.tensor(d.X_train,dtype=torch.float32)
        ytt=torch.tensor(ytrs,dtype=torch.float32)
        ldr=TorchDL(TensorDataset(Xtt,ytt),batch_size=batch_size,shuffle=True)
        self.model=EnergyLSTM(d.X_train.shape[1]).to(self.device)
        opt=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        crit=nn.MSELoss()
        for ep in range(epochs):
            self.model.train()
            for xb,yb in ldr:
                xb,yb=xb.to(self.device),yb.to(self.device)
                opt.zero_grad(); crit(self.model(xb),yb).backward(); opt.step()
            if progress_cb: progress_cb(ep+1,epochs,"RNN")
        self.model.eval(); return self

    def save(self):
        # Save weights and n_features only — scaler saved separately via joblib
        torch.save({"state_dict":self.model.state_dict(),
                    "n_features":self.data.X_train.shape[1]}, RNN_PATH)
        joblib.dump(self.target_scaler, RNN_SCALER_PATH)

    def load(self, n_features:int):
        ck=torch.load(RNN_PATH, map_location=self.device, weights_only=True)
        self.model=EnergyLSTM(ck["n_features"]).to(self.device)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        self.target_scaler=joblib.load(RNN_SCALER_PATH)
        return self

    def predict(self,x_scaled:np.ndarray) -> tuple[float,float,float]:
        self.model.eval()
        t=torch.tensor(x_scaled.reshape(1,-1),dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out=self.target_scaler.inverse_transform(self.model(t).cpu().numpy())[0]
        eff_val=float(out[2])
        if eff_val>1.0: eff_val/=100.0
        return float(out[0]),float(out[1]),eff_val

# --- PLACEHOLDER_ENGINE3 ---

class SupervisedEngine:
    def __init__(self,data:DataLoader):
        self.data=data; self.mlr=None; self.mlp=None; self.scaler=data.scaler

    def train(self,progress_cb=None):
        d=self.data
        yr=np.column_stack([d.y_train["co2"],d.y_train["cost"],d.y_train["eff"]])
        self.mlr=LinearRegression(); self.mlr.fit(d.X_train,yr)
        if progress_cb: progress_cb(1,2,"MLR")
        self.mlp=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=300,
                               random_state=42,early_stopping=True)
        self.mlp.fit(d.X_train,d.y_train["fault"])
        if progress_cb: progress_cb(2,2,"MLP")
        return self

    def save(self):
        joblib.dump(self.mlr,MLR_PATH); joblib.dump(self.mlp,MLP_PATH)
        joblib.dump(self.scaler,SCALER_PATH)

    def load(self):
        self.mlr=joblib.load(MLR_PATH); self.mlp=joblib.load(MLP_PATH)
        self.scaler=joblib.load(SCALER_PATH); return self

    def predict_state(self,raw:np.ndarray) -> dict[str,float]:
        xs=self.scaler.transform(raw.reshape(1,-1).astype(np.float32))
        rp=self.mlr.predict(xs)[0]
        fp=int(self.mlp.predict(xs)[0])
        fprob=float(self.mlp.predict_proba(xs)[0][1])
        eff_val=float(rp[2])
        if eff_val>1.0: eff_val/=100.0
        return {"co2_emission_30d":float(rp[0]),"cost_energy_bill_30d":float(rp[1]),
                "current_energy_efficiency":eff_val,
                "fault_detected":fp,"fault_probability":fprob}


class EnergyEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,sup,X_pool,feat_names,ctrl_idx,thresholds,scale=0.1,var_bounds=None):
        super().__init__()
        self.sup=sup; self.X_pool=X_pool; self.ctrl_idx=ctrl_idx
        self.thr=thresholds; self.scale=scale
        # var_bounds: dict {feature_index: (lo, hi)} — None means (0, +inf)
        self.var_bounds=var_bounds or {}
        self.n=X_pool.shape[1]; self.na=len(ctrl_idx)
        self.observation_space=spaces.Box(-5.,5.,(self.n,),np.float32)
        self.action_space=spaces.Box(-1.,1.,(self.na,),np.float32)
        self._raw=None; self._step=0; self._max=50
        self._std=X_pool.std(axis=0)+1e-8

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self._raw=self.X_pool[self.np_random.integers(0,len(self.X_pool))].copy().astype(np.float32)
        self._step=0
        return self.sup.scaler.transform(self._raw.reshape(1,-1))[0].astype(np.float32),{}

    def step(self,action):
        self._step+=1; nr=self._raw.copy()
        for i,idx in enumerate(self.ctrl_idx):
            nr[idx]+=action[i]*self.scale*self._std[idx]
            lo,hi=self.var_bounds.get(idx,(0.,np.inf))
            nr[idx]=float(np.clip(nr[idx],lo,hi))
        try: p=self.sup.predict_state(nr)
        except Exception:
            return self.sup.scaler.transform(self._raw.reshape(1,-1))[0].astype(np.float32),-10.,False,False,{}
        t=self.thr; r=0.
        if p["co2_emission_30d"]>t["max_co2"]:          r-=5.*((p["co2_emission_30d"]-t["max_co2"])/(t["max_co2"]+1e-8))
        if p["current_energy_efficiency"]<t["min_eff"]:  r-=5.*((t["min_eff"]-p["current_energy_efficiency"])/(t["min_eff"]+1e-8))
        if p["cost_energy_bill_30d"]>t["max_cost"]:      r-=2.*((p["cost_energy_bill_30d"]-t["max_cost"])/(t["max_cost"]+1e-8))
        r-=0.1*float(np.linalg.norm(action))
        if p["co2_emission_30d"]<=t["max_co2"] and p["current_energy_efficiency"]>=t["min_eff"]: r+=2.
        self._raw=nr
        return self.sup.scaler.transform(nr.reshape(1,-1))[0].astype(np.float32),r,self._step>=self._max,False,{}

# --- PLACEHOLDER_ENGINE4 ---

class RLOptimizer:
    def __init__(self,sup,data,thresholds,active_indices=None,var_bounds=None):
        self.sup=sup; self.data=data; self.thr=thresholds; self.agent=None
        self.active=active_indices if active_indices is not None else data.controllable_indices
        # var_bounds: dict {feature_index: (lo, hi)}
        self.var_bounds=var_bounds or {}

    def _env(self):
        return EnergyEnv(self.sup,self.data.X_test_raw,
                         self.data.feature_cols,self.active,self.thr,
                         var_bounds=self.var_bounds)

    def train(self,progress_cb=None):
        env=make_vec_env(self._env,n_envs=1)
        self.agent=PPO("MlpPolicy",env,verbose=0,
                       n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4)
        chunk=max(RL_TIMESTEPS//20,256); done=0
        while done<RL_TIMESTEPS:
            s=min(chunk,RL_TIMESTEPS-done)
            self.agent.learn(total_timesteps=s,reset_num_timesteps=(done==0))
            done+=s
            if progress_cb: progress_cb(done,RL_TIMESTEPS,"RL")
        return self

    def save(self): self.agent.save(str(RL_PATH.with_suffix("")))

    def load(self):
        env=make_vec_env(self._env,n_envs=1)
        self.agent=PPO.load(str(RL_PATH.with_suffix("")),env=env); return self

    def suggest(self,inputs:np.ndarray) -> np.ndarray:
        x=inputs.astype(np.float32)
        obs=self.sup.scaler.transform(x.reshape(1,-1))[0].astype(np.float32)
        env=self._env(); env._raw=x.copy()
        best,br=x.copy(),-np.inf
        for _ in range(20):
            a,_=self.agent.predict(obs,deterministic=True)
            obs,r,done,trunc,_=env.step(a)
            if r>br: br=r; best=env._raw.copy()
            if done or trunc: break
        # final clip to enforce bounds on all controllable indices
        for idx,(lo,hi) in self.var_bounds.items():
            best[idx]=float(np.clip(best[idx],lo,hi))
        # ensure no controllable output is negative (default floor = 0)
        for idx in self.active:
            if idx not in self.var_bounds:
                best[idx]=max(0.,best[idx])
        return best


def models_cached() -> bool:
    return all(p.exists() for p in [MLR_PATH,MLP_PATH,RNN_PATH,RNN_SCALER_PATH,SCALER_PATH,RL_PATH])

def save_meta(cm,fc,cc,ci):
    with open(META_PATH,"wb") as f:
        pickle.dump({"col_map":cm,"feature_cols":fc,"controllable_cols":cc,"controllable_indices":ci},f)


# ═══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════════════════════
IS_MAC = platform.system() == "Darwin"

def _fr(parent, bg=CARD, **kw) -> tk.Frame:
    return tk.Frame(parent, bg=bg, **kw)

def _lbl(parent, text, size=10, bold=False, color=TEXT, **kw) -> tk.Label:
    return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                    font=("Helvetica" if IS_MAC else "Segoe UI", size,
                          "bold" if bold else "normal"), **kw)

def _btn(parent, text, cmd, bg=ACCENT, fg=WHITE, size=10, width=16) -> tk.Button:
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                  activebackground=bg, activeforeground=fg,
                  relief="flat", cursor="hand2",
                  font=("Helvetica" if IS_MAC else "Segoe UI", size, "bold"),
                  width=width, padx=8, pady=5)
    def _hl(c): b.config(bg=c)
    lighter = "#{:02x}{:02x}{:02x}".format(
        min(255,int(bg[1:3],16)+30), min(255,int(bg[3:5],16)+30), min(255,int(bg[5:7],16)+30))
    b.bind("<Enter>", lambda e: _hl(lighter))
    b.bind("<Leave>", lambda e: _hl(bg))
    return b

def _entry(parent, var, width=16) -> tk.Entry:
    e = tk.Entry(parent, textvariable=var, width=width,
                 bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                 relief="solid", bd=1, font=("Helvetica" if IS_MAC else "Segoe UI", 10))
    return e

def _sep(parent) -> tk.Frame:
    return tk.Frame(parent, bg=BORDER, height=1)

# --- PLACEHOLDER_UI ---

def _scrollable(parent) -> tuple[tk.Canvas, tk.Frame]:
    """Cross-platform scrollable frame."""
    canvas = tk.Canvas(parent, bg=CARD, highlightthickness=0, bd=0)
    vsb    = tk.Scrollbar(parent, orient="vertical", command=canvas.yview,
                          bg=SURFACE, troughcolor=BG, relief="flat")
    inner  = tk.Frame(canvas, bg=CARD)
    win_id = canvas.create_window((0,0), window=inner, anchor="nw")

    def _resize_inner(e):
        canvas.configure(scrollregion=canvas.bbox("all"))
    def _resize_canvas(e):
        canvas.itemconfig(win_id, width=e.width)

    inner.bind("<Configure>", _resize_inner)
    canvas.bind("<Configure>", _resize_canvas)
    canvas.configure(yscrollcommand=vsb.set)

    # Platform-specific scroll
    if IS_MAC:
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(-1*e.delta, "units"))
    else:
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*e.delta/120), "units"))
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1,"units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1,"units"))

    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    return canvas, inner


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Startup
# ═══════════════════════════════════════════════════════════════════════════════
class StartupPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master = master
        _fr(self, bg=BG).pack(pady=50)
        _lbl(self, "⚡  ENERGY MANAGEMENT SYSTEM", size=20, bold=True,
             color=YELLOW).pack()
        _lbl(self, "Predictive & Optimization Engine  |  ESG Compliance",
             size=11, color=SUBTEXT).pack(pady=(4,0))
        _lbl(self, "MLR · MLP · RNN/LSTM · PPO (RL)",
             size=10, color=ACCENT).pack(pady=(2,0))
        _sep(self).pack(fill="x", padx=80, pady=20)

        card = _fr(self, bg=CARD)
        card.pack(padx=100, fill="x", ipady=20)
        self.status_lbl = _lbl(card, "Initialising...", size=11, color=SUBTEXT)
        self.status_lbl.pack(pady=(16,6))

        # Progress bar — configure style once here
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("G.Horizontal.TProgressbar",
                        troughcolor=SURFACE, background=GREEN,
                        bordercolor=BORDER, lightcolor=GREEN, darkcolor=GREEN)
        self.pvar = tk.DoubleVar(value=0)
        ttk.Progressbar(card, variable=self.pvar, maximum=100, length=380,
                        style="G.Horizontal.TProgressbar").pack(pady=(0,10))
        self.detail_lbl = _lbl(card, "", size=9, color=SUBTEXT)
        self.detail_lbl.pack(pady=(0,8))
        # This button is hidden until models are ready
        self._go_btn = _btn(card, "Go to Inputs  →", self._go,
                            bg=GREEN, fg=BG, size=11, width=18)
        self._go_btn.pack(pady=(0,16))
        self._go_btn.pack_forget()  # hidden initially

        _sep(self).pack(fill="x", padx=80, pady=12)
        _lbl(self, "v3  |  backups: v1_backup.py  v2_backup.py",
             size=8, color=BORDER).pack()

    def start(self):
        """Called by App.__init__ to kick off the background worker."""
        self.after(300, self._run)

    def _go(self):
        """Navigate to the inputs page (button callback)."""
        self.master.show("inputs")

# --- PLACEHOLDER_STARTUP_RUN ---

    def _run(self):
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                if not DATA_PATH.exists():
                    q.put(("err", f"Dataset not found:\n{DATA_PATH}\n\nPlace 1945dataset.csv next to this script.")); return
                q.put(("s","Loading dataset...")); q.put(("p",5,""))
                data = DataLoader(DATA_PATH).load().preprocess()
                q.put(("p",12,"Dataset ready"))

                if models_cached():
                    q.put(("s","Loading cached models..."))
                    sup = SupervisedEngine(data); sup.load(); data.scaler=sup.scaler
                    q.put(("p",40,"MLR + MLP loaded"))
                    rnn = RNNEngine(data); rnn.load(data.X_train.shape[1])
                    q.put(("p",65,"RNN loaded"))
                    rl  = RLOptimizer(sup,data,{"max_co2":400,"max_cost":1000,"min_eff":0.5})
                    rl.load()
                    q.put(("p",95,"RL agent loaded"))
                else:
                    q.put(("s","First run — training models (~15 min)..."))
                    def pcb(d,t,n): q.put(("p",12+int(d/t*28),f"Training {n}…"))
                    sup=SupervisedEngine(data); sup.train(progress_cb=pcb); sup.save()
                    q.put(("p",42,"MLR + MLP done"))
                    def rcb(ep,t,n): q.put(("p",42+int(ep/t*23),f"RNN epoch {ep}/{t}"))
                    rnn=RNNEngine(data); rnn.train(progress_cb=rcb); rnn.save()
                    q.put(("p",66,"RNN done"))
                    def lcb(d,t,n): q.put(("p",66+int(d/t*28),f"RL step {d}/{t}"))
                    rl=RLOptimizer(sup,data,{"max_co2":400,"max_cost":1000,"min_eff":0.5})
                    rl.train(progress_cb=lcb); rl.save()
                    save_meta(data.col_map,data.feature_cols,
                              data.controllable_cols,data.controllable_indices)

                q.put(("p",100,"Ready"))
                q.put(("done",data,sup,rnn,rl))
            except Exception:
                import traceback; q.put(("err",traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()
        self._poll(q)

    def _poll(self, q):
        try:
            while True:
                m = q.get_nowait()
                if   m[0]=="s":    self.status_lbl.config(text=m[1])
                elif m[0]=="p":
                    self.pvar.set(m[1]); self.detail_lbl.config(text=m[2])
                    if m[1] >= 100:
                        self.status_lbl.config(text="Models ready!", fg=GREEN)
                        self._go_btn.pack(pady=(0,16))   # show the button
                elif m[0]=="done": self.master.on_ready(*m[1:]); return
                elif m[0]=="err":
                    messagebox.showerror("Startup Error", m[1])
                    self.master.destroy(); return
        except queue.Empty:
            pass
        self.after(120, lambda: self._poll(q))


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Inputs + Thresholds
# ═══════════════════════════════════════════════════════════════════════════════
class InputPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master = master
        self._vars: dict[str, tk.StringVar] = {}
        self._thr:  dict[str, tk.StringVar] = {}
        self._built = False

    def build(self):
        if self._built: return
        self._built = True
        app = self.master

        # top bar
        top = _fr(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(14,0))
        _lbl(top, "Hardware Inputs & ESG Thresholds", size=15, bold=True,
             color=WHITE).pack(side="left")
        _btn(top, "⟳  Random Sample", self._random,
             bg=ACCENT2, width=15).pack(side="right")

        _sep(self).pack(fill="x", padx=20, pady=8)

# --- PLACEHOLDER_INPUT2 ---

        # two-column body
        body = _fr(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── left: variable inputs ─────────────────────────────────────────────
        lw = _fr(body, bg=BG)
        lw.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        _lbl(lw, "Hardware Variables", size=11, bold=True,
             color=ACCENT).pack(anchor="w", pady=(0,4))
        lcard = _fr(lw, bg=CARD)
        lcard.pack(fill="both", expand=True)
        _, self._inner = _scrollable(lcard)

        # ── right: thresholds ─────────────────────────────────────────────────
        rw = _fr(body, bg=BG)
        rw.grid(row=0, column=1, sticky="nsew")
        _lbl(rw, "ESG Thresholds", size=11, bold=True,
             color=YELLOW).pack(anchor="w", pady=(0,4))
        rc = _fr(rw, bg=CARD)
        rc.pack(fill="x")
        for key, lbl_text, default in [
            ("max_co2",  "Max CO₂ (30d)",        "400.0"),
            ("max_cost", "Max Energy Cost (30d)", "1000.0"),
            ("min_eff",  "Min Efficiency",        "0.5"),
        ]:
            row = _fr(rc, bg=CARD)
            row.pack(fill="x", padx=12, pady=8)
            _lbl(row, lbl_text, size=10, color=TEXT).pack(anchor="w")
            v = tk.StringVar(value=default)
            self._thr[key] = v
            _entry(row, v, width=14).pack(anchor="w", pady=(2,0))

        # bottom bar
        _sep(self).pack(fill="x", padx=20, pady=8)
        bot = _fr(self, bg=BG)
        bot.pack(fill="x", padx=20, pady=(0,12))
        _btn(bot, "Run Prediction  →", self._submit,
             bg=GREEN, fg=BG, size=11, width=18).pack(side="right")

        self._populate()

    def _populate(self):
        for w in self._inner.winfo_children(): w.destroy()
        self._vars.clear()
        data = self.master.data
        for col in data.feature_cols:
            is_ctrl = col in data.controllable_cols
            row = _fr(self._inner, bg=CARD)
            row.pack(fill="x", padx=8, pady=3)
            tag_c = GREEN if is_ctrl else SUBTEXT
            tag_t = "controllable" if is_ctrl else "read-only"
            _lbl(row, col, size=10, bold=True, color=TEXT).grid(
                row=0, column=0, sticky="w", padx=(6,0))
            _lbl(row, f"[{tag_t}]", size=8, color=tag_c).grid(
                row=0, column=1, sticky="w", padx=(6,0))
            v = tk.StringVar(value="0.0")
            self._vars[col] = v
            _entry(row, v, width=14).grid(row=1, column=0, columnspan=2,
                                           sticky="w", padx=6, pady=(2,4))

    def _random(self):
        data = self.master.data
        idx  = np.random.randint(0, len(data.X_test_raw))
        for col, val in zip(data.feature_cols, data.X_test_raw[idx]):
            if col in self._vars:
                self._vars[col].set(f"{val:.4f}")

# --- PLACEHOLDER_INPUT3 ---

    def _submit(self):
        data = self.master.data
        vals = []
        for col in data.feature_cols:
            try:
                vals.append(float(self._vars[col].get().strip()))
            except ValueError:
                messagebox.showerror("Input Error",
                    f"Invalid value for '{col}': '{self._vars[col].get()}'")
                return
        thr = {}
        for key, var in self._thr.items():
            try:
                thr[key] = float(var.get().strip())
            except ValueError:
                messagebox.showerror("Threshold Error",
                    f"Invalid threshold '{key}': '{var.get()}'")
                return
        self.master.current_inputs = np.array(vals, dtype=np.float32)
        self.master.thresholds     = thr
        self.master.show("summary")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Summary
# ═══════════════════════════════════════════════════════════════════════════════
class SummaryPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master = master

    def refresh(self):
        for w in self.winfo_children(): w.destroy()
        app = self.master
        pmlr = app.supervised.predict_state(app.current_inputs)
        xs   = app.supervised.scaler.transform(app.current_inputs.reshape(1,-1))[0]
        rnn  = app.rnn.predict(xs)
        t    = app.thresholds
        app.last_pmlr = pmlr; app.last_rnn = rnn

        # violations
        viols = []
        if pmlr["co2_emission_30d"]          > t["max_co2"]:  viols.append(f"CO₂ {pmlr['co2_emission_30d']:.2f} > {t['max_co2']}")
        if pmlr["cost_energy_bill_30d"]       > t["max_cost"]: viols.append(f"Cost {pmlr['cost_energy_bill_30d']:.2f} > {t['max_cost']}")
        if pmlr["current_energy_efficiency"]  < t["min_eff"]:  viols.append(f"Eff {pmlr['current_energy_efficiency']:.4f} < {t['min_eff']}")
        app.last_viols = viols

        # header
        top = _fr(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(14,0))
        _lbl(top, "Prediction Summary", size=15, bold=True, color=WHITE).pack(side="left")
        _btn(top, "← Back", lambda: app.show("inputs"), bg=SURFACE, width=8).pack(side="right")
        _sep(self).pack(fill="x", padx=20, pady=8)

# --- PLACEHOLDER_SUMMARY2 ---

        body = _fr(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        body.columnconfigure(0, weight=3); body.columnconfigure(1, weight=2)

        # ── left: predictions table ───────────────────────────────────────────
        lc = _fr(body, bg=CARD)
        lc.grid(row=0, column=0, sticky="nsew", padx=(0,10), pady=4)
        _lbl(lc, "Model Predictions", size=11, bold=True,
             color=ACCENT).pack(anchor="w", padx=12, pady=(10,4))
        _sep(lc).pack(fill="x", padx=12)

        # header row
        hr = _fr(lc, bg=SURFACE)
        hr.pack(fill="x", padx=12, pady=(4,0))
        for txt, w in [("Metric",24),("MLR",10),("RNN",10),("Status",14)]:
            _lbl(hr, txt, size=9, bold=True, color=SUBTEXT,
                 width=w, anchor="w").pack(side="left", padx=4, pady=4)

        rows = [
            ("CO₂ Emission (30d)",  pmlr["co2_emission_30d"],         rnn[0], "max", t["max_co2"]),
            ("Energy Cost (30d)",   pmlr["cost_energy_bill_30d"],      rnn[1], "max", t["max_cost"]),
            ("Energy Efficiency",   pmlr["current_energy_efficiency"], rnn[2], "min", t["min_eff"]),
        ]
        for metric, mv, rv, mode, thr_v in rows:
            viol = mv > thr_v if mode=="max" else mv < thr_v
            sc, st = (RED,"⚠ VIOLATION") if viol else (GREEN,"✓ OK")
            r = _fr(lc, bg=CARD); r.pack(fill="x", padx=12, pady=2)
            _lbl(r, metric,       size=10, color=TEXT,   width=24, anchor="w").pack(side="left", padx=4)
            _lbl(r, f"{mv:.3f}",  size=10, color=TEXT,   width=10, anchor="e").pack(side="left", padx=4)
            _lbl(r, f"{rv:.3f}",  size=10, color=ACCENT, width=10, anchor="e").pack(side="left", padx=4)
            _lbl(r, st, size=10, bold=True, color=sc, width=14, anchor="center").pack(side="left", padx=4)

        # fault row
        fault = pmlr["fault_detected"]; fprob = pmlr["fault_probability"]
        fr2 = _fr(lc, bg=CARD); fr2.pack(fill="x", padx=12, pady=2)
        fc2, ft = (RED,"FAULT") if fault else (GREEN,"Normal")
        _lbl(fr2,"Fault Detection",size=10,color=TEXT,width=24,anchor="w").pack(side="left",padx=4)
        _lbl(fr2,f"{fault} ({fprob:.1%})",size=10,color=TEXT,width=10,anchor="e").pack(side="left",padx=4)
        _lbl(fr2,"—",size=10,color=SUBTEXT,width=10,anchor="e").pack(side="left",padx=4)
        _lbl(fr2,ft,size=10,bold=True,color=fc2,width=14,anchor="center").pack(side="left",padx=4)

# --- PLACEHOLDER_SUMMARY3 ---

        # ── right: ESG status ─────────────────────────────────────────────────
        rc = _fr(body, bg=CARD)
        rc.grid(row=0, column=1, sticky="nsew", padx=(10,0), pady=4)
        sh, sc2 = ("⚠  ESG VIOLATION", RED) if viols else ("✓  ESG COMPLIANT", GREEN)
        _lbl(rc, sh, size=13, bold=True, color=sc2).pack(pady=(18,4))
        _lbl(rc, f"{len(viols)} threshold(s) exceeded" if viols else "All thresholds within limits",
             size=10, color=SUBTEXT).pack()
        _sep(rc).pack(fill="x", padx=12, pady=10)
        _lbl(rc, "Active Thresholds", size=10, bold=True, color=TEXT).pack(anchor="w", padx=12)
        for k, v in [("Max CO₂", t["max_co2"]),("Max Cost", t["max_cost"]),("Min Eff", t["min_eff"])]:
            tr = _fr(rc, bg=CARD); tr.pack(fill="x", padx=12, pady=2)
            _lbl(tr, k, size=10, color=SUBTEXT, width=12, anchor="w").pack(side="left")
            _lbl(tr, str(v), size=10, bold=True, color=TEXT).pack(side="left")
        if viols:
            _sep(rc).pack(fill="x", padx=12, pady=10)
            _lbl(rc, "Violations", size=10, bold=True, color=RED).pack(anchor="w", padx=12)
            for v in viols:
                _lbl(rc, f"• {v}", size=9, color=RED).pack(anchor="w", padx=20, pady=1)

        # bottom
        _sep(self).pack(fill="x", padx=20, pady=8)
        bot = _fr(self, bg=BG); bot.pack(fill="x", padx=20, pady=(0,12))
        _btn(bot, "← New Input", lambda: app.show("inputs"),
             bg=SURFACE, width=12).pack(side="left")
        if viols:
            _btn(bot, "Run RL Optimizer  →", lambda: app.show("optimize"),
                 bg=YELLOW, fg=BG, size=11, width=18).pack(side="right")
        else:
            _lbl(bot, "System is ESG compliant — no optimization needed.",
                 size=10, color=GREEN).pack(side="right")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Optimize
# ═══════════════════════════════════════════════════════════════════════════════
class OptimizePage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master = master
        self._chk: dict[str, tk.BooleanVar] = {}
        self._lo:  dict[str, tk.StringVar]  = {}
        self._hi:  dict[str, tk.StringVar]  = {}

    def refresh(self):
        for w in self.winfo_children(): w.destroy()
        self._chk.clear(); self._lo.clear(); self._hi.clear()
        self._build()

    def _build(self):
        app = self.master
        top = _fr(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(14,0))
        _lbl(top, "RL Optimization Engine", size=15, bold=True, color=WHITE).pack(side="left")
        _btn(top, "← Summary", lambda: app.show("summary"), bg=SURFACE, width=10).pack(side="right")
        _sep(self).pack(fill="x", padx=20, pady=8)

# --- PLACEHOLDER_OPTIMIZE2 ---

        body = _fr(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ── left: variable checkboxes ─────────────────────────────────────────
        lc = _fr(body, bg=CARD)
        lc.grid(row=0, column=0, sticky="nsew", padx=(0,10), pady=4)
        _lbl(lc, "Variables to Adjust", size=11, bold=True,
             color=YELLOW).pack(anchor="w", padx=12, pady=(10,2))
        _lbl(lc, "Uncheck to lock. Set Min/Max to constrain the RL range.",
             size=9, color=SUBTEXT).pack(anchor="w", padx=12, pady=(0,4))
        _sep(lc).pack(fill="x", padx=12)

        # header row for the bounds columns
        hdr = _fr(lc, bg=CARD); hdr.pack(fill="x", padx=8, pady=(2,0))
        _lbl(hdr, "Variable", size=9, bold=True, color=SUBTEXT).pack(side="left", padx=(24,0))
        _lbl(hdr, "Min", size=9, bold=True, color=SUBTEXT).pack(side="right", padx=(0,4))
        _lbl(hdr, "Max", size=9, bold=True, color=SUBTEXT).pack(side="right", padx=(0,4))

        _, chk_inner = _scrollable(lc)
        for col in app.data.controllable_cols:
            v = tk.BooleanVar(value=True)
            self._chk[col] = v
            lo_v = tk.StringVar(value="0")
            hi_v = tk.StringVar(value="")
            self._lo[col] = lo_v; self._hi[col] = hi_v
            row = _fr(chk_inner, bg=CARD); row.pack(fill="x", padx=8, pady=3)
            tk.Checkbutton(row, variable=v, bg=CARD, fg=TEXT,
                           selectcolor=SURFACE, activebackground=CARD,
                           activeforeground=TEXT, relief="flat",
                           font=("Helvetica" if IS_MAC else "Segoe UI", 10)
                           ).pack(side="left")
            _lbl(row, col, size=10, color=TEXT).pack(side="left", padx=4)
            # Max entry (right-most)
            tk.Entry(row, textvariable=hi_v, width=6, bg=SURFACE, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     font=("Helvetica" if IS_MAC else "Segoe UI", 9)
                     ).pack(side="right", padx=(0,4))
            # Min entry
            tk.Entry(row, textvariable=lo_v, width=6, bg=SURFACE, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     font=("Helvetica" if IS_MAC else "Segoe UI", 9)
                     ).pack(side="right", padx=(0,2))

        br = _fr(lc, bg=CARD); br.pack(fill="x", padx=12, pady=6)
        _btn(br,"All",  lambda:[v.set(True)  for v in self._chk.values()], bg=SURFACE, width=5).pack(side="left",padx=(0,4))
        _btn(br,"None", lambda:[v.set(False) for v in self._chk.values()], bg=SURFACE, width=5).pack(side="left")

        _sep(lc).pack(fill="x", padx=12, pady=4)
        self._run_btn = _btn(lc, "▶  Run Optimizer", self._run,
                             bg=YELLOW, fg=BG, size=11, width=16)
        self._run_btn.pack(pady=10)
        self._status = _lbl(lc, "", size=9, color=SUBTEXT)
        self._status.pack(pady=(0,10))

        # ── right: results ────────────────────────────────────────────────────
        self._res = _fr(body, bg=CARD)
        self._res.grid(row=0, column=1, sticky="nsew", padx=(10,0), pady=4)
        _lbl(self._res, "Results will appear here after optimization.",
             size=11, color=SUBTEXT).pack(expand=True)

        _sep(self).pack(fill="x", padx=20, pady=8)
        bot = _fr(self, bg=BG); bot.pack(fill="x", padx=20, pady=(0,12))
        _btn(bot, "← New Input", lambda: app.show("inputs"),
             bg=SURFACE, width=12).pack(side="left")

    def _run(self):
        app = self.master
        selected = [c for c,v in self._chk.items() if v.get()]
        if not selected:
            messagebox.showwarning("No Variables","Select at least one variable."); return
        active = [app.data.feature_cols.index(c) for c in selected if c in app.data.feature_cols]

        # build var_bounds: {feature_index: (lo, hi)}
        var_bounds: dict[int, tuple[float,float]] = {}
        for col in selected:
            if col not in app.data.feature_cols: continue
            idx = app.data.feature_cols.index(col)
            lo_s = self._lo.get(col, tk.StringVar(value="0")).get().strip()
            hi_s = self._hi.get(col, tk.StringVar(value="")).get().strip()
            try:    lo = float(lo_s) if lo_s != "" else 0.
            except: lo = 0.
            try:    hi = float(hi_s) if hi_s != "" else np.inf
            except: hi = np.inf
            if lo > hi: lo, hi = hi, lo  # swap if user entered them backwards
            var_bounds[idx] = (lo, hi)

        self._run_btn.config(state="disabled", text="Running...")
        self._status.config(text="Computing…"); self.update()
        q: queue.Queue = queue.Queue()

# --- PLACEHOLDER_OPTIMIZE3 ---

        def worker():
            try:
                rl = RLOptimizer(app.supervised, app.data, app.thresholds,
                                 active_indices=active, var_bounds=var_bounds)
                try:
                    rl.load()
                    if rl.agent.action_space.shape[0] != len(active):
                        raise ValueError("action space mismatch")
                except Exception:
                    q.put(("s","Retraining RL for selected variables…"))
                    rl.train(); rl.save()
                opt = rl.suggest(app.current_inputs)
                q.put(("done", opt, selected))
            except Exception:
                import traceback; q.put(("err", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_rl(q)

    def _poll_rl(self, q):
        try:
            while True:
                m = q.get_nowait()
                if   m[0]=="s":    self._status.config(text=m[1])
                elif m[0]=="done":
                    self._run_btn.config(state="normal", text="▶  Run Optimizer")
                    self._status.config(text="Done.")
                    self._show_results(m[1], m[2]); return
                elif m[0]=="err":
                    self._run_btn.config(state="normal", text="▶  Run Optimizer")
                    messagebox.showerror("Optimizer Error", m[1]); return
        except queue.Empty:
            pass
        self.after(150, lambda: self._poll_rl(q))

    def _show_results(self, opt: np.ndarray, selected: list[str]):
        for w in self._res.winfo_children(): w.destroy()
        app = self.master
        pb = app.last_pmlr
        pa = app.supervised.predict_state(opt)
        xs = app.supervised.scaler.transform(opt.reshape(1,-1))[0]
        ra = app.rnn.predict(xs)
        t  = app.thresholds

        _lbl(self._res, "Optimization Results", size=11, bold=True,
             color=GREEN).pack(anchor="w", padx=12, pady=(10,4))
        _sep(self._res).pack(fill="x", padx=12)

        # adjustments
        _lbl(self._res, "Suggested Adjustments", size=10, bold=True,
             color=YELLOW).pack(anchor="w", padx=12, pady=(8,4))
        hr = _fr(self._res, bg=SURFACE); hr.pack(fill="x", padx=12)
        for txt,w in [("Variable",22),("Before",10),("After",10),("Δ",10)]:
            _lbl(hr,txt,size=9,bold=True,color=SUBTEXT,width=w,anchor="w").pack(side="left",padx=4,pady=3)

# --- PLACEHOLDER_OPTIMIZE4 ---

        for col, orig, o in zip(app.data.feature_cols, app.current_inputs, opt):
            if col not in selected: continue
            d = o - orig; dc = GREEN if d>=0 else RED; ds = f"+{d:.4f}" if d>=0 else f"{d:.4f}"
            r = _fr(self._res, bg=CARD); r.pack(fill="x", padx=12, pady=1)
            _lbl(r,col,       size=10,color=TEXT,   width=22,anchor="w").pack(side="left",padx=4)
            _lbl(r,f"{orig:.4f}",size=10,color=SUBTEXT,width=10,anchor="e").pack(side="left",padx=4)
            _lbl(r,f"{o:.4f}", size=10,color=WHITE, width=10,anchor="e").pack(side="left",padx=4)
            _lbl(r,ds,         size=10,color=dc,    width=10,anchor="e").pack(side="left",padx=4)

        _sep(self._res).pack(fill="x", padx=12, pady=8)
        _lbl(self._res,"Projected Outcome",size=10,bold=True,color=ACCENT).pack(anchor="w",padx=12,pady=(0,4))
        hr2 = _fr(self._res, bg=SURFACE); hr2.pack(fill="x", padx=12)
        for txt,w in [("Metric",22),("Before",10),("After MLR",10),("After RNN",10),("Status",12)]:
            _lbl(hr2,txt,size=9,bold=True,color=SUBTEXT,width=w,anchor="w").pack(side="left",padx=4,pady=3)

        metrics = [
            ("CO₂ Emission (30d)",  "co2_emission_30d",         ra[0], "max", t["max_co2"]),
            ("Energy Cost (30d)",   "cost_energy_bill_30d",      ra[1], "max", t["max_cost"]),
            ("Energy Efficiency",   "current_energy_efficiency", ra[2], "min", t["min_eff"]),
        ]
        new_viols = []
        for lbl_t, key, rv, mode, thr_v in metrics:
            bv = pb[key]; av = pa[key]
            # If MLR value is negative (physically invalid), fall back to RNN
            if av < 0:
                check_v = rv if rv >= 0 else None
            else:
                check_v = av
            if check_v is None:
                viol = True  # both negative → treat as violation / unknown
                sc, st = (YELLOW, "? INVALID")
            else:
                viol = check_v > thr_v if mode=="max" else check_v < thr_v
                sc, st = (RED,"⚠ VIOLATION") if viol else (GREEN,"✓ OK")
            if viol: new_viols.append(lbl_t)
            r = _fr(self._res, bg=CARD); r.pack(fill="x", padx=12, pady=1)
            _lbl(r,lbl_t,      size=10,color=TEXT,   width=22,anchor="w").pack(side="left",padx=4)
            _lbl(r,f"{bv:.4f}",size=10,color=SUBTEXT,width=10,anchor="e").pack(side="left",padx=4)
            _lbl(r,f"{av:.4f}",size=10,color=WHITE,  width=10,anchor="e").pack(side="left",padx=4)
            _lbl(r,f"{rv:.4f}",size=10,color=ACCENT, width=10,anchor="e").pack(side="left",padx=4)
            _lbl(r,st,size=10,bold=True,color=sc,width=12,anchor="center").pack(side="left",padx=4)

        _sep(self._res).pack(fill="x", padx=12, pady=8)
        if not new_viols:
            _lbl(self._res,"✓  Post-optimization state is ESG compliant.",
                 size=11,bold=True,color=GREEN).pack(pady=8)
        else:
            _lbl(self._res,f"⚠  {', '.join(new_viols)} still violated. Try increasing RL_TIMESTEPS.",
                 size=10,bold=True,color=YELLOW).pack(pady=8)


# ═══════════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Energy Management System  v3")
        self.geometry("1100x720"); self.minsize(900,580)
        self.configure(bg=BG)

        # shared state
        self.data: DataLoader | None = None
        self.supervised: SupervisedEngine | None = None
        self.rnn: RNNEngine | None = None
        self.current_inputs: np.ndarray | None = None
        self.thresholds: dict = {"max_co2":400.,"max_cost":1000.,"min_eff":0.5}
        self.last_pmlr = None; self.last_rnn = None; self.last_viols = []

# --- PLACEHOLDER_APP2 ---

        # nav bar
        nav = _fr(self, bg=SURFACE)
        nav.pack(fill="x", side="top")
        _lbl(nav,"⚡ EMS",size=12,bold=True,color=YELLOW).pack(side="left",padx=16,pady=8)
        self._nav_btns: dict[str,tk.Button] = {}
        for name, title in [("inputs","Inputs"),("summary","Summary"),("optimize","Optimize")]:
            b = tk.Button(nav, text=title, bg=SURFACE, fg=BORDER,
                          relief="flat", font=("Helvetica" if IS_MAC else "Segoe UI",10),
                          padx=14, pady=8, cursor="hand2",
                          activebackground=CARD, activeforeground=TEXT,
                          state="disabled",
                          command=lambda n=name: self.show(n))
            b.pack(side="left")
            self._nav_btns[name] = b
        _lbl(nav,"v5  |  backups: v1_backup.py  v2_backup.py",
             size=8,color=BORDER).pack(side="right",padx=16)

        # page container
        cont = _fr(self, bg=BG)
        cont.pack(fill="both", expand=True)
        cont.rowconfigure(0,weight=1); cont.columnconfigure(0,weight=1)

        self._pages: dict[str,tk.Frame] = {}
        for name, cls in [("startup",StartupPage),("inputs",InputPage),
                           ("summary",SummaryPage),("optimize",OptimizePage)]:
            p = cls(self); p.grid(row=0,column=0,sticky="nsew", in_ = cont)
            self._pages[name] = p

        self.show("startup")
        self._pages["startup"].start()

    def show(self, name:str):
        if name in ("inputs","summary","optimize") and self.data is None: return
        p = self._pages[name]
        if name=="inputs" and not self._pages["inputs"]._built:
            p.build()
        elif name=="summary" and self.current_inputs is not None:
            p.refresh()
        elif name=="optimize" and self.current_inputs is not None:
            p.refresh()
        p.tkraise()
        for n,b in self._nav_btns.items():
            b.config(fg=WHITE if n==name else SUBTEXT,
                     bg=CARD  if n==name else SURFACE,
                     state="normal")

    def on_ready(self, data, sup, rnn, rl):
        self.data = data
        self.supervised = sup
        self.rnn = rnn
        self.rl  = rl
        # Enable nav buttons visually now that models are loaded
        for b in self._nav_btns.values():
            b.config(state="normal")
        # Navigate to inputs page
        self.after(100, lambda: self.show("inputs"))


if __name__ == "__main__":
    try:
        App().mainloop()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
