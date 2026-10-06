"""Leakage-aware hyperparameter search for RazorRisk synthetic models.

Run from the repository root:
    python ml/hyperparameter_search.py

Selection metric: mean CV PR-AUC. The final held-out test split is never used
for hyperparameter selection. User-level folds keep every user's transactions
in one fold. The GNN uses the existing user graph and fold-specific label masks;
labels never enter graph construction.

The script also generates OOF predictions with the selected base-model
configurations and tunes the balanced stacker on those OOF predictions.
"""
from __future__ import annotations
import itertools, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ml.train_tabular_model import load_transactions, add_merchant_target_encoding, FEATURES, TabularModel
from ml.train_gnn import RiskGNN
from ml.risk_graph import build_user_graph, detect_communities, fetch_node_features, build_adjacency
from db.database import get_raw_sqlite_connection, read_sql_query
from ml.common import RNG_SEED

OUT = ROOT / "ml" / "models" / "hyperparameters.json"
N_SPLITS = 3

XGB_GRID = [
    dict(n_estimators=250, max_depth=3, learning_rate=0.04, min_child_weight=2, gamma=0.0, reg_lambda=2.0, subsample=0.9, colsample_bytree=0.9),
    dict(n_estimators=350, max_depth=4, learning_rate=0.035, min_child_weight=3, gamma=0.05, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.9),
    dict(n_estimators=450, max_depth=4, learning_rate=0.025, min_child_weight=4, gamma=0.05, reg_lambda=3.0, subsample=0.9, colsample_bytree=0.85),
    dict(n_estimators=300, max_depth=5, learning_rate=0.03, min_child_weight=4, gamma=0.1, reg_lambda=3.0, subsample=0.85, colsample_bytree=0.85),
]
GNN_GRID = [
    dict(hidden_dim_1=8, hidden_dim_2=4, learning_rate=0.03, epochs=250),
    dict(hidden_dim_1=16, hidden_dim_2=8, learning_rate=0.03, epochs=350),
    dict(hidden_dim_1=16, hidden_dim_2=8, learning_rate=0.05, epochs=400),
    dict(hidden_dim_1=24, hidden_dim_2=12, learning_rate=0.03, epochs=350),
]
STACKER_GRID = [
    dict(C=0.01, class_weight="balanced"), dict(C=0.05, class_weight="balanced"),
    dict(C=0.1, class_weight="balanced"), dict(C=0.5, class_weight="balanced"),
    dict(C=1.0, class_weight="balanced"),
]


def user_folds(user_ids, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG_SEED)
    return list(skf.split(np.asarray(user_ids), np.asarray(y)))


def xgb_eval(df, user_ids, y_user):
    folds = user_folds(user_ids, y_user)
    results=[]
    for p in XGB_GRID:
        aps=[]; aucs=[]
        for train_idx, val_idx in folds:
            train_users=set(np.asarray(user_ids)[train_idx]); val_users=set(np.asarray(user_ids)[val_idx])
            train_mask=df.user_id.isin(train_users).to_numpy(); val_mask=df.user_id.isin(val_users).to_numpy()
            d, global_rate, rates = add_merchant_target_encoding(df.copy(), train_mask)
            model=TabularModel().fit(d.loc[train_mask, FEATURES], d.loc[train_mask,'is_fraud'],
                                      (d.loc[train_mask,'is_fraud'].eq(0).sum()/max(d.loc[train_mask,'is_fraud'].eq(1).sum(),1)),
                                      hyperparams=p)
            s=model.predict_proba(d.loc[val_mask, FEATURES]); y=d.loc[val_mask,'is_fraud'].to_numpy()
            aps.append(average_precision_score(y,s)); aucs.append(roc_auc_score(y,s))
        results.append((float(np.mean(aps)), float(np.mean(aucs)), p))
        print(f"XGB {p} -> PR-AUC={np.mean(aps):.6f} ROC-AUC={np.mean(aucs):.6f}")
    return max(results, key=lambda x:x[0])


def gnn_fit_predict(X, A, train_mask, y, p):
    mu=X[train_mask].mean(0); sigma=X[train_mask].std(0); sigma[sigma==0]=1
    Xn=(X-mu)/sigma
    npos=int(y[train_mask].sum()); nneg=int(train_mask.sum()-npos)
    pos_weight=nneg/max(npos,1)
    model=RiskGNN(in_dim=X.shape[1], hidden_dim_1=p['hidden_dim_1'], hidden_dim_2=p['hidden_dim_2'], seed=RNG_SEED)
    for _ in range(p['epochs']):
        model.forward(Xn,A); model.backward(y,train_mask,pos_weight,A,p['learning_rate'])
    return model.forward(Xn,A)


def gnn_eval(conn, user_ids, X, A, y_user):
    folds=user_folds(user_ids,y_user); results=[]
    for p in GNN_GRID:
        aps=[]; aucs=[]
        for train_idx,val_idx in folds:
            train_mask=np.zeros(len(user_ids),dtype=bool); train_mask[train_idx]=True
            scores=gnn_fit_predict(X,A,train_mask,y_user,p)
            y=y_user[val_idx]; s=scores[val_idx]
            aps.append(average_precision_score(y,s)); aucs.append(roc_auc_score(y,s))
        results.append((float(np.mean(aps)),float(np.mean(aucs)),p))
        print(f"GNN {p} -> PR-AUC={np.mean(aps):.6f} ROC-AUC={np.mean(aucs):.6f}")
    return max(results,key=lambda x:x[0])


def oof_xgb(df,user_ids,y_user,p):
    oof=np.zeros(len(df)); folds=user_folds(user_ids,y_user)
    for train_idx,val_idx in folds:
        tr_users=set(np.asarray(user_ids)[train_idx]); va_users=set(np.asarray(user_ids)[val_idx])
        tr=df.user_id.isin(tr_users).to_numpy(); va=df.user_id.isin(va_users).to_numpy()
        d,_,_=add_merchant_target_encoding(df.copy(),tr)
        m=TabularModel().fit(d.loc[tr,FEATURES],d.loc[tr,'is_fraud'],d.loc[tr,'is_fraud'].eq(0).sum()/max(d.loc[tr,'is_fraud'].eq(1).sum(),1),hyperparams=p)
        oof[va]=m.predict_proba(d.loc[va,FEATURES])
    return oof


def oof_gnn(user_ids,X,A,y_user,p,df):
    scores_user=np.zeros(len(user_ids)); seen=np.zeros(len(user_ids),dtype=bool)
    for train_idx,val_idx in user_folds(user_ids,y_user):
        train_mask=np.zeros(len(user_ids),dtype=bool); train_mask[train_idx]=True
        scores=gnn_fit_predict(X,A,train_mask,y_user,p)
        scores_user[val_idx]=scores[val_idx]; seen[val_idx]=True
    mapping={u:s for u,s in zip(user_ids,scores_user)}
    return df.user_id.map(mapping).to_numpy()


def stacker_eval(xgb_oof,gnn_oof,df):
    # OOF shared signals: derive from each user's complete graph context only.
    conn=get_raw_sqlite_connection(); G=build_user_graph(conn)
    device_counts={}
    ip_counts={}
    cur=conn.cursor()
    for col,dct in [('device_id',device_counts),('ip_address',ip_counts)]:
        rows=cur.execute(f'SELECT {col}, COUNT(DISTINCT user_id) FROM transactions GROUP BY {col}').fetchall()
        dct.update({k:int(v) for k,v in rows})
    conn.close()
    shared_dev=df.device_id.map(lambda x:min(max(device_counts.get(x,1)-1,0),10)/10).to_numpy()
    shared_ip=df.ip_address.map(lambda x:min(max(ip_counts.get(x,1)-1,0),10)/10).to_numpy()
    X=np.column_stack([xgb_oof,gnn_oof,shared_dev,shared_ip]); y=df.is_fraud.to_numpy()
    results=[]
    # OOF predictions are already out-of-fold; evaluate stacker CV using splits of OOF rows.
    skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RNG_SEED)
    for p in STACKER_GRID:
        aps=[]; aucs=[]
        for tr,va in skf.split(X,y):
            m=LogisticRegression(C=p['C'],class_weight=p['class_weight'],max_iter=1000,random_state=RNG_SEED)
            m.fit(X[tr],y[tr]); s=m.predict_proba(X[va])[:,1]
            aps.append(average_precision_score(y[va],s)); aucs.append(roc_auc_score(y[va],s))
        results.append((float(np.mean(aps)),float(np.mean(aucs)),p))
        print(f"STACKER {p} -> PR-AUC={np.mean(aps):.6f} ROC-AUC={np.mean(aucs):.6f}")
    return max(results,key=lambda x:x[0])


def main():
    conn=get_raw_sqlite_connection(); df=load_transactions(conn)
    identity = read_sql_query('SELECT transaction_id, device_id, ip_address FROM transactions')
    df = df.merge(identity, on='transaction_id', how='left')
    cur=conn.cursor(); user_ids=[r[0] for r in cur.execute('SELECT user_id FROM users ORDER BY user_id').fetchall()]
    fraud_users={r[0] for r in cur.execute('SELECT DISTINCT user_id FROM transactions WHERE is_fraud_ground_truth=1').fetchall()}
    y_user=np.array([1 if u in fraud_users else 0 for u in user_ids])
    print(f"Dataset: {len(df)} transactions, {len(user_ids)} users, {int(y_user.sum())} fraud users")
    bx=xgb_eval(df,user_ids,y_user); bg=gnn_eval(conn,user_ids,*fetch_node_features(conn,build_user_graph(conn),detect_communities(build_user_graph(conn))[1]),y_user) if False else None
    G=build_user_graph(conn); _,comm=detect_communities(G); guids,X=fetch_node_features(conn,G,comm); A=build_adjacency(G,guids)
    bg=gnn_eval(conn,guids,X,A,y_user)
    print('\nGenerating OOF predictions for stacker tuning...')
    xgb_oof=oof_xgb(df,user_ids,y_user,bx[2]); gnn_oof=oof_gnn(guids,X,A,y_user,bg[2],df)
    bs=stacker_eval(xgb_oof,gnn_oof,df)
    result={'selection_metric':'mean_cv_pr_auc','cv_folds':N_SPLITS,'xgboost':{'best_params':bx[2],'mean_cv_pr_auc':bx[0],'mean_cv_roc_auc':bx[1]},'gnn':{'best_params':bg[2],'mean_cv_pr_auc':bg[0],'mean_cv_roc_auc':bg[1]},'stacker':{'best_params':bs[2],'mean_cv_pr_auc':bs[0],'mean_cv_roc_auc':bs[1]},'note':'Final test set is excluded from all hyperparameter selection. Stacker tuning uses OOF base predictions.'}
    OUT.write_text(json.dumps(result,indent=2)); print('\nBEST CONFIGURATION'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
