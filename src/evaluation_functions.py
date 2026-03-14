#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from utils.recommendation_functions import (
    recommend_by_popularity,
    recommend_by_content_user,
    recommend_by_collaborative,
    recommend_by_hybrid,
)

####################################
### 1. Evaluación Modelo Popularidad
####################################

def evaluate_popularity_LMO_at_k(
    k,
    games,
    user_train,
    user_test,
    results_dir="res",
    tag=None,
):
    """
    k. Longitud de la lista a recomendar
    games. Catálogo de juegos. Necesario appid, name, positive_reviews.
    user_train, user_test. df con user_id, appid.
    results_dir. Directorio donde se guardarán los resultados.
    tag. Sufijo para el fichero de salida en caso de versiones.
    """
    precisions, recalls, f1s, ndcgs = [], [], [], []
    hit_flags = []          # Para HitRate@k
    all_rec_items = set()   # Para Item Coverage@k

    # Asegurar tipos
    user_train["user_id"] = user_train["user_id"].astype(int)
    user_test["user_id"] = user_test["user_id"].astype(int)

    # Recorrer todos los usuarios del test
    for user in user_test["user_id"].unique():
        test_items = user_test.loc[user_test["user_id"] == user, "appid"].tolist()
        if not test_items:
            continue
        M = len(test_items)

        # Necesitamos que el usuario tenga historial en train
        train_games = user_train[user_train["user_id"] == user]["appid"].tolist()
        if not train_games:
            continue

        # Recomendación por popularidad
        recs = recommend_by_popularity(
            user_id=user,
            games=games,
            interactions_df=user_train,
            k=k,
        )
        if recs.empty:
            continue

        rec_ids = recs["appid"].tolist()
        rec_scores = recs["popularity_score"].tolist()

        # Flag de si el item recomendado pertenece al test del usuario
        y_true = [1 if g in test_items else 0 for g in rec_ids]
        tp = sum(y_true)

        # Métricas (nivel usuario)
        precision = tp / k
        recall = tp / M
        f1 = (2 * precision * recall) / (precision + recall) if tp else 0.0
        ndcg = ndcg_score([y_true], [rec_scores]) if tp else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ndcgs.append(ndcg)
        hit_flags.append(1 if tp > 0 else 0) # HitRate@k
        all_rec_items.update(rec_ids) # Item coverage

    num_users = len(precisions)
    hit_rate = float(np.mean(hit_flags)) if hit_flags else 0.0

    # Catálogo con todos los juegos
    n_items_catalog = games["appid"].nunique()
    item_coverage = (
        float(len(all_rec_items) / n_items_catalog) if n_items_catalog > 0 else 0.0
    )

    # Métricas finales
    metrics = {
        "k": int(k),
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "f1@k": float(np.mean(f1s)) if f1s else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate@k": hit_rate,
        "item_coverage@k": item_coverage,
        "num_users": int(num_users),
    }

    # Resultados
    suffix = f"_{tag}" if tag else ""
    filename = f"metrics_popularity_LMO_k{k}{suffix}.csv"
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, filename), index=False)

    return metrics


########################## 
### 2. Modelo de contenido
##########################
def evaluate_content_user_LMO_at_k(
    k,
    games,
    user_train,
    user_test,
    U_norm=None,
    user_to_pos=None,
    X_combined=None,
    idx=None,
    results_dir="res",
    tag=None,
):
    """
    k. Longitud de la lista a recomendar
    games. Catálogo de juegos. Necesario appid, name, positive_reviews.
    user_train, user_test. df con user_id, appid.
    U_norm. Matriz de perfiles de usuario normalizados.
    user_to_pos. Diccionario que mapea user_id - índice de fila en U_norm.
    X_combined. Matriz de características de los ítems.
    idx. Diccionario que mapea appid - índice de fila en X_combined.
    results_dir. Directorio donde se guardarán los resultados.
    tag. Sufijo para el fichero de salida en caso de versiones.
    """
    precisions, recalls, f1s, ndcgs = [], [], [], []
    hit_flags = []          # Para HitRate@k
    all_rec_items = set()   # Para Item Coverage@k

    # Asegurar tipos
    user_train["user_id"] = user_train["user_id"].astype(int)
    user_test["user_id"] = user_test["user_id"].astype(int)

    # Recorrer todos los usuarios del test
    for user in user_test["user_id"].unique():
        test_items = user_test.loc[user_test["user_id"] == user, "appid"].tolist()
        if not test_items:
            continue
        M = len(test_items)

        # Necesitamos que el usuario tenga historial en train
        train_games = user_train[user_train["user_id"] == user]["appid"].tolist()
        if not train_games:
            continue

        # Recomendación por contenido
        recs = recommend_by_content_user(
            user_id=user,
            user_train=user_train,
            U_norm=U_norm,
            user_to_pos=user_to_pos,
            X_combined=X_combined,
            games=games,
            idx=idx,
            top_n=k,
        )
        if recs.empty:
            continue

        rec_ids = recs["appid"].tolist()
        rec_scores = recs["score_content"].tolist()

        # Flag de si el item recomendado pertenece al test del usuario
        y_true = [1 if g in test_items else 0 for g in rec_ids]
        tp = sum(y_true)

        # Métricas (nivel usuario)
        precision = tp / k
        recall = tp / M
        f1 = (2 * precision * recall) / (precision + recall) if tp else 0.0
        ndcg = ndcg_score([y_true], [rec_scores]) if tp else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ndcgs.append(ndcg)
        hit_flags.append(1 if tp > 0 else 0) # HitRate@k
        all_rec_items.update(rec_ids)  # Item coverage

    num_users = len(precisions)
    hit_rate = float(np.mean(hit_flags)) if hit_flags else 0.0

    # Catálogo con todos los juegos
    n_items_catalog = games["appid"].nunique()
    item_coverage = (
        float(len(all_rec_items) / n_items_catalog) if n_items_catalog > 0 else 0.0
    )

    # Métricas finales
    metrics = {
        "k": int(k),
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "f1@k": float(np.mean(f1s)) if f1s else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate@k": hit_rate,
        "item_coverage@k": item_coverage,
        "num_users": int(num_users),
    }

    # Resultados
    suffix = f"_{tag}" if tag else ""
    filename = f"metrics_content_LMO_k{k}{suffix}.csv"
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, filename), index=False)

    return metrics


##########################
### 3. Modelo colaborativo
##########################

def evaluate_collaborative_LMO_at_k(
    k,
    user_train,
    user_test,
    knn,
    R_norm,
    R,
    user_map,
    inv_game_map,
    idx,
    results_dir="res",
    tag=None,
):
    """
    k. Longitud de la lista a recomendar
    user_train, user_test. df con user_id, appid.
    knn. Modelo de vecinos más cercanos ya entrenado.
    R_norm. Matriz usuario–ítem normalizada.
    R. Matriz usuario–ítem original (sin normalizar).
    user_map. Diccionario que mapea user_id - índice de fila en las matrices R_norm y R.
    inv_game_map. Diccionario que mapea col_index - appid.
    idx. Diccionario que mapea appid - índice de fila en X_combined.
    results_dir. Directorio donde se guardarán los resultados.
    tag. Sufijo para el fichero de salida en caso de versiones.
    """
    precisions, recalls, f1s, ndcgs = [], [], [], []
    hit_flags = []          # Para HitRate@k
    all_rec_items = set()   # Para Item Coverage@k

    # Asegurar tipos
    user_train["user_id"] = user_train["user_id"].astype(int)
    user_test["user_id"] = user_test["user_id"].astype(int)

    # Recorrer todos los usuarios del test
    for user in user_test["user_id"].unique():
        test_items = user_test.loc[user_test["user_id"] == user, "appid"].tolist()
        if not test_items:
            continue
        M = len(test_items)

        # Necesitamos que el usuario tenga historial en train
        train_games = user_train[user_train["user_id"] == user]["appid"].tolist()
        if not train_games:
            continue

        # Recomendación colaborativo
        recs = recommend_by_collaborative(
            user_id=user,
            knn=knn,
            R_norm=R_norm,
            R=R,
            user_map=user_map,
            inv_game_map=inv_game_map,
            idx=idx,
            top_n=k,
        )
        if recs.empty:
            continue

        rec_ids = recs["appid"].tolist()
        rec_scores = (
            recs["score"].tolist()
            if "score" in recs.columns
            else list(range(len(rec_ids), 0, -1))
        )

        # Flag de si el item recomendado pertenece al test del usuario
        y_true = [1 if g in test_items else 0 for g in rec_ids]
        tp = sum(y_true)

        # Métricas (nivel usuario)
        precision = tp / k
        recall = tp / M
        f1 = (2 * precision * recall) / (precision + recall) if tp else 0.0
        ndcg = ndcg_score([y_true], [rec_scores]) if tp else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ndcgs.append(ndcg)
        hit_flags.append(1 if tp > 0 else 0) # HitRate@k
        all_rec_items.update(rec_ids)  # Item coverage

    num_users = len(precisions)
    hit_rate = float(np.mean(hit_flags)) if hit_flags else 0.0

    # Catálogo con todos los juegos
    n_items_catalog = len(inv_game_map) if inv_game_map is not None else len(all_rec_items)
    item_coverage = (
        float(len(all_rec_items) / n_items_catalog) if n_items_catalog > 0 else 0.0
    )

    # Métricas finales
    metrics = {
        "k": int(k),
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "f1@k": float(np.mean(f1s)) if f1s else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate@k": hit_rate,
        "item_coverage@k": item_coverage,
        "num_users": int(num_users),
    }

    # Resultados
    suffix = f"_{tag}" if tag else ""
    filename = f"metrics_collaborative_k{k}{suffix}.csv"
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, filename), index=False)

    return metrics


#####################
### 4. Modelo híbrido
#####################

def evaluate_hybrid_LMO_at_k(
    k,
    alpha,
    user_train,
    user_test,
    U_norm,
    user_to_pos,
    X_combined,
    R_norm,
    R,
    user_map,
    game_map,
    inv_game_map,
    games,
    idx,
    knn,
    results_dir="res",
    tag=None,
    top_neighbors=50,
):
    """
    k. Longitud de la lista a recomendar.
    alpha: peso del contenido (α=1 → contenido puro, α=0 → colaborativo puro).
    user_train, user_test. df con user_id, appid.
    U_norm. Matriz de perfiles de usuario normalizados.
    user_to_pos. Diccionario que mapea user_id - índice de fila en U_norm.
    X_combined. Matriz de características de los ítems. 
    R_norm. Matriz usuario–ítem normalizada.
    R. Matriz usuario–ítem original (sin normalizar). 
    user_map. Diccionario que mapea user_id - índice de fila en las matrices R_norm y R.
    game_map. Diccionario que mapea appid - índice de columna en R_norm / R.
    inv_game_map. Diccionario que mapea col_index - appid.
    games. Catálogo de juegos. Necesario appid, name, positive_reviews.
    idx. Diccionario que mapea appid - índice de fila en X_combined.
    knn. Modelo de vecinos más cercanos ya entrenado. 
    results_dir. Directorio donde se guardarán los resultados.
    tag. Sufijo para el fichero de salida en caso de versiones.
    top_neighbors. Número de vecinos considerados en la parte colaborativa.    
    """
    precisions, recalls, f1s, ndcgs = [], [], [], []
    hit_flags = []          # Para HitRate@k
    all_rec_items = set()   # Para Item Coverage@k

    # Asegurar tipos
    user_train["user_id"] = user_train["user_id"].astype(int)
    user_test["user_id"] = user_test["user_id"].astype(int)

    # Recorrer todos los usuarios del test
    for user in user_test["user_id"].unique():
        test_items = user_test.loc[user_test["user_id"] == user, "appid"].tolist()
        if not test_items:
            continue
        M = len(test_items)

        # Necesitamos que el usuario tenga historial en train
        train_games = user_train[user_train["user_id"] == user]["appid"].tolist()
        if not train_games:
            continue

        # Recomendación por híbrido
        recs = recommend_by_hybrid(
            user_id=user,
            alpha=alpha,
            user_train=user_train,
            U_norm=U_norm,
            user_to_pos=user_to_pos,
            X_combined=X_combined,
            R_norm=R_norm,
            R=R,
            user_map=user_map,
            game_map=game_map,
            inv_game_map=inv_game_map,
            games=games,
            idx=idx,
            knn=knn,
            top_n=k,
            top_neighbors=top_neighbors,
        )
        if recs.empty:
            continue

        rec_ids = recs["appid"].tolist()
        rec_scores = recs["score_hybrid"].tolist()

        # Flag de si el item recomendado pertenece al test del usuario
        y_true = [1 if g in test_items else 0 for g in rec_ids]
        tp = sum(y_true)

        # Métricas (nivel usuario)
        precision = tp / k
        recall = tp / M
        f1 = (2 * precision * recall) / (precision + recall) if tp else 0.0
        ndcg = ndcg_score([y_true], [rec_scores]) if tp else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ndcgs.append(ndcg)
        hit_flags.append(1 if tp > 0 else 0) # HitRate@k
        all_rec_items.update(rec_ids)  # Item coverage

    num_users = len(precisions)
    hit_rate = float(np.mean(hit_flags)) if hit_flags else 0.0

    # Catálogo con todos los juegos
    n_items_catalog = games["appid"].nunique()
    item_coverage = (
        float(len(all_rec_items) / n_items_catalog) if n_items_catalog > 0 else 0.0
    )

    # Métricas finales
    metrics = {
        "k": int(k),
        "alpha": float(alpha),
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "f1@k": float(np.mean(f1s)) if f1s else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate@k": hit_rate,
        "item_coverage@k": item_coverage,
        "num_users": int(num_users),
    }

    # Resultados
    suffix = f"_a{alpha}_{tag}" if tag else f"_a{alpha}"
    filename = f"metrics_hybrid_k{k}{suffix}.csv"
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, filename), index=False)

    return metrics
