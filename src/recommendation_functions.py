import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse
from sklearn.preprocessing import normalize

########################################
# 1. RECOMENDACIÓN BASADA EN POPULARIDAD 
########################################

def recommend_by_popularity(user_id, games, interactions_df, k=10):
    """
    user_id. Int. Identificador del usuario.
    games. Df. Catálogo de juegos que incluye appid, name y positive_reviews.
    interactions_df. Df. Interacciones usuario-juego (TRAIN).
    k. Int. Número de juegos a recomendar.
    """

    user_id = int(user_id)

    # Juegos ya jugados por el usuario
    seen = set(
        interactions_df.loc[interactions_df["user_id"] == user_id, "appid"].tolist()
    )

    # Candidatos: juegos a los que NO ha jugado el usuario
    candidates = games[~games["appid"].isin(seen)].copy()

    # Puntuación de popularidad
    candidates["popularity_score"] = candidates["positive_reviews"].astype(float)

    # Ranking por popularidad
    recs = (
        candidates
        .sort_values("popularity_score", ascending=False)
        .head(k)[["appid", "name", "popularity_score"]]
        .reset_index(drop=True)
    )
    recs.insert(0, "rank", np.arange(1, len(recs) + 1))
    return recs



#######################################
# 2. RECOMENDACIÓN BASADA EN CONTENIDO 
#######################################

def recommend_by_content_user(
    user_id,
    user_train,
    U_norm,
    user_to_pos,
    X_combined,
    games,
    idx,
    top_n=10,
):
    """
    user_id. Int. Identificador del usuario.
    user_train. Df.Interacciones de entrenamiento [user_id, appid].
    U_norm. no usado (se mantiene por compatibilidad).
    user_to_pos. no usado (se mantiene por compatibilidad).
    X_combined. matriz (n_items x d). Representación combinada de contenido (estructural + texto).
    games. Df. Catálogo con metadatos de juegos.
    idx. Df. Debe contener, como mínimo, la columna 'appid' alineada con X_combined.
    top_n- Int. Número de juegos a recomendar.
    """

    user_id = int(user_id)

    # Juegos del usuario
    user_games_train = user_train.loc[user_train["user_id"] == user_id, "appid"].values
    if len(user_games_train) == 0:
        return pd.DataFrame()

    # Mapa appid -> índice de fila en X_combined
    appids_all = idx["appid"].values
    appid_to_row = {appid: i for i, appid in enumerate(appids_all)}

    row_indices = [appid_to_row[a] for a in user_games_train if a in appid_to_row]
    if len(row_indices) == 0:
        return pd.DataFrame()

    # Perfil de usuario = media de los vectores de los juegos consumidos (TRAIN)
    items_mat = X_combined[row_indices]

    if issparse(items_mat):
        # media sobre el eje 0 → (1, n_features), tipo np.matrix
        user_vec = items_mat.mean(axis=0)
        user_vec = np.asarray(user_vec)  # np.matrix -> ndarray (1, n_features)
    else:
        user_vec = items_mat.mean(axis=0, keepdims=True)

    # Normalizar L2
    user_vec = normalize(user_vec, norm="l2")

    # Similitud usuario–juego (1 x n_items)
    scores = cosine_similarity(user_vec, X_combined)[0]

    # Excluir juegos ya jugados en TRAIN
    mask_played = np.isin(appids_all, user_games_train)
    scores[mask_played] = -np.inf

    # Top-N recomendaciones
    n_items = len(appids_all)
    if top_n is None or top_n > n_items:
        top_n = n_items

    top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    appids_rec = appids_all[top_idx]

    recs = pd.DataFrame(
        {
            "appid": appids_rec,
            "score_content": scores[top_idx],
        }
    ).merge(
        games[["appid", "name", "free", "mac", "linux"]],
        on="appid",
        how="left",
    )

    return recs.sort_values("score_content", ascending=False).head(top_n).reset_index(drop=True)



#########################################
# 3. RECOMENDACIÓN BASADA EN COLABORATIVO 
#########################################

def recommend_by_collaborative(
    user_id,
    knn,
    R_norm,
    R,
    user_map,
    inv_game_map,
    idx,
    top_n=10,
    top_neighbors=50,
):
    """
    user_id. Int. Identificador del usuario objetivo.
    knn. sklearn.neighbors.NearestNeighbors entrenado sobre R_norm.
    R_norm. csr_matrix usuario-juego normalizada.
    R. csr_matrix usuario-juego original.
    user_map. dict user_id -> índice de fila en R/R_norm.
    inv_game_map. np.array índice -> appid.
    idx. DataFrame catálogo de juegos (['appid', 'name']).
    top_n. Int. Nº de juegos a recomendar (si None -> todos ordenados).
    top_neighbors. Nº de vecinos más similares a considerar.
    """

    user_id = int(user_id)

    if user_id not in user_map:
        # Usuario no presente en la matriz colaborativa
        return pd.DataFrame()

    u = user_map[user_id]

    # Vecinos más similares
    distances, indices = knn.kneighbors(R_norm[u], n_neighbors=min(top_neighbors + 1, R_norm.shape[0]))
    distances, indices = distances.ravel(), indices.ravel()

    # Excluir al propio usuario
    mask = indices != u
    neighbors = indices[mask]
    sims = 1.0 - distances[mask]   # distancia coseno -> similitud
    sims[sims < 0] = 0.0

    if sims.sum() <= 0:
        return pd.DataFrame()

    # Matriz de interacciones de vecinos (m x n)
    neighbor_matrix = R[neighbors].toarray()
    scores = np.dot(sims, neighbor_matrix)

    # Normalización por la suma de similitudes
    scores /= (sims.sum() + 1e-8)

    # Eliminamos juegos ya jugados por el usuario u
    played = R[u].indices
    scores[played] = -np.inf

    n_items = scores.shape[0]
    if top_n is None or top_n > n_items:
        top_n = n_items

    # Seleccionamos el top
    top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    appids = inv_game_map[top_idx]

    recs = pd.DataFrame({
        "appid": appids,
        "score": scores[top_idx],
    }).merge(
        idx, on="appid", how="left"
    )

    recs = recs.sort_values("score", ascending=False).reset_index(drop=True)
    return recs.head(top_n)


##########################
# 4. RECOMENDACIÓN HÍBRIDA
##########################

### A. NORMALIZADOR

def minmax_scale(x):
    x = np.asarray(x, dtype=float)

    # Si no hay valores finitos, devolvemos ceros
    if not np.isfinite(x).any():
        return np.zeros_like(x)

    mask = np.isfinite(x)
    lo, hi = np.min(x[mask]), np.max(x[mask])

    if hi > lo:
        x_scaled = (x - lo) / (hi - lo)
        x_scaled[~mask] = 0.0
        return x_scaled
    else:
        # todo igual
        return np.zeros_like(x)


### B. PUNTUACIONES CONTENIDO (vector completo)

def score_content_user(user_id, U_norm, user_to_pos, X_combined, user_train, idx):
    """
    user_id. Int. Identificador del usuario.
    U_norm. no usado (se mantiene por compatibilidad).
    user_to_pos. no usado (se mantiene por compatibilidad).
    X_combined. matriz (n_items x d). Representación combinada de contenido (estructural + texto).
    user_train. Df.Interacciones de entrenamiento [user_id, appid].
    idx. Df. Debe contener, como mínimo, la columna 'appid' alineada con X_combined.
    """
    from sklearn.preprocessing import normalize

    user_id = int(user_id)

    # Juegos del usuario en TRAIN
    user_games_train = user_train.loc[user_train["user_id"] == user_id, "appid"].values
    if len(user_games_train) == 0:
        # Sin historial → vector de ceros
        return np.zeros(X_combined.shape[0], dtype=float)

    # Mapa appid -> índice en X_combined
    appids_all = idx["appid"].values
    appid_to_row = {appid: i for i, appid in enumerate(appids_all)}

    row_indices = [appid_to_row[a] for a in user_games_train if a in appid_to_row]
    if len(row_indices) == 0:
        return np.zeros(X_combined.shape[0], dtype=float)

    items_mat = X_combined[row_indices]

    if issparse(items_mat):
        user_vec = items_mat.mean(axis=0)
        user_vec = np.asarray(user_vec)  # np.matrix -> ndarray
    else:
        user_vec = items_mat.mean(axis=0, keepdims=True)

    # Normalizar L2
    user_vec = normalize(user_vec, norm="l2")

    # Similitud usuario–juego
    scores = cosine_similarity(user_vec, X_combined)[0]
    return scores


### C. PUNTUACIONES COLABORATIVO (vector completo en orden inv_game_map)

def score_collab_user(user_id, R_norm, R, user_map, inv_game_map, knn, top_neighbors=50):
    """
    user_id. Int. Identificador del usuario objetivo.
    R_norm. csr_matrix usuario-juego normalizada.
    R. csr_matrix usuario-juego original.
    user_map. dict user_id -> índice de fila en R/R_norm.
    inv_game_map. np.array índice -> appid.
    knn. sklearn.neighbors.NearestNeighbors entrenado sobre R_norm.
    top_neighbors. Nº de vecinos más similares a considerar.
    """
    user_id = int(user_id)

    if user_id not in user_map:
        raise ValueError(f"[Colaborativo] Usuario {user_id} no existe en user_map")

    u_idx = user_map[user_id]

    distances, indices = knn.kneighbors(
        R_norm[u_idx],
        n_neighbors=min(top_neighbors + 1, R_norm.shape[0]),
    )
    distances = distances.ravel()
    indices = indices.ravel()

    # Excluimos al propio usuario
    mask = indices != u_idx
    neighbors = indices[mask]
    sims = 1.0 - distances[mask]
    sims[sims < 0] = 0.0

    if sims.sum() <= 0:
        return np.zeros(R.shape[1], dtype=float)

    neighbor_matrix = R[neighbors].toarray()
    scores = sims @ neighbor_matrix
    scores /= (sims.sum() + 1e-8)

    return scores  # shape (n_items_cf,)


### D. SISTEMA DE RECOMENDACIÓN HÍBRIDO

def recommend_by_hybrid(user_id, alpha, user_train, U_norm, user_to_pos, X_combined, R_norm, R, user_map, game_map, inv_game_map, games, idx, knn, top_n=10, top_neighbors=50,
):
    """
    user_id. Int. Identificador del usuario objetivo.
    alpha. Int. Valor de importancia de cada bloque.
    user_train. Df.Interacciones de entrenamiento [user_id, appid].
    U_norm. no usado (se mantiene por compatibilidad).
    user_to_pos. no usado (se mantiene por compatibilidad).
    X_combined. matriz (n_items x d). Representación combinada de contenido (estructural + texto).
    R_norm. csr_matrix usuario-juego normalizada.
    R. csr_matrix usuario-juego original.
    user_map. dict user_id -> índice de fila en R/R_norm.
    inv_game_map. np.array índice -> appid.
    games. Df. Catálogo de juegos que incluye appid, name y positive_reviews.
    idx. Df. Debe contener, como mínimo, la columna 'appid' alineada con X_combined.    
    knn. sklearn.neighbors.NearestNeighbors entrenado sobre R_norm.
    top_n. Int. Nº de juegos a recomendar (si None -> todos ordenados).
    top_neighbors. Nº de vecinos más similares a considerar.

    """

    assert 0.0 <= alpha <= 1.0
    user_id = int(user_id)

    # Scores de contenido 
    scores_content_raw = score_content_user(
        user_id=user_id,
        U_norm=U_norm,
        user_to_pos=user_to_pos,
        X_combined=X_combined,
        user_train=user_train,
        idx=idx,
    )
    # Scores colaborativos
    scores_cf_raw = score_collab_user(
        user_id=user_id,
        R_norm=R_norm,
        R=R,
        user_map=user_map,
        inv_game_map=inv_game_map,
        knn=knn,
        top_neighbors=top_neighbors,
    )

    # Reordenar scores_cf_raw al orden de idx["appid"]
    appids_all = idx["appid"].values
    n_items = len(appids_all)
    scores_cf_aligned = np.zeros(n_items, dtype=float)

    for i, appid in enumerate(appids_all):
        col_idx = game_map.get(int(appid))
        if col_idx is not None and 0 <= col_idx < len(scores_cf_raw):
            scores_cf_aligned[i] = scores_cf_raw[col_idx]
        else:
            scores_cf_aligned[i] = 0.0  # sin info colaborativa → 0

    # Normalizar ambos bloques
    scores_content = minmax_scale(scores_content_raw)
    scores_cf = minmax_scale(scores_cf_aligned)

    # Score híbrido lineal
    score_hybrid = alpha * scores_content + (1.0 - alpha) * scores_cf

    # Excluir juegos ya jugados
    played_train = set(
        user_train.loc[user_train["user_id"] == user_id, "appid"].values
    )
    mask_played = np.array([appid in played_train for appid in appids_all])
    score_hybrid[mask_played] = -np.inf

    # Top-N
    if top_n is None or top_n > n_items:
        top_n = n_items

    top_idx = np.argpartition(-score_hybrid, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-score_hybrid[top_idx])]

    appids_rec = appids_all[top_idx]

    recs = pd.DataFrame({
        "appid": appids_rec,
        "score_hybrid": score_hybrid[top_idx],
    }).merge(
        games[["appid", "name"]],
        on="appid",
        how="left",
    )

    return recs.reset_index(drop=True).head(top_n)