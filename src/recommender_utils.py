# Importamos las librerias
import requests
import sqlite3
import os
import time

# Definimos los path que vamos a utilizar
db_path = "../data/steam.db"
appreviews_url = "https://store.steampowered.com/appreviews/{appid}?json=1&language=all&filter=all"
appreviews_users_url = "https://store.steampowered.com/appreviews/{appid}?json=1&language=all&filter=all&num_per_page=50"
api_url_owned = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"


# **************************************************
# 1. TABLA GAMES -- ids
# **************************************************

# 1A. Crear las tablas game y progress
def init_db(db_path="data/steam.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS games (
            appid INTEGER PRIMARY KEY,
            name TEXT
        )
    """)
    # Esta tabla nos sirve en caso de que paremos el progreso.
    c.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY,
            last_index INTEGER
        )
    """)

    c.execute("INSERT OR IGNORE INTO progress (id, last_index) VALUES (1, 0)")
    conn.commit()
    conn.close()

# 1B. Obtener listado de aplicaciones (games y más)
def get_all_apps():
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data["applist"]["apps"]

# 1C. Comprobar si una app es realmente un juego

def is_game(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data and str(appid) in data and data[str(appid)]["success"]:
            return data[str(appid)]["data"].get("type") == "game"
    except:
        return False
    return False

# 1D. Guardar los games en la tabla games
def save_to_db(games, db_path="data/steam.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executemany("INSERT OR IGNORE INTO games (appid, name) VALUES (?, ?)", games)
    conn.commit()
    conn.close()

# 1E. Guardar los appId procesados en la tabla progress
def update_progress(idx, db_path="data/steam.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("UPDATE progress SET last_index=? WHERE id=1", (idx,))
    conn.commit()
    conn.close()

# 1F. Recuperar el último índice procesado para reanudar el proceso
def get_last_progress(db_path="data/steam.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT last_index FROM progress WHERE id=1")
    last_idx = c.fetchone()[0]
    conn.close()
    return last_idx

# 1E. Guardado final de los games en la tabla game
def game_main(db_path="data/steam.db", save_every=100):
    init_db(db_path)                              # 1A
    all_apps = get_all_apps()                     # 1B
    last_idx = get_last_progress(db_path)         # 1F

    print(f"Total apps en Steam: {len(all_apps)}")
    print(f"Reanudando desde índice: {last_idx}")

    games = []
    for i, app in enumerate(all_apps[last_idx:], start=last_idx):
        appid, name = app["appid"], app["name"]
        if not name:
            continue

        if is_game(appid):            # 1C
            games.append((appid, name))

        # Guardado incremental
        if len(games) >= save_every:
            save_to_db(games, db_path)   # 1D
            update_progress(i, db_path)  # 1E
            print(f"[{time.strftime('%H:%M:%S')}] Procesados {i} apps, {len(games)} juegos guardados")
            games = []

    # Guardar lo que quede
    if games:
        save_to_db(games, db_path)                # 1D
        update_progress(len(all_apps), db_path)   # 1E

    print("Proceso completado ✅")


# **************************************************
# 2. TABLA GAMES -- reviews
# **************************************************

# 2A. Recuperar número reviews
def get_reviews(appid):
    """
    Devuelve (total_positive, total_reviews) para un appid de Steam.
    Si falla devuelve (None, None).
    """
    try:
        resp = requests.get(appreviews_url.format(appid=appid), timeout=10)
        data = resp.json()
        summary = data.get("query_summary", {})
        total_positive = summary.get("total_positive", 0)
        total_reviews = summary.get("total_reviews", 0)
        return total_positive, total_reviews
    except Exception as e:
        print(f"⚠️ Error con appid {appid}: {e}")
        return None, None
    
# 2B. Guardar reviews y ratio

def save_reviews(db_path=db_path, save_every=50, sleep_time=1):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Asegurar columnas
    cur.execute("PRAGMA table_info(games)")
    existing_cols = [col[1] for col in cur.fetchall()]

    if "positive_reviews" not in existing_cols:
        cur.execute("ALTER TABLE games ADD COLUMN positive_reviews INTEGER")
    if "total_reviews" not in existing_cols:
        cur.execute("ALTER TABLE games ADD COLUMN total_reviews INTEGER")
    if "score_reviews" not in existing_cols:
        cur.execute("ALTER TABLE games ADD COLUMN score_reviews REAL")

    con.commit()

    # Obtener todos los appids
    cur.execute("SELECT appid FROM games WHERE total_reviews IS NULL")
    appids = [row[0] for row in cur.fetchall()]

    print(f"📊 Encontrados {len(appids)} appids pendientes de actualizar.")

    for i, appid in enumerate(appids, start=1):
        total_positive, total_reviews = get_reviews(appid)   # 2A

        score_reviews = (
            total_positive / total_reviews
            if total_positive is not None and total_reviews and total_reviews > 0
            else None
        )

        cur.execute("""
            UPDATE games
            SET positive_reviews = ?,
                total_reviews = ?,
                score_reviews = ?
            WHERE appid = ?
        """, (total_positive, total_reviews, score_reviews, appid))

        # Commit en lotes
        if i % save_every == 0:
            con.commit()
            print(f"✅ Guardados {i} appids...")

        # Pausa para respetar rate limit
        time.sleep(sleep_time)

    con.commit()
    con.close()
    print("🎉 Proceso terminado.")


# **************************************************
# 3. TABLA GAMES -- información games
# **************************************************

# 3A. Obtener información de los game de games
def get_appdetails(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if not data or str(appid) not in data:
            return None
        entry = data[str(appid)]
        if not entry.get("success"):
            return None
        return entry.get("data", None)  # <- si no hay "data", devuelve None
    except Exception as e:
        print(f"⚠️ Error appid {appid}: {e}")
        return None

# 3B.Guardamos la información en games   
def save_appdetails(db_path=db_path, save_every=50, sleep_time=1):
    """
    Enriquecer tabla games con columnas de appdetails.
    Guarda cambios cada `save_every` juegos y permite reanudar.
    """
    con = sqlite3.connect(db_path, timeout=20)
    cur = con.cursor()

    # Asegurar columnas necesarias
    cur.execute("PRAGMA table_info(games)")
    existing_cols = [col[1] for col in cur.fetchall()]

    new_columns = {
        "free": "BOOLEAN",
        "description": "TEXT",
        "average_forever": "INTEGER",
        "median_forever": "INTEGER",
        "developers": "TEXT",
        "publishers": "TEXT",
        "windows": "BOOLEAN",
        "mac": "BOOLEAN",
        "linux": "BOOLEAN",
        "release_date": "TEXT",
        "content_descriptors_notes": "TEXT",
        "multijugador": "BOOLEAN",
        "cooperativo": "BOOLEAN",
        "competitivo": "BOOLEAN",
        "multidispositivo": "BOOLEAN",
        # géneros
        "genre_action": "BOOLEAN",
        "genre_sports": "BOOLEAN",
        "genre_strategy": "BOOLEAN",
        "genre_indie": "BOOLEAN",
        "genre_adventure": "BOOLEAN",
        "genre_simulation": "BOOLEAN",
        "genre_mmo": "BOOLEAN",
        "genre_rpg": "BOOLEAN",
        "genre_free": "BOOLEAN",
        "genre_casual": "BOOLEAN",
        "genre_accounting": "BOOLEAN",
        "genre_animation": "BOOLEAN",
        "genre_audio": "BOOLEAN",
        "genre_design": "BOOLEAN",
        "genre_education": "BOOLEAN",
        "genre_photo": "BOOLEAN",
        "genre_training": "BOOLEAN",
        "genre_utilities": "BOOLEAN",
        "genre_video": "BOOLEAN",
        "genre_web": "BOOLEAN",
        "genre_dev": "BOOLEAN",
        "genre_early": "BOOLEAN",
        "genre_sexual": "BOOLEAN",
        "genre_nudity": "BOOLEAN",
        "genre_violent": "BOOLEAN",
        "genre_gore": "BOOLEAN",
        "genre_movie": "BOOLEAN",
        "genre_doc": "BOOLEAN",
        "genre_eposodic": "BOOLEAN",
        "genre_short": "BOOLEAN",
        "genre_tutorial": "BOOLEAN",
        "genre_360": "BOOLEAN",
        "genre_racing": "BOOLEAN",
    }

    for col, ctype in new_columns.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE games ADD COLUMN {col} {ctype}")
    con.commit()

    # Seleccionar juegos sin enriquecer aún
    cur.execute("SELECT appid FROM games WHERE free IS NULL")
    appids = [row[0] for row in cur.fetchall()]

    print(f"📊 Encontrados {len(appids)} juegos pendientes de actualizar.")

    batch = 0
    for i, appid in enumerate(appids, start=1):
        details = get_appdetails(appid)    # 3A
        if not details:
            continue

        # Variables simples
        free = details.get("is_free")
        description = details.get("detailed_description")
        avg = details.get("average_forever")
        med = details.get("median_forever")
        devs = ",".join(details.get("developers", []))
        pubs = ",".join(details.get("publishers", []))

        windows = details.get("platforms", {}).get("windows")
        mac = details.get("platforms", {}).get("mac")
        linux = details.get("platforms", {}).get("linux")

        release_date = details.get("release_date", {}).get("date")
        content_notes = details.get("content_descriptors", {}).get("notes")

        # Categorías → banderas
        cats = [c["id"] for c in details.get("categories", [])]
        multijugador = any(c in [1,9,24,27,36,38,39,44,47,48,49] for c in cats)
        cooperativo = any(c in [9,38,39,48] for c in cats)
        competitivo = any(c in [36,47,49] for c in cats)
        multidispositivo = any(c in [40,41,42,43] for c in cats)

        # Géneros → inicializar en False
        genre_flags = {col: False for col in new_columns if col.startswith("genre_")}
        genre_map = {
            1: "genre_action", 18: "genre_sports", 2: "genre_strategy", 23: "genre_indie",
            25: "genre_adventure", 28: "genre_simulation", 29: "genre_mmo", 3: "genre_rpg",
            37: "genre_free", 4: "genre_casual", 50: "genre_accounting", 51: "genre_animation",
            52: "genre_audio", 53: "genre_design", 54: "genre_education", 55: "genre_photo",
            56: "genre_training", 57: "genre_utilities", 58: "genre_video", 59: "genre_web",
            60: "genre_dev", 70: "genre_early", 71: "genre_sexual", 72: "genre_nudity",
            73: "genre_violent", 74: "genre_gore", 80: "genre_movie", 81: "genre_doc",
            82: "genre_eposodic", 83: "genre_short", 84: "genre_tutorial", 85: "genre_360",
            9: "genre_racing"
        }

        for g in details.get("genres", []):
            try:
                gid = int(g["id"])  # 🔑 aseguramos que sea int
                if gid in genre_map:
                    genre_flags[genre_map[gid]] = True
            except:
                continue

        # Construir UPDATE dinámico
        cur.execute(f"""
            UPDATE games SET
                free=?, description=?, average_forever=?, median_forever=?,
                developers=?, publishers=?, windows=?, mac=?, linux=?,
                release_date=?, content_descriptors_notes=?,
                multijugador=?, cooperativo=?, competitivo=?, multidispositivo=?,
                {','.join(f"{k}=?" for k in genre_flags.keys())}
            WHERE appid=?
        """, (
            free, description, avg, med, devs, pubs, windows, mac, linux,
            release_date, content_notes,
            multijugador, cooperativo, competitivo, multidispositivo,
            *genre_flags.values(), appid
        ))

        batch += 1
        if batch % save_every == 0:
            con.commit()
            print(f"✅ Guardados {batch} juegos...")
            time.sleep(sleep_time)

    con.commit()
    con.close()
    print("🎉 Proceso terminado.")

# **************************************************
# 4. TABLAS USERS Y USER_REVIEWS 
# **************************************************

# 4A. Creamos la tabla users y user_review
def init_user_tables(con):
    """
    Crea las tablas users y user_reviews si no existen.
    """
    cur = con.cursor()
    # Tabla de usuarios
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY
        )
    """)
    # Relación entre juegos y usuarios
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_reviews (
            appid INTEGER,
            user_id TEXT,
            PRIMARY KEY (appid, user_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (appid) REFERENCES games(appid)
        )
    """)
    con.commit()


# 4B. Guardar el user_id de usuarios que hayan dejado reseñas (hasta 50 usuarios por juegos). 50 por usuario(ver url)
def get_users_from_reviews(appid, retries=3, timeout=20):
    url = appreviews_users_url.format(appid=appid)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            data = resp.json()
            if "reviews" in data:
                return [rev["author"]["steamid"] for rev in data["reviews"]]
            return []
        except Exception as e:
            print(f"⚠️ Error con appid {appid} (intento {attempt}/{retries}): {e}")
            time.sleep(5)
    return []


# 4C. Guardar los usuarios y sus reviews
def populate_users(db_path=db_path, batch_size=20, sleep_time=2):

    con = sqlite3.connect(db_path)
    init_user_tables(con)   # 4A
    cur = con.cursor()

    # Buscar juegos que aún no tienen usuarios asociados en user_reviews
    cur.execute("""
        SELECT g.appid
        FROM games g
        WHERE NOT EXISTS (
            SELECT 1 FROM user_reviews ur WHERE ur.appid = g.appid
        )
    """)
    appids = [row[0] for row in cur.fetchall()]

    print(f"📊 {len(appids)} juegos pendientes de extraer usuarios.")

    for i, appid in enumerate(appids, start=1):
        users = get_users_from_reviews(appid)   # 4B

        # Insertar usuarios y relaciones
        for u in users:
            cur.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (u,))
            cur.execute("INSERT OR IGNORE INTO user_reviews (appid, user_id) VALUES (?, ?)", (appid, u))

        if i % batch_size == 0:
            con.commit()
            print(f"✅ Guardados {i} juegos...")

        time.sleep(sleep_time)

    con.commit()
    con.close()
    print("🎉 Proceso terminado.")


# **************************************************
# 5. TABLAS USERS Y USER_GAMES 
# **************************************************

# 5A. Crear la tabla user_games (usuarios y juegos jugados)
def init_user_games_table(con):
    """
    Crea la tabla user_games si no existe.
    """
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_games (
            user_id TEXT,
            appid INTEGER,
            PRIMARY KEY (user_id, appid),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (appid) REFERENCES games(appid)
        )
    """)
    con.commit()

# 5B.Obtener los juegos jugados por cada usuario
def get_owned_games(user_id, api_key, retries=3, timeout=20):
    """
    Devuelve la lista de appids de los juegos que posee un usuario.
    """
    params = {
        "key": api_key,
        "steamid": user_id,
        "include_appinfo": 0,
        "format": "json"
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(api_url_owned, params=params, timeout=timeout)
            data = resp.json()
            games = data.get("response", {}).get("games", [])
            return [g["appid"] for g in games]
        except Exception as e:
            print(f"⚠️ Error con user {user_id} (intento {attempt}/{retries}): {e}")
            time.sleep(5)
    return []

# Guardar la información en user_games
def populate_user_games(api_key, db_path="../data/steam.db", batch_size=50, sleep_time=2):
    """
    Para cada usuario en la tabla users, consulta los juegos que posee
    y los cruza con games. Guarda en user_games.
    Guarda cada `batch_size` usuarios y retoma desde donde se quedó.
    """
    con = sqlite3.connect(db_path)
    init_user_games_table(con)   # 5A
    cur = con.cursor()

    # Obtener usuarios que aún no han sido procesados en user_games
    cur.execute("""
        SELECT u.user_id
        FROM users u
        WHERE NOT EXISTS (
            SELECT 1 FROM user_games ug WHERE ug.user_id = u.user_id
        )
    """)
    users = [row[0] for row in cur.fetchall()]

    # Obtener lista de appids válidos de la tabla games
    cur.execute("SELECT appid FROM games")
    valid_appids = {row[0] for row in cur.fetchall()}

    print(f"📊 {len(users)} usuarios pendientes de procesar.")

    for i, user_id in enumerate(users, start=1):
        owned = get_owned_games(user_id, api_key)   # 5B

        # Filtrar solo juegos que estén en la tabla games
        owned_filtered = [appid for appid in owned if appid in valid_appids]

        # Insertar relaciones
        for appid in owned_filtered:
            cur.execute(
                "INSERT OR IGNORE INTO user_games (user_id, appid) VALUES (?, ?)",
                (user_id, appid)
            )

        if i % batch_size == 0:
            con.commit()
            print(f"✅ Guardados {i} usuarios...")

        time.sleep(sleep_time)

    con.commit()
    con.close()
    print("🎉 Proceso terminado.")
