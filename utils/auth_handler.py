import hashlib
import json
import os
import secrets

from utils.path_tool import get_abs_path


USER_STORE_PATH = get_abs_path("data/users.json")


def _ensure_user_store():
    os.makedirs(os.path.dirname(USER_STORE_PATH), exist_ok=True)
    if not os.path.exists(USER_STORE_PATH):
        with open(USER_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


def _load_users() -> dict:
    _ensure_user_store()
    with open(USER_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    _ensure_user_store()
    with open(USER_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def register_user(user_id: str, password: str):
    user_id = user_id.strip()

    if not user_id:
        raise ValueError("用户ID不能为空")

    if not password:
        raise ValueError("密码不能为空")

    users = _load_users()
    if user_id in users:
        raise ValueError("该用户ID已存在，请直接登录")

    salt = secrets.token_hex(16)
    users[user_id] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)


def verify_user(user_id: str, password: str) -> bool:
    user_id = user_id.strip()
    users = _load_users()
    user = users.get(user_id)

    if not user:
        return False

    password_hash = _hash_password(password, user["salt"])
    return secrets.compare_digest(password_hash, user["password_hash"])
