from curl_cffi import request
import time
from fastapi import HTTPException
from functools import wraps

def rate_limiter(redis_client, key: str, limit: int, window_sec: int):
    now = int(time.time())
    window_key = f"rate_limit:{key}:{now // window_sec}"

    count = redis_client.incr(window_key)
    redis_client.expire(window_key, window_sec)

    if count > limit:
        raise HTTPException(status_code=429, detail="Too many requests")

def rate_limit_decorator(limit: int = 10, window_sec: int = 60, key_prefix: str = ""):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):

            from Backend.state import Redis_client

            if Redis_client:
                try:
                    key = key_prefix

                    if "request" in kwargs:
                        try:
                            req_data = await kwargs["request"].json()
                            if isinstance(req_data, dict) and "ticker" in req_data:
                                key = f"{key_prefix}:{req_data['ticker'].strip().upper()}"
                        except Exception:
                            pass

                    rate_limiter(Redis_client, key, limit, window_sec)
                except Exception:
                    raise
            return await func(*args, **kwargs)
        return wrapper
    return decorator