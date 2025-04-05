def format_value(x, format_str):
    try:
        return format_str % x
    except Exception:
        return str(x)
