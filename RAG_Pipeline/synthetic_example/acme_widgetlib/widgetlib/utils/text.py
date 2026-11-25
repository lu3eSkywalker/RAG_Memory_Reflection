"""Text helpers used across acme_widgetlib."""

def normalize_sku(raw):
    """Return a canonical SKU string.

    Strips whitespace, upper-cases, and prepends SKU_PREFIX when missing.
    """
    from widgetlib.config.defaults import SKU_PREFIX
    value = (raw or "").strip().upper()
    if not value.startswith(SKU_PREFIX):
        value = SKU_PREFIX + value
    return value


def slugify_label(label):
    """Lower-case, replace spaces with underscores, drop non-alnum."""
    out = []
    for ch in (label or "").lower():
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")