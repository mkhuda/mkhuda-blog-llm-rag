def chroma_where_from_filters(filters: dict):
    """Translate filters from reasoning JSON into Chroma 'where' clause."""
    where = {}
    if not filters:
        return where
    date_range = filters.get("date")
    if date_range:
        where["date"] = {}
        if "$gte" in date_range:
            where["date"]["$gte"] = date_range["$gte"]
        if "$lt" in date_range:
            where["date"]["$lt"] = date_range["$lt"]
    return where