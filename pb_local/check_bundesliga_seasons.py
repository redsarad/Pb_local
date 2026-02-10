import penaltyblog as pb

seasons_to_try = [
    "2025-2026",
    "2024-2025",
    "2023-2024",
    "2022-2023",
]

for s in seasons_to_try:
    try:
        fb = pb.scrapers.FootballData("DEU Bundesliga 1", s)
        df = fb.get_fixtures()
        print(s, "OK - matches:", len(df), "| date range:", df["date"].min(), "to", df["date"].max())
    except Exception as e:
        print(s, "FAILED:", type(e).__name__, "-", e)
