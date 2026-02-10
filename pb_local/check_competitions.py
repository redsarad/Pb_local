import penaltyblog as pb

print("FootballData:")
print(pb.scrapers.FootballData.list_competitions())

print("\nFBRef:")
print(pb.scrapers.FBRef.list_competitions())

print("\nUnderstat:")
print(pb.scrapers.Understat.list_competitions())
