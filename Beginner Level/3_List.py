# --------------------------------------------------------
# 3.List
print('---------- 3. List ----------')
# --------------------------------------------------------

justice_league = ["Superman", "Batman", "Wonder Woman", "Flash", "Aquaman", "Green Lantern"]
print(len(justice_league))

# Appending the new team members
justice_league.append("Batgirl")
justice_league.append("NightWing")
print(justice_league)

# Changing wonder woman as the leader, So woman moved to starting of the list.
justice_league.remove("Wonder Woman")
justice_league.insert(0, 'Wonder Woman')
print(justice_league)

# Moving Superman in between of Aquaman and Flash.
justice_league.remove('Superman')
justice_league.insert(3, 'Superman')
print(justice_league)

# Change the new justice league
justice_league = ["Cyborg", "Shazam", "Hawkgirl", "Martian Manhunter", "Green Arrow"]
print(justice_league)

# Sorting the justice league alphabetically
justice_league = sorted(justice_league)
print(justice_league)
print(f'The leader of the justice_league is the first position of the list is {justice_league[0]}')
