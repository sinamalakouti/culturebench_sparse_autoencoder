# Activity prompts dictionary

ACTIVITY_PROMPTS = {
    "eating_restaurant": "people eating in a restaurant",
    "eating_home": "a family eating at home",
    "greeting_two_people": "two people greeting each other",
    "greeting_one_man_and_one_woman": "a man and a woman greeting each other",
    "greeting_two_woman": "two women greeting each other",
    "dancing_traditional_dance": "people dancing",
    "concert_indoor": "an indoor concert",
    "concert_outdoor": "an outdoor concert",
    "games_soccer_street": "friends playing soccer in the street",
    "games_home": "friends playing games at home",
    "games_local_sport": "friends playing a local sport",
    "funeral_funeral": "a funeral ceremony",
    "wedding_wedding": "a wedding ceremony",
    "new_year_new_year": "a new year's celebration",
    "religious_religious_ceremony": "a religious ceremony",
    "religious_praying": "people praying",
    "engagement_proposal": "an engagement proposal",
}

PREFIX_PROMPT = "A photorealistic photo of "


# ACTIVITY_PROMPTS = {
#     "eating_restaurant": "A photo of people eating food in a restaurant",
#     "eating_home": "A photo of a family enjoying food at home",
#     "greeting_two_man": "A photo capturing greeting rituals between two people",
#     "greeting_one_man_and_one_woman": "A photo capturing greeting rituals between one man and one woman",
#     "greeting_two_woman": "A photo capturing greeting rituals between two women",
#     "dancing_traditional_dance": "A photo of people performing dance",
#     "concert_indoor": "A photo of people attending a concert in an indoor venue",
#     "concert_outdoor": "A photo of people attending a concert in an outdoor venue",
#     "games_soccer_street": "A photo of friends playing soccer in streets",
#     "games_home": "A photo of friends playing games at home",
#     "games_local_sport": "A photo of friends playing a popular local sport",
#     "funeral_funeral": "A photo depicting a funeral ceremony",
#     "wedding_wedding": "A photo of a wedding ceremony",
#     "wedding_engagement": "A photo of a wedding engagement ceremony",
#     "new_year_new_year": "A photo of New Year's celebration",
#     "religious_religious_ceremony": "A photo of people participating in an important religious ceremony",
#     "religious_praying": "A photo showing religious worship or prayer in a typical place of worship",
#     "engagement_proposal": "A photo of a traditional engagement or proposal ceremony",
#     "praying_prayer": "A photo of people praying",
# }


# Mapping of main categories to their subcategories
ACTIVITY_TO_SUBACTIVITIES = {
    "eating": ["restaurant", "home"],
    "greeting": ["two_people", "one_man_and_one_woman", "two_woman"],
    "dancing": ["traditional_dance"],
    "concert": ["indoor", "outdoor"],
    "games": ["soccer_street", "home", "local_sport"],
    "funeral": ["funeral"],
    "wedding": ["wedding"],
    "new_year": ["new_year"],
    "religious": ["religious_ceremony", "praying"],
    "engagement": ["proposal"],
}

COUNTRIES = [
    "IRAN",
    "NIGERIA",
    "FRANCE",
    "USA",
    "CHINA",
    "INDIA",
    "BRAZIL",
    "INDONESIA",
    "MEXICO",
    "TURKEY",
    "NEPAL",
    "PHILIPPINES",
    "SOUTH_AFRICA",
    "GERMANY",
    "SPAIN",
    "ITALY",
]

GLOBAL_SOUTH_COUNTRIES = [
    "IRAN",
    "NIGERIA",
    "CHINA",
    "INDIA",
    "BRAZIL",
    "INDONESIA",
    "MEXICO",
    "TURKEY",
    "NEPAL",
    "PHILIPPINES",
    "SOUTH_AFRICA",
]
GLOBAL_NORTH_COUNTRIES = [
    "FRANCE",
    "USA",
    "GERMANY",
    "SPAIN",
    "ITALY",
]


list_of_activities = [
    "eating",
    "greeting",
    "dancing",
    "concert",
    "games",
    "funeral",
    "wedding",
    "celebrations",
    "religious",
]
