from tabulate import tabulate

# Rating equivalency
ratings = [dict(zip(
    ["Moodys", "Fitch", "S&P", "FitchFactor", "MoodysFactor"], x)) for x in [
    ["Caa1",   "CCC+",  "CCC+",   62.8,         4770],
    ["Caa2",   "CCC",   "CCC",    62.8,         6500],
    ["Caa3",   "CCC-",  "CCC-",   62.8,         8070],
    ["Ca",     "CC",    "CC",    100,          10000],
    ["Ca",     "C",     "C",     100,          10000],
    ["C",      "DDD",   "RD",    100,          10000],
    [None,     "DD",    "SD",    100,           None],
    [None,     "D",     "D",     100,           None],
    ["NR",     "NR",    "NR",   None,           None],
    ["NA",     "NA",    "NA",   None,           None],
]]

print(tabulate(ratings, headers="keys", tablefmt="simple"))
