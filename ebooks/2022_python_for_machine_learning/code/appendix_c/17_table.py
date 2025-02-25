# Rating equivalency
ratings = [dict(zip(
    ["Moodys", "Fitch", "S&P", "FitchFactor", "MoodysFactor"], x)) for x in [
    ["Aaa",    "AAA",   "AAA",     0.19,           1],
    ["Aa1",    "AA+",   "AA+",     0.35,          10],
    ["Aa2",    "AA",    "AA",      0.64,          20],
    ["Aa3",    "AA-",   "AA-",     0.86,          40],
    ["A1",     "A+",    "A+",      1.17,          70],
    ["A2",     "A",     "A",       1.58,         120],
    ["A3",     "A-",    "A-",      2.25,         180],
    ["Baa1",   "BBB+",  "BBB+",    3.19,         260],
    ["Baa2",   "BBB",   "BBB",     4.54,         360],
    ["Baa3",   "BBB-",  "BBB-",    7.13,         610],
    ["Ba1",    "BB+",   "BB+",    12.19,         940],
    ["Ba2",    "BB",    "BB",     17.43,        1350],
    ["Ba3",    "BB-",   "BB-",    22.8,         1766],
    ["B1",     "B+",    "B+",     27.8,         2220],
    ["B2",     "B",     "B",      27.8,         2720],
    ["B3",     "B-",    "B-",     32.18,        3490],
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

print(ratings[0])
# {'Moodys': 'Aaa', 'Fitch': 'AAA', 'S&P': 'AAA', 'FitchFactor': 0.19, 'MoodysFactor': 1}
