# requires `pip install python-dateutil`
import dateutil.parser

d = dateutil.parser.parse("13 May 2020")
print(d)
# datetime.datetime(2020, 5, 13, 0, 0)
