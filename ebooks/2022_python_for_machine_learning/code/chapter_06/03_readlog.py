import urllib.request
import re

# Read the log file, split into lines
logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()

# using regular expression to extract IP address and status code from a line
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

print(ip_and_code(lines[0]))
