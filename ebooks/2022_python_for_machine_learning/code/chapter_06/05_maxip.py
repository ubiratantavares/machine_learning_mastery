import collections
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

def is404(pair):
    return pair[1] == "404"
def getIP(pair):
    return pair[0]
def count_ip(count_item):
    ip, count = count_item
    return (count, ip)

ipcodepairs = [ip_and_code(x) for x in lines]
ip404 = [ip for ip,code in ipcodepairs if code=="404"]
ipcount = collections.Counter(ip404)
countip = [(count,ip) for ip,count in ipcount.items()]
print(max(countip))
