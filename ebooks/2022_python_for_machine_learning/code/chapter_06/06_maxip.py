import collections
import urllib.request
import re

# using regular expression to extract IP address and status code from a line
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"

print(
    max(
        [(count,ip) for ip,count in
            collections.Counter([
                ip for ip, code in
                [ip_and_code(x) for x in
                     urllib.request.urlopen(logurl)
                     .read()
                     .decode("utf8")
                     .splitlines()
                ]
                if code=="404"
            ]).items()
        ]
    )
)
