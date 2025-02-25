from datetime import datetime
import pytz
from flask import Flask

app = Flask("time now")

@app.route("/now/<path:timezone>")
def timenow(timezone):
    try:
        zone = pytz.timezone(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %z %Z\n")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown time zone: {timezone}\n"

app.run()
