import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

# Launch Chrome browser in headless mode
options = webdriver.ChromeOptions()
options.add_argument("headless")
browser = webdriver.Chrome(options=options)

# Load web page
browser.get("https://www.yahoo.com")
# Network transport takes time. Wait until the page is fully loaded
def is_ready(browser):
    return browser.execute_script(r"""
        return document.readyState === 'complete'
    """)
WebDriverWait(browser, 30).until(is_ready)

# Scroll to bottom of the page to trigger JavaScript action
browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(1)
WebDriverWait(browser, 30).until(is_ready)

# Search for news headlines and print
elements = browser.find_elements(By.XPATH, "//h3/a[u[@class='StretchedBox']]")
for elem in elements:
    print(elem.text)

# Close the browser once finish
browser.close()
